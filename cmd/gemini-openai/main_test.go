package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bojackduy/gemini-proxy/internal/gemini"
)

// fakeBackend answers Ask without a browser.
type fakeBackend struct{}

func TestReadyBusyAndPendingAreNotAuthFailures(t *testing.T) {
	newHandler := func(client browserClient, sessions map[string]*session) *handler {
		h := &handler{model: "gemini-web", client: client, gate: make(chan struct{}, 1), sessions: sessions}
		h.gate <- struct{}{}
		return h
	}
	readyz := func(h *handler, sid string) *httptest.ResponseRecorder {
		r := httptest.NewRequest("GET", "/readyz", nil)
		if sid != "" {
			r.Header.Set("x-opencode-session-id", sid)
		}
		w := httptest.NewRecorder()
		h.routes().ServeHTTP(w, r)
		return w
	}

	t.Run("busy gate is unavailable, never an auth failure", func(t *testing.T) {
		h := &handler{model: "gemini-web", client: &fakeReadyBackend{}, gate: make(chan struct{}, 1), sessions: map[string]*session{}}
		// Empty gate: a browser turn holds it.
		w := readyz(h, "")
		if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), "browser turn busy") || strings.Contains(w.Body.String(), "not logged in") {
			t.Fatalf("busy: %d %s", w.Code, w.Body.String())
		}
	})

	t.Run("one pending session does not report the provider unready", func(t *testing.T) {
		h := newHandler(&fakeReadyBackend{}, map[string]*session{"stuck": {Pending: true}})
		w := readyz(h, "")
		if w.Code != http.StatusOK {
			t.Fatalf("a pending session must not brick the provider: %d %s", w.Code, w.Body.String())
		}
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if body["ready"] != true || body["pending"] != float64(1) || !strings.Contains(w.Body.String(), "stuck") {
			t.Fatalf("readiness must count and name the pending identities: %s", w.Body.String())
		}
	})

	t.Run("the pending identity itself reports not-ready", func(t *testing.T) {
		h := newHandler(&fakeReadyBackend{}, map[string]*session{"stuck": {Pending: true}, "live": {}})
		w := readyz(h, "stuck")
		if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), "pending uncertain turn; inspect owned chat") {
			t.Fatalf("pending identity: %d %s", w.Code, w.Body.String())
		}
		if strings.Contains(w.Body.String(), "not logged in") {
			t.Fatalf("pending must never be reported as an auth failure: %s", w.Body.String())
		}
		if w := readyz(h, "live"); w.Code != http.StatusOK {
			t.Fatalf("a clean identity stays ready while another is pending: %d %s", w.Code, w.Body.String())
		}
		// Journal truth outranks a transient busy gate.
		busy := newHandler(&fakeReadyBackend{}, map[string]*session{"stuck": {Pending: true}})
		<-busy.gate
		if w := readyz(busy, "stuck"); w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), "pending uncertain turn") {
			t.Fatalf("pending identity must beat a busy gate: %d %s", w.Code, w.Body.String())
		}
	})

	t.Run("a logged-out tab still discloses the pending count", func(t *testing.T) {
		h := newHandler(&fakeRecoveryBackend{}, map[string]*session{"stuck": {Pending: true}})
		w := readyz(h, "")
		if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), `"pending":1`) {
			t.Fatalf("not-ready must never hide pending sessions: %d %s", w.Code, w.Body.String())
		}
	})
}

func (f *fakeBackend) Ask(_ interface{}, _ string, _ bool, _ interface{}) (string, error) {
	return "", nil
}

func TestSessionIDPriority(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-Session-Affinity", "aff")
	r.Header.Set("X-Opencode-Session", "opencode")
	r.Header.Set("X-Session-Id", "primary")
	if got := sessionID(r); got != "primary" {
		t.Fatalf("priority: got %q", got)
	}
	r2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r2.Header.Set("X-Session-Affinity", "aff")
	if got := sessionID(r2); got != "aff" {
		t.Fatalf("fallback: got %q", got)
	}
}

func TestCanonicalEqualIgnoresArgEncoding(t *testing.T) {
	a := message{Role: "assistant", ToolCalls: []toolCall{{ID: "1", Function: callFunction{Name: "f", Arguments: `{"a":1,"b":2}`}}}}
	b := message{Role: "assistant", ToolCalls: []toolCall{{ID: "1", Function: callFunction{Name: "f", Arguments: `{"b":2,"a":1}`}}}}
	if !canonicalEqual(a, b) {
		t.Fatal("equivalent argument encodings should compare equal")
	}
	c := message{Role: "assistant", ToolCalls: []toolCall{{ID: "1", Function: callFunction{Name: "f", Arguments: `{"a":1}`}}}}
	if canonicalEqual(a, c) {
		t.Fatal("different arguments must not compare equal")
	}
}

func TestEffectiveToolChoice(t *testing.T) {
	if got := effectiveToolChoice(nil, 0); got != "none" {
		t.Fatalf("no tools: %q", got)
	}
	if got := effectiveToolChoice(nil, 2); got != "auto" {
		t.Fatalf("nil choice: %q", got)
	}
	if got := effectiveToolChoice("required", 1); got != "required" {
		t.Fatalf("string choice: %q", got)
	}
	raw, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]any{"name": "f"}})
	var v any
	_ = json.Unmarshal(raw, &v)
	if got := effectiveToolChoice(v, 1); got != "f" {
		t.Fatalf("forced choice: %q", got)
	}
}

func TestFormatForBrowserQuotesHistory(t *testing.T) {
	m := message{Role: "assistant", ToolCalls: []toolCall{{ID: "x", Function: callFunction{Name: "f", Arguments: "{}"}}}}
	got := formatForBrowser(m)
	if !strings.Contains(got, "inert") {
		t.Fatalf("history tool calls must be labeled inert: %q", got)
	}
}

func TestModelsEndpoint(t *testing.T) {
	h := &handler{model: "gemini-web", sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	req := httptest.NewRequest("GET", "/v1/models", nil)
	rec := httptest.NewRecorder()
	h.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
}

func TestHealthz(t *testing.T) {
	h := &handler{model: "gemini-web", sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	req := httptest.NewRequest("GET", "/healthz", nil)
	rec := httptest.NewRecorder()
	h.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || strings.TrimSpace(rec.Body.String()) != "ok" {
		t.Fatalf("healthz: %d %q", rec.Code, rec.Body.String())
	}
}

func TestEmptyMessagesRejected(t *testing.T) {
	h := &handler{model: "gemini-web", sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gemini-web","messages":[]}`))
	rec := httptest.NewRecorder()
	h.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

// TestPendingTurnNeverCallsBrowser pins the per-session guard: a pending
// record with no recoverable binding at all (no pending prompt) is refused
// fast, with no browser read, no submission and no state change. The fake's
// recovery readers are nil, so any read attempt would panic here.
func TestPendingTurnNeverCallsBrowser(t *testing.T) {
	fake := &fakeRecoveryBackend{}
	h := &handler{model: "gemini-web", client: fake, sessions: map[string]*session{"s": {Pending: true}}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "hello"}}})
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("pending must fail closed: %d %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "pending uncertain turn; inspect owned chat") {
		t.Fatalf("refusal must name the state: %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "POST /v1/sessions/abandon") {
		t.Fatalf("refusal must name the operator path for an unresolvable record: %s", w.Body.String())
	}
	if fake.asks != 0 {
		t.Fatalf("a pending session must never submit: asks=%d", fake.asks)
	}
	if s := h.sessions["s"]; !s.Pending || s.PendingPrompt != "" || len(s.Messages) != 0 {
		t.Fatalf("refusal must leave the session untouched: %+v", s)
	}
}

// TestPendingSessionDoesNotBrickOthers is the anti-brick proof: an uncertain
// identity refuses only itself while a different identity completes normally.
func TestPendingSessionDoesNotBrickOthers(t *testing.T) {
	fake := &fakeRecoveryBackend{askAnswer: "other session fine"}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000, gate: make(chan struct{}, 1),
		statePath: filepath.Join(t.TempDir(), "openai-sessions.json"),
		sessions:  map[string]*session{"stuck": {Pending: true}}}
	h.gate <- struct{}{}
	w := postJSON(t, h, "live", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "hello"}}})
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "other session fine") {
		t.Fatalf("a pending session must not brick another identity: %d %s", w.Code, w.Body.String())
	}
	if fake.asks != 1 {
		t.Fatalf("the other identity must submit exactly once: asks=%d", fake.asks)
	}
	if s := h.sessions["stuck"]; !s.Pending || s.PendingPrompt != "" || len(s.Messages) != 0 {
		t.Fatalf("the pending identity must be untouched by another turn: %+v", s)
	}
}

// TestAbandonClearsOnlyNamedSessions pins the operator path: an explicit,
// named, gated clear that drops only the pending binding — never a wildcard,
// never automatic, and byte-identical for every session it did not name.
func TestAbandonClearsOnlyNamedSessions(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "openai-sessions.json")
	h := &handler{model: "gemini-web", statePath: state, maxChars: 200000, gate: make(chan struct{}, 1), sessions: map[string]*session{}}
	h.gate <- struct{}{}
	h.sessions["stuck"] = &session{
		Pending: true, PendingPrompt: "legacy prompt", URL: "https://gemini.google.com/app/abc",
		PendingMessages: []message{{Role: "user", Content: "legacy"}}, PendingTools: weatherTools(),
		PendingChoice: "auto", PendingReqHash: "pending-hash",
		Messages: []message{{Role: "user", Content: "kept"}}, LastAnswer: "kept answer",
		LastReqHash: "keep-hash", LastRespID: "keep-id", LastToolIDs: []string{"call_keep"},
	}
	h.sessions["other"] = &session{
		Pending: true, PendingPrompt: "other prompt", URL: "https://gemini.google.com/app/xyz",
		PendingMessages: []message{{Role: "user", Content: "other"}},
		Messages:        []message{{Role: "user", Content: "other kept"}},
	}
	h.saveLocked()
	rawOther := func() string {
		t.Helper()
		b, err := os.ReadFile(state)
		if err != nil {
			t.Fatal(err)
		}
		var file map[string]json.RawMessage
		if err := json.Unmarshal(b, &file); err != nil {
			t.Fatal(err)
		}
		return string(file["other"])
	}
	before := rawOther()

	post := func(body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		h.routes().ServeHTTP(w, httptest.NewRequest("POST", "/v1/sessions/abandon", strings.NewReader(body)))
		return w
	}

	// Refusals: nothing named, a wildcard, a mixed list, and a non-POST verb.
	for _, body := range []string{
		`{"sessions":[],"reason":"x"}`,
		`{"sessions":["*"]}`,
		`{"sessions":["all"]}`,
		`{"sessions":["stuck",""]}`,
		`{"sessions":["st*ck"]}`,
	} {
		if w := post(body); w.Code != http.StatusBadRequest {
			t.Fatalf("body %s must be refused whole: %d %s", body, w.Code, w.Body.String())
		}
	}
	wg := httptest.NewRecorder()
	h.routes().ServeHTTP(wg, httptest.NewRequest("GET", "/v1/sessions/abandon", nil))
	if wg.Code != http.StatusMethodNotAllowed {
		t.Fatalf("abandon must be POST only: %d", wg.Code)
	}
	if s := h.sessions["stuck"]; !s.Pending || s.PendingPrompt != "legacy prompt" {
		t.Fatalf("a refused request must clear nothing: %+v", s)
	}

	// A busy gate refuses too: the operator path never races a browser turn.
	busy := &handler{model: "gemini-web", gate: make(chan struct{}, 1), sessions: map[string]*session{"stuck": {Pending: true}}}
	wb := httptest.NewRecorder()
	busy.routes().ServeHTTP(wb, httptest.NewRequest("POST", "/v1/sessions/abandon", strings.NewReader(`{"sessions":["stuck"]}`)))
	if wb.Code != http.StatusServiceUnavailable || !strings.Contains(wb.Body.String(), "browser turn busy") {
		t.Fatalf("busy gate must refuse without clearing: %d %s", wb.Code, wb.Body.String())
	}
	if !busy.sessions["stuck"].Pending {
		t.Fatal("a busy refusal must clear nothing")
	}

	// The explicit clear: one named pending session plus one unknown id.
	w := post(`{"sessions":["stuck","missing"],"reason":"legacy record no code path can resolve"}`)
	if w.Code != http.StatusOK {
		t.Fatalf("abandon: %d %s", w.Code, w.Body.String())
	}
	var out struct {
		Requested int `json:"requested"`
		Abandoned int `json:"abandoned"`
		Results   []struct {
			Session string `json:"session"`
			Outcome string `json:"outcome"`
		} `json:"results"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Requested != 2 || out.Abandoned != 1 || len(out.Results) != 2 ||
		out.Results[0].Session != "stuck" || out.Results[0].Outcome != "abandoned" ||
		out.Results[1].Session != "missing" || out.Results[1].Outcome != "unknown" {
		t.Fatalf("every named session needs its own outcome: %s", w.Body.String())
	}

	s := h.sessions["stuck"]
	if s.Pending || s.PendingPrompt != "" || len(s.PendingMessages) != 0 || len(s.PendingTools) != 0 || s.PendingChoice != nil || s.PendingReqHash != "" {
		t.Fatalf("abandon must clear the whole pending binding: %+v", s)
	}
	if s.URL != "https://gemini.google.com/app/abc" || len(s.Messages) != 1 || s.Messages[0].Content != "kept" ||
		s.LastAnswer != "kept answer" || s.LastReqHash != "keep-hash" || s.LastRespID != "keep-id" || len(s.LastToolIDs) != 1 {
		t.Fatalf("abandon must retain transcript, URL, cache identity, result and tools: %+v", s)
	}
	if o := h.sessions["other"]; !o.Pending || o.PendingPrompt != "other prompt" {
		t.Fatalf("only named sessions may change: %+v", o)
	}
	if after := rawOther(); after != before {
		t.Fatalf("unnamed sessions must stay byte-identical:\nbefore %s\nafter  %s", before, after)
	}

	// Durable: a restart loads the cleared binding and the untouched neighbour.
	restarted := &handler{statePath: state, sessions: map[string]*session{}}
	if err := restarted.load(); err != nil {
		t.Fatalf("reload: %v", err)
	}
	if rs := restarted.sessions["stuck"]; rs == nil || rs.Pending || rs.PendingPrompt != "" {
		t.Fatalf("abandon must be durable: %+v", rs)
	}
	if ro := restarted.sessions["other"]; ro == nil || !ro.Pending {
		t.Fatalf("unnamed pending sessions must stay pending on disk: %+v", ro)
	}

	// The abandoned identity serves requests again (nothing was cleared but
	// the pending binding, so its transcript is still usable).
	fake := &fakeRecoveryBackend{askAnswer: "resumed"}
	h.client = fake
	w2 := postJSON(t, h, "stuck", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "continued"}}})
	if w2.Code != http.StatusOK || fake.asks != 1 {
		t.Fatalf("an abandoned session must serve again: %d asks=%d %s", w2.Code, fake.asks, w2.Body.String())
	}
}

func TestExactCacheNeverCallsBrowser(t *testing.T) {
	req := chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "hello"}}}
	h := &handler{model: "gemini-web", sessions: map[string]*session{"s": {LastReqHash: reqHash(&req), LastRespID: "cached", LastAnswer: "hello", LastCreated: 1}}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	raw, _ := json.Marshal(req)
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(raw)))
	r.Header.Set("X-Session-Id", "s")
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"id":"cached"`) {
		t.Fatalf("cache failed: %d %s", w.Code, w.Body.String())
	}
}

func TestPromptHashUsesAdoptionNormalization(t *testing.T) {
	if got := promptHash("hello\r\nworld"); got != gemini.PromptHash("hello\nworld") {
		t.Fatal("CRLF must hash like LF: the sent-log and adoption use one normalization")
	}
	if got := promptHash("  hello  "); got != promptHash("hello") {
		t.Fatal("surrounding whitespace must be insignificant")
	}
	if promptHash("hello") == promptHash("world") {
		t.Fatal("distinct prompts must hash distinctly")
	}
}

func TestSentDiagnosisLines(t *testing.T) {
	h := &handler{sessions: map[string]*session{}}
	// Unknown: pending with no recorded submission.
	s := &session{Pending: true, PendingPrompt: "do work"}
	if got := h.sentDiagnosis("s", s); !strings.HasPrefix(got, "sent-log:") || !strings.Contains(got, "no submission recorded") {
		t.Fatalf("unknown: %q", got)
	}
	// Consistent: last submit equals the pending prompt.
	h.appendSent(sentEntry{Event: "submitted", Session: "s", URL: "https://gemini.google.com/app/abc", PromptHash: promptHash("do work"), Bytes: 7, Chunk: 1, Chunks: 1})
	s.URL = "https://gemini.google.com/app/abc"
	if got := h.sentDiagnosis("s", s); !strings.Contains(got, "consistent") || !strings.Contains(got, "==") {
		t.Fatalf("consistent: %q", got)
	}
	// MISMATCH: pending overwritten by a retry the thread never saw.
	s.PendingPrompt = "newer retry"
	if got := h.sentDiagnosis("s", s); !strings.Contains(got, "MISMATCH") || !strings.Contains(got, "pending-overwritten-by-retry") {
		t.Fatalf("mismatch: %q", got)
	}
	// Clean: no pending prompt, last submit reported.
	clean := &session{}
	if got := h.sentDiagnosis("s", clean); !strings.Contains(got, "session clean") || !strings.Contains(got, "last submit") {
		t.Fatalf("clean: %q", got)
	}
	// Clean with nothing recorded.
	h2 := &handler{sessions: map[string]*session{}}
	if got := h2.sentDiagnosis("fresh", &session{}); !strings.Contains(got, "no submission recorded") {
		t.Fatalf("clean-empty: %q", got)
	}
}

func TestPending409CarriesDiagnosis(t *testing.T) {
	h := &handler{model: "gemini-web",
		sessions: map[string]*session{"s": {Pending: true, PendingPrompt: "newer retry", URL: "https://gemini.google.com/app/abc"}},
		gate:     make(chan struct{}, 1)}
	h.gate <- struct{}{}
	h.appendSent(sentEntry{Event: "submitted", Session: "s", URL: "https://gemini.google.com/app/abc", PromptHash: promptHash("older submission"), Bytes: 16, Chunk: 1, Chunks: 1})
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gemini-web","messages":[{"role":"user","content":"hello"}]}`))
	r.Header.Set("X-Session-Id", "s")
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r)
	if w.Code != http.StatusConflict {
		t.Fatalf("pending must fail closed: %d %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "sent-log: MISMATCH") {
		t.Fatalf("every 409 must carry the sent-log diagnosis line: %s", w.Body.String())
	}
}

func TestSentLogSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "openai-sessions.json")
	h := &handler{statePath: state, sessions: map[string]*session{}}
	h.appendSent(sentEntry{Event: "completed", Session: "s", URL: "https://gemini.google.com/app/abc", PromptHash: promptHash("do work"), Bytes: 7, Chunk: 1, Chunks: 1, AnswerHash: gemini.AnswerHash("done")})
	// A restarted adapter loads the journal but not the memory ring.
	restarted := &handler{statePath: state, sessions: map[string]*session{}}
	last := restarted.lastSentFor("s")
	if last == nil || last.Event != "completed" || last.AnswerHash != gemini.AnswerHash("done") {
		t.Fatalf("durable tail must survive restart: %+v", last)
	}
	got := restarted.sentDiagnosis("s", &session{Pending: true, PendingPrompt: "do work", URL: "https://gemini.google.com/app/abc"})
	if !strings.Contains(got, "consistent") {
		t.Fatalf("post-restart diagnosis must work from the file tail: %q", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "sent-log.jsonl")); err != nil {
		t.Fatalf("sent-log must live alongside the state file: %v", err)
	}
}

func TestOversizeTurnNeverSubmitsOrLogs(t *testing.T) {
	h := &handler{model: "gemini-web", maxChars: 10, sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	big := strings.Repeat("x", 100)
	raw, _ := json.Marshal(chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: big}}})
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(raw)))
	r.Header.Set("X-Session-Id", "s")
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize must be 413: %d %s", w.Code, w.Body.String())
	}
	// Documented chunking behavior: the submit path never splits. An
	// over-budget turn is rejected whole before any submission intent
	// exists, so no sent-log entry may record it and the session must
	// hold no uncertain pending state (a wrapper can never be cut
	// mid-turn because turns are never cut at all).
	if last := h.lastSentFor("s"); last != nil {
		t.Fatalf("rejected oversize turn must not log a submission: %+v", last)
	}
	if s := h.sessions["s"]; s != nil && (s.Pending || s.PendingPrompt != "") {
		t.Fatalf("rejected oversize turn must leave no pending state: %+v", s)
	}
}

// fakeRecoveryBackend is a read-only thread double: it answers recovery
// reads without a browser and counts Ask calls, which must stay zero —
// recovery records answers, never submits or executes.
type fakeRecoveryBackend struct {
	asks       int
	readTurn   func(url, prompt string) (gemini.TurnResult, error)
	readLatest func(url string) (string, gemini.TurnResult, error)
	// Ask double: records the submission for reseed/happy-path tests.
	askAnswer string
	askErr    error
	gotPrompt string
	gotFresh  bool
	urlNow    string
}

func (f *fakeRecoveryBackend) Ask(_ context.Context, prompt string, newSession bool, _ time.Duration) (string, error) {
	f.asks++
	f.gotPrompt = prompt
	f.gotFresh = newSession
	if f.askErr != nil {
		return "", f.askErr
	}
	return f.askAnswer, nil
}

func (f *fakeRecoveryBackend) CurrentURL(_ context.Context) string { return f.urlNow }

func (f *fakeRecoveryBackend) LoginState(_ context.Context) (*gemini.Status, error) {
	return nil, errors.New("unused in recovery tests")
}

func (f *fakeRecoveryBackend) ReadThreadTurn(_ context.Context, url, prompt string) (gemini.TurnResult, error) {
	return f.readTurn(url, prompt)
}

func (f *fakeRecoveryBackend) ReadLatestThreadTurn(_ context.Context, url string) (string, gemini.TurnResult, error) {
	return f.readLatest(url)
}

// fakeReadyBackend reports a signed-in tab so readiness paths can be exercised
// without a browser. Only the login probe differs from fakeRecoveryBackend.
type fakeReadyBackend struct{ fakeRecoveryBackend }

func (f *fakeReadyBackend) LoginState(context.Context) (*gemini.Status, error) {
	return &gemini.Status{LoggedIn: true, Composer: true}, nil
}

func structuralMismatch() *gemini.UserBubbleMismatchError {
	return &gemini.UserBubbleMismatchError{Path: "user-query", Rendered: 5000, Want: 60, DiffOff: 12, Spans: "12-4012/12-4012"}
}

func weatherTools() []toolDef {
	return []toolDef{{Type: "function", Function: toolFunction{
		Name: "get_weather", Description: "weather",
		Parameters: map[string]any{"type": "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}}},
	}}}
}

func TestSpansStructural(t *testing.T) {
	if spansStructural("") {
		t.Fatal("empty evidence must fail closed, not structural")
	}
	if spansStructural("моха") {
		t.Fatal("unparseable evidence must fail closed, not structural")
	}
	if !spansStructural("12-4012/12-4012") {
		t.Fatal("multi-KB interior divergence is structural")
	}
	if !spansStructural("0-10/0-10,500-900/500-900") {
		t.Fatal("scattered spans are structural")
	}
	if !spansStructural("4096+rest/60+rest") {
		t.Fatal("anchors that never re-converge are structural")
	}
	if spansStructural("120-148/120-150") {
		t.Fatal("one small span that re-converges is a bounded difference, not structural")
	}
}

func TestMismatchMessageContract(t *testing.T) {
	m := structuralMismatch()
	if !strings.Contains(m.Error(), "does not exactly match") {
		t.Fatalf("mismatch text must keep the deterministic() contract: %q", m.Error())
	}
}

func TestAdoptInterruptedClearsDirtyAndServesCache(t *testing.T) {
	req := chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "hello"}}}
	fake := &fakeRecoveryBackend{
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			if url != "https://gemini.google.com/app/abc" || prompt != "SUBMITTED-PROMPT" {
				return gemini.TurnResult{}, structuralMismatch()
			}
			return gemini.TurnResult{Answer: "adopted answer", URL: url}, nil
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "", gemini.TurnResult{}, errors.New("unused on the adoption path")
		},
	}
	h := &handler{model: "gemini-web", client: fake, sessions: map[string]*session{
		"s": {Pending: true, PendingPrompt: "SUBMITTED-PROMPT", URL: "https://gemini.google.com/app/abc",
			PendingMessages: req.Messages, PendingReqHash: reqHash(&req)},
	}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	if !h.recoverPending(r, "s") {
		t.Fatal("exact thread answer must adopt")
	}
	s := h.sessions["s"]
	if s.Pending || s.PendingPrompt != "" || len(s.PendingMessages) != 0 {
		t.Fatalf("adoption must clear Dirty: %+v", s)
	}
	if s.LastAnswer != "adopted answer" || s.LastReqHash != reqHash(&req) || s.LastRespID == "" {
		t.Fatalf("adoption must record the completed turn for duplicate-cache: %+v", s)
	}
	if len(s.Messages) != len(req.Messages)+1 || s.Messages[len(s.Messages)-1].Content != "adopted answer" {
		t.Fatalf("adoption must extend the transcript: %+v", s.Messages)
	}
	if fake.asks != 0 {
		t.Fatalf("recovery must never submit: asks=%d", fake.asks)
	}
	// The exact retry now hits the duplicate cache end to end.
	raw, _ := json.Marshal(req)
	r2 := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(raw)))
	r2.Header.Set("X-Session-Id", "s")
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r2)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "adopted answer") {
		t.Fatalf("post-adoption retry must serve the cache: %d %s", w.Code, w.Body.String())
	}
	if fake.asks != 0 {
		t.Fatalf("duplicate cache must not submit either: asks=%d", fake.asks)
	}
}

func TestAutoResyncTransplantsClearCutShape(t *testing.T) {
	answer := `<opencode_bridge_call>{"name":"get_weather","arguments":{"city":"Hanoi"}}</opencode_bridge_call>`
	fake := &fakeRecoveryBackend{
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			return gemini.TurnResult{}, structuralMismatch()
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "ORIGINAL SUBMISSION", gemini.TurnResult{Answer: answer, URL: url}, nil
		},
	}
	h := &handler{model: "gemini-web", client: fake, sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	pending := []message{{Role: "user", Content: "newer retry"}}
	h.sessions["s"] = &session{Pending: true, PendingPrompt: "NEWER RETRY PROMPT", URL: "https://gemini.google.com/app/abc",
		PendingMessages: pending, PendingTools: weatherTools()}
	h.appendSent(sentEntry{Event: "completed", Session: "s", URL: "https://gemini.google.com/app/abc",
		PromptHash: gemini.PromptHash("ORIGINAL SUBMISSION"), Bytes: 19, Chunk: 1, Chunks: 1})
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	if !h.recoverPending(r, "s") {
		t.Fatal("clear-cut desync shape must auto-resync")
	}
	s := h.sessions["s"]
	if s.Pending {
		t.Fatal("auto-resync must clear Dirty")
	}
	if s.LastAnswer != answer {
		t.Fatalf("transplant must record the real thread answer: %q", s.LastAnswer)
	}
	if len(s.Messages) != len(pending)+1 || len(s.Messages[len(s.Messages)-1].ToolCalls) != 1 ||
		s.Messages[len(s.Messages)-1].ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("transplanted calls must be recorded for the client: %+v", s.Messages)
	}
	if !strings.Contains(s.AutoResyncEvidence, "auto-resync") || !strings.Contains(s.AutoResyncEvidence, "pending-overwritten-by-retry") {
		t.Fatalf("evidence must explain the transplant: %q", s.AutoResyncEvidence)
	}
	last := h.lastSentFor("s")
	if last == nil || last.Event != "completed" || last.AnswerHash != gemini.AnswerHash(answer) {
		t.Fatalf("transplant must append a sent-log completion with the answer hash: %+v", last)
	}
	if fake.asks != 0 {
		t.Fatal("auto-resync records calls; it never executes or submits them")
	}
}

func TestAutoResyncRefusesWithoutLineage(t *testing.T) {
	fake := &fakeRecoveryBackend{
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			return gemini.TurnResult{}, structuralMismatch()
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "SOMEONE ELSE'S TURN", gemini.TurnResult{Answer: "real answer", URL: url}, nil
		},
	}
	h := &handler{model: "gemini-web", client: fake, adoptWait: time.Millisecond, sessions: map[string]*session{
		"s": {Pending: true, PendingPrompt: "NEWER", URL: "https://gemini.google.com/app/abc",
			PendingMessages: []message{{Role: "user", Content: "newer"}}, PendingTools: weatherTools()},
	}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	// No sent-log entry ties this thread to this session: refuse.
	if h.recoverPending(httptest.NewRequest("POST", "/v1/chat/completions", nil), "s") {
		t.Fatal("auto-resync without sent-log lineage must refuse")
	}
	s := h.sessions["s"]
	if !s.Pending || s.AutoResyncEvidence != "" {
		t.Fatalf("refusal must leave Dirty untouched and record no evidence: %+v", s)
	}
	// End to end the session still 409s with the diagnosis line.
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gemini-web","messages":[{"role":"user","content":"hi"}]}`))
	r.Header.Set("X-Session-Id", "s")
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r)
	if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "sent-log:") {
		t.Fatalf("ambiguous must keep 409 + diagnosis: %d %s", w.Code, w.Body.String())
	}
	if fake.asks != 0 {
		t.Fatal("refused recovery must not submit")
	}
}

func TestAmbiguousRecoveryKeepsDirty(t *testing.T) {
	fake := &fakeRecoveryBackend{
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			return gemini.TurnResult{}, errors.New("stored thread answer is not a finished rendering; inspect before recovery")
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "", gemini.TurnResult{}, errors.New("stored thread holds no complete assistant answer to adopt")
		},
	}
	h := &handler{model: "gemini-web", client: fake, adoptWait: time.Millisecond, sessions: map[string]*session{
		"s": {Pending: true, PendingPrompt: "P", URL: "https://gemini.google.com/app/abc",
			PendingMessages: []message{{Role: "user", Content: "p"}}},
	}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	if h.recoverPending(httptest.NewRequest("POST", "/v1/chat/completions", nil), "s") {
		t.Fatal("unfinished thread state must not recover")
	}
	if !h.sessions["s"].Pending {
		t.Fatal("ambiguous cases keep Dirty untouched")
	}
	if fake.asks != 0 {
		t.Fatal("unknown-state turns are never submitted")
	}
}

func TestRecoveryEvidenceSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "openai-sessions.json")
	answer := `<opencode_bridge_call>{"name":"get_weather","arguments":{"city":"Hanoi"}}</opencode_bridge_call>`
	fake := &fakeRecoveryBackend{
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			return gemini.TurnResult{}, structuralMismatch()
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "ORIGINAL SUBMISSION", gemini.TurnResult{Answer: answer, URL: url}, nil
		},
	}
	h := &handler{model: "gemini-web", client: fake, statePath: state, sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	pending := []message{{Role: "user", Content: "newer retry"}}
	h.sessions["s"] = &session{Pending: true, PendingPrompt: "NEWER RETRY PROMPT", URL: "https://gemini.google.com/app/abc",
		PendingMessages: pending, PendingTools: weatherTools()}
	h.appendSent(sentEntry{Event: "completed", Session: "s", URL: "https://gemini.google.com/app/abc",
		PromptHash: gemini.PromptHash("ORIGINAL SUBMISSION"), Bytes: 19, Chunk: 1, Chunks: 1})
	if !h.recoverPending(httptest.NewRequest("POST", "/v1/chat/completions", nil), "s") {
		t.Fatal("setup: auto-resync must succeed")
	}
	restarted := &handler{statePath: state, sessions: map[string]*session{}}
	if err := restarted.load(); err != nil {
		t.Fatalf("reload: %v", err)
	}
	s := restarted.sessions["s"]
	if s == nil || s.Pending || !strings.Contains(s.AutoResyncEvidence, "auto-resync") || s.LastAnswer != answer {
		t.Fatalf("journal + evidence must survive restart: %+v", s)
	}
	if got := restarted.sentDiagnosis("s", s); !strings.Contains(got, "session clean") {
		t.Fatalf("post-restart diagnosis must see the clean session: %q", got)
	}
}

func TestClassifyHistoryRewrite(t *testing.T) {
	u := func(s string) message { return message{Role: "user", Content: s} }
	a := func(s string) message { return message{Role: "assistant", Content: s} }
	// Append-only: saved is a prefix of incoming.
	if kind, _ := classifyHistoryRewrite([]message{u("1"), u("2")}, []message{u("1")}); kind != "" {
		t.Fatalf("append must not classify: %q", kind)
	}
	// Truncation: incoming shorter, prefix-equal.
	if kind, mm := classifyHistoryRewrite([]message{u("1")}, []message{u("1"), a("old")}); kind != rewriteTruncation || mm.Field != "history.truncated" {
		t.Fatalf("truncation: %q %+v", kind, mm)
	}
	// Divergence: same shape, changed content.
	if kind, _ := classifyHistoryRewrite([]message{u("1"), u("changed")}, []message{u("1"), u("2")}); kind != rewriteDivergence {
		t.Fatalf("divergence: %q", kind)
	}
	// Compaction: system summary prefix.
	incoming := []message{{Role: "system", Content: "Compacted conversation summary so far"}, u("next")}
	if kind, _ := classifyHistoryRewrite(incoming, []message{u("1"), a("old")}); kind != rewriteCompaction {
		t.Fatalf("compaction: %q", kind)
	}
	// Ambiguous: dangling tool call with no result.
	dangling := []message{u("x"), {Role: "assistant", ToolCalls: []toolCall{{ID: "c1", Type: "function", Function: callFunction{Name: "f", Arguments: "{}"}}}}}
	if kind, _ := classifyHistoryRewrite(dangling, []message{u("1")}); kind != rewriteAmbiguous {
		t.Fatalf("incoherent input must be ambiguous, never reseeded: %q", kind)
	}
}

func committedWeatherHistory() []message {
	return []message{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []toolCall{{ID: "call_1", Type: "function", Function: callFunction{Name: "get_weather", Arguments: `{"city":"Hanoi"}`}}}},
		{Role: "tool", ToolCallID: "call_1", Content: "72F"},
	}
}

func postJSON(t *testing.T, h *handler, sid string, req chatRequest) *httptest.ResponseRecorder {
	t.Helper()
	raw, _ := json.Marshal(req)
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(raw)))
	r.Header.Set("X-Session-Id", sid)
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r)
	return w
}

func TestCleanReseedCarriesExecutedTools(t *testing.T) {
	fake := &fakeRecoveryBackend{askAnswer: "fresh answer", urlNow: "https://gemini.google.com/app/NEW"}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000,
		sessions: map[string]*session{
			"s": {Messages: committedWeatherHistory(), URL: "https://gemini.google.com/app/OLD"},
		}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	// Revert: the client drops the executed exchange and asks anew.
	incoming := []message{{Role: "user", Content: "weather?"}, {Role: "user", Content: "something else"}}
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: incoming})
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "fresh answer") {
		t.Fatalf("reseed must continue as a normal submission: %d %s", w.Code, w.Body.String())
	}
	if !fake.gotFresh {
		t.Fatal("reseed must abandon the old thread for a fresh Gemini chat")
	}
	if !strings.Contains(fake.gotPrompt, "PRIOR EXECUTION RECORD") || !strings.Contains(fake.gotPrompt, "call_1") {
		t.Fatalf("seed must carry the executed exchange as inert record: %q", fake.gotPrompt)
	}
	if !strings.Contains(fake.gotPrompt, "inert") {
		t.Fatalf("carried calls must render inert, never as pending: %q", fake.gotPrompt)
	}
	s := h.sessions["s"]
	if len(s.Messages) != len(incoming) {
		t.Fatalf("journal must hold the rewritten timeline, not the carried record: %+v", s.Messages)
	}
	if s.ReseedFrom != "https://gemini.google.com/app/OLD" || s.URL != "https://gemini.google.com/app/NEW" {
		t.Fatalf("journal must link old→new thread: from=%q url=%q", s.ReseedFrom, s.URL)
	}
	if !strings.Contains(s.ReseedEvidence, "rewrite-divergence") || !strings.Contains(s.ReseedEvidence, "call_1(get_weather)") {
		t.Fatalf("evidence must carry classification + carried list: %q", s.ReseedEvidence)
	}
	var reseedNote string
	for i := len(h.sentLog) - 1; i >= 0; i-- {
		if h.sentLog[i].Event == "reseed" {
			reseedNote = h.sentLog[i].Note
		}
	}
	if !strings.Contains(reseedNote, "old=https://gemini.google.com/app/OLD") || !strings.Contains(reseedNote, "classification=rewrite-divergence") {
		t.Fatalf("sent-log must persist the linkage: %q", reseedNote)
	}
}

func TestDirtyReseedAbandonsUnknown(t *testing.T) {
	fake := &fakeRecoveryBackend{
		askAnswer: "reseed answer", urlNow: "https://gemini.google.com/app/NEW",
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			return gemini.TurnResult{}, errors.New("stored thread answer is not a finished rendering; inspect before recovery")
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "", gemini.TurnResult{}, errors.New("stored thread holds no complete assistant answer to adopt")
		},
	}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000, adoptWait: time.Millisecond,
		sessions: map[string]*session{
			"s": {Pending: true, PendingPrompt: "PENDING PROMPT", URL: "https://gemini.google.com/app/OLD",
				Messages:        []message{{Role: "user", Content: "q0"}, {Role: "assistant", Content: "a0"}},
				PendingMessages: []message{{Role: "user", Content: "q0"}, {Role: "assistant", Content: "a0"}, {Role: "user", Content: "q-pending"}}},
		}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	h.appendSent(sentEntry{Event: "submitted", Session: "s", URL: "https://gemini.google.com/app/OLD",
		PromptHash: promptHash("PENDING PROMPT"), Bytes: 14, Chunk: 1, Chunks: 1})
	// Rewrite over uncertainty: committed prefix rewritten, pending unknown.
	incoming := []message{{Role: "user", Content: "q0"}, {Role: "user", Content: "q-new"}}
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: incoming})
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "reseed answer") {
		t.Fatalf("dirty rewrite must reseed and continue: %d %s", w.Code, w.Body.String())
	}
	s := h.sessions["s"]
	if s.Pending {
		t.Fatal("reseed must clear the abandoned uncertainty")
	}
	if !strings.Contains(s.ReseedEvidence, "pending:") || !strings.Contains(s.ReseedEvidence, "never submitted as pending") {
		t.Fatalf("unknown turn must be listed as not-carried: %q", s.ReseedEvidence)
	}
	if !fake.gotFresh {
		t.Fatal("dirty reseed must start a fresh chat, never reuse the uncertain thread")
	}
}

func TestLoadBearingUnknownRefusesReseed(t *testing.T) {
	fake := &fakeRecoveryBackend{
		askAnswer: "must never submit", urlNow: "https://gemini.google.com/app/NEW",
		readTurn: func(url, prompt string) (gemini.TurnResult, error) {
			return gemini.TurnResult{}, errors.New("stored thread answer is not a finished rendering; inspect before recovery")
		},
		readLatest: func(url string) (string, gemini.TurnResult, error) {
			return "", gemini.TurnResult{}, errors.New("stored thread holds no complete assistant answer to adopt")
		},
	}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000, adoptWait: time.Millisecond,
		sessions: map[string]*session{
			"s": {Pending: true, PendingPrompt: "PENDING PROMPT", URL: "https://gemini.google.com/app/OLD",
				Messages:        []message{{Role: "user", Content: "q0"}},
				PendingMessages: []message{{Role: "user", Content: "q0"}, {Role: "user", Content: "q-pending"}}},
		}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	// The sent-log proves a COMPLETED answer for the pending prompt exists
	// on the old thread: abandoning it would lose work — fail closed.
	h.appendSent(sentEntry{Event: "completed", Session: "s", URL: "https://gemini.google.com/app/OLD",
		PromptHash: promptHash("PENDING PROMPT"), Bytes: 14, Chunk: 1, Chunks: 1, AnswerHash: "abc"})
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "q0"}, {Role: "user", Content: "q-new"}}})
	if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "sent-log:") {
		t.Fatalf("load-bearing unknown must fail closed to 409 + diagnosis: %d %s", w.Code, w.Body.String())
	}
	if !h.sessions["s"].Pending || h.sessions["s"].ReseedEvidence != "" {
		t.Fatal("refusal must leave Dirty untouched and record nothing")
	}
	if fake.asks != 0 {
		t.Fatal("unknown-state turns are never submitted")
	}
}

func TestHappyPathAppendUnchanged(t *testing.T) {
	fake := &fakeRecoveryBackend{askAnswer: "delta answer"}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000,
		sessions: map[string]*session{
			"s": {Messages: []message{{Role: "user", Content: "q0"}}, URL: "https://gemini.google.com/app/OLD"},
		}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	h.hasActive, h.activeSID = true, "s"
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "q0"}, {Role: "user", Content: "q1"}}})
	if w.Code != http.StatusOK {
		t.Fatalf("delta must succeed: %d %s", w.Code, w.Body.String())
	}
	if fake.gotFresh || strings.Contains(fake.gotPrompt, "PRIOR EXECUTION RECORD") {
		t.Fatalf("append continuation must never reseed: fresh=%v prompt=%q", fake.gotFresh, fake.gotPrompt)
	}
	if s := h.sessions["s"]; s.ReseedEvidence != "" || len(s.Messages) != 2 {
		t.Fatalf("happy path must not journal reseed state: %+v", s)
	}
}

func TestReseedOversizeNeverSplits(t *testing.T) {
	fake := &fakeRecoveryBackend{askAnswer: "must never submit"}
	h := &handler{model: "gemini-web", client: fake, maxChars: 50,
		sessions: map[string]*session{
			"s": {Messages: committedWeatherHistory(), URL: "https://gemini.google.com/app/OLD"},
		}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	incoming := []message{{Role: "user", Content: "weather?"}, {Role: "user", Content: "something else"}}
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: incoming})
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize reseed seed must be 413: %d %s", w.Code, w.Body.String())
	}
	if fake.asks != 0 {
		t.Fatal("rejected seeds are never submitted, whole or split")
	}
	if last := h.lastSentFor("s"); last != nil {
		t.Fatalf("rejected reseed must log no submission: %+v", last)
	}
	if len(h.sessions["s"].Messages) != len(committedWeatherHistory()) {
		t.Fatal("rejected reseed must leave the journal untouched")
	}
}

func TestCarrySkipsIncompleteCommitted(t *testing.T) {
	s := &session{Messages: []message{
		{Role: "user", Content: "q"},
		{Role: "assistant", ToolCalls: []toolCall{{ID: "call_9", Type: "function", Function: callFunction{Name: "f", Arguments: "{}"}}}},
	}}
	carried, _, incomplete := carryExecuted(s, nil, []message{{Role: "user", Content: "reverted"}})
	if len(carried) != 0 {
		t.Fatalf("calls without recorded results must not carry: %+v", carried)
	}
	if len(incomplete) != 1 || !strings.Contains(incomplete[0], "call_9") {
		t.Fatalf("dropped unknowns must be named: %+v", incomplete)
	}
}

func TestReseedEvidenceSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "openai-sessions.json")
	fake := &fakeRecoveryBackend{askAnswer: "fresh answer", urlNow: "https://gemini.google.com/app/NEW"}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000, statePath: state,
		sessions: map[string]*session{
			"s": {Messages: committedWeatherHistory(), URL: "https://gemini.google.com/app/OLD"},
		}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	incoming := []message{{Role: "user", Content: "weather?"}, {Role: "user", Content: "something else"}}
	if w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: incoming}); w.Code != http.StatusOK {
		t.Fatalf("setup: reseed must succeed: %d %s", w.Code, w.Body.String())
	}
	restarted := &handler{statePath: state, sessions: map[string]*session{}}
	if err := restarted.load(); err != nil {
		t.Fatalf("reload: %v", err)
	}
	s := restarted.sessions["s"]
	if s == nil || s.Pending || s.ReseedFrom != "https://gemini.google.com/app/OLD" ||
		!strings.Contains(s.ReseedEvidence, "rewrite-divergence") || len(s.Messages) != len(incoming) {
		t.Fatalf("reseed linkage must survive restart: %+v", s)
	}
}

// ---------------------------------------------------------------------------
// Output images: multipart emission, rich-turn preference, recovery carriage.
// ---------------------------------------------------------------------------

func testImageParts() []gemini.ResponseImage {
	return []gemini.ResponseImage{{Alt: "sales chart", URL: "data:image/png;base64,QUJD"}}
}

func TestAssistantContentMultipart(t *testing.T) {
	// Text-only turns stay a plain string: wire-identical to before.
	if c := assistantContent("hello", nil); c != "hello" {
		t.Fatalf("text-only must stay a string: %#v", c)
	}
	parts, ok := assistantContent("Here: [Image: sales chart]", testImageParts()).([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("image turns must be multipart: %#v", assistantContent("x", testImageParts()))
	}
	text, ok := parts[0].(map[string]any)
	if !ok || text["type"] != "text" {
		t.Fatalf("first part must be text: %#v", parts[0])
	}
	img, ok := parts[1].(map[string]any)
	if !ok || img["type"] != "image_url" {
		t.Fatalf("second part must be image_url: %#v", parts[1])
	}
	if url, _ := img["image_url"].(map[string]any)["url"].(string); url != "data:image/png;base64,QUJD" {
		t.Fatalf("image URL must pass through: %#v", img)
	}
	// Empty-URL slots never become parts; all-empty degrades to the string.
	if c := assistantContent("t", []gemini.ResponseImage{{URL: " "}}); c != "t" {
		t.Fatalf("empty URLs must degrade to text: %#v", c)
	}
}

func TestStreamTextOf(t *testing.T) {
	if got := streamTextOf("hello"); len(got) != 1 || got[0] != "hello" {
		t.Fatalf("string content: %#v", got)
	}
	if got := streamTextOf(""); len(got) != 0 {
		t.Fatalf("empty string: %#v", got)
	}
	got := streamTextOf(assistantContent("Here", testImageParts()))
	if len(got) != 1 || got[0] != "Here" {
		t.Fatalf("multipart streams text only (fixtures describe images): %#v", got)
	}
}

// fakeImageBackend offers the rich AskWithImages surface.
type fakeImageBackend struct {
	fakeRecoveryBackend
	richAnswer string
	richImages []gemini.ResponseImage
	richErr    error
	usedRich   bool
}

func (f *fakeImageBackend) AskWithImages(_ context.Context, prompt string, newSession bool, _ time.Duration) (string, []gemini.ResponseImage, error) {
	f.usedRich = true
	f.gotPrompt = prompt
	f.gotFresh = newSession
	if f.richErr != nil {
		return "", nil, f.richErr
	}
	return f.richAnswer, f.richImages, nil
}

func TestAskTurnPrefersRichSurface(t *testing.T) {
	rich := &fakeImageBackend{richAnswer: "Here: [Image: sales chart]", richImages: testImageParts()}
	h := &handler{model: "gemini-web", client: rich, sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	answer, images, err := h.askTurn(context.Background(), "draw", true)
	if err != nil || answer != "Here: [Image: sales chart]" || len(images) != 1 {
		t.Fatalf("rich surface must win: %q %+v %v", answer, images, err)
	}
	if !rich.usedRich {
		t.Fatal("AskWithImages must be preferred when offered")
	}
	// Legacy backends without the rich surface keep working (no images).
	legacy := &fakeRecoveryBackend{askAnswer: "plain"}
	h2 := &handler{model: "gemini-web", client: legacy, sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	answer, images, err = h2.askTurn(context.Background(), "hi", true)
	if err != nil || answer != "plain" || len(images) != 0 {
		t.Fatalf("legacy Ask fallback broken: %q %+v %v", answer, images, err)
	}
}

func TestCompletionCarriesImagePartsEndToEnd(t *testing.T) {
	fake := &fakeImageBackend{richAnswer: "Here: [Image: sales chart]", richImages: testImageParts()}
	h := &handler{model: "gemini-web", client: fake, maxChars: 200000, sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	w := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "draw a chart"}}})
	if w.Code != http.StatusOK {
		t.Fatalf("image turn must succeed: %d %s", w.Code, w.Body.String())
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content any `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(resp.Choices[0].Message.Content)
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil || len(parts) != 2 {
		t.Fatalf("completion must be text + image_url parts: %s", string(raw))
	}
	if parts[0].Type != "text" || !strings.Contains(parts[0].Text, "[Image: sales chart]") {
		t.Fatalf("text part must carry the fixture: %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != "data:image/png;base64,QUJD" {
		t.Fatalf("image part must carry the bytes URL: %+v", parts[1])
	}
	// Exact retry serves the duplicate cache with the same parts.
	w2 := postJSON(t, h, "s", chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "draw a chart"}}})
	if w2.Code != http.StatusOK || !strings.Contains(w2.Body.String(), "data:image/png;base64,QUJD") {
		t.Fatalf("duplicate cache must replay image parts: %d %s", w2.Code, w2.Body.String())
	}
	if fake.asks != 0 {
		t.Fatalf("cache must not resubmit: asks=%d", fake.asks)
	}
}

func TestRecordCompletedTurnCarriesImages(t *testing.T) {
	req := chatRequest{Model: "gemini-web", Messages: []message{{Role: "user", Content: "draw"}}}
	h := &handler{model: "gemini-web", sessions: map[string]*session{}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	s := &session{Pending: true, PendingPrompt: "DRAW", URL: "https://gemini.google.com/app/abc",
		PendingMessages: req.Messages, PendingReqHash: reqHash(&req)}
	h.sessions["s"] = s
	if !h.recordCompletedTurn("s", s, "Here: [Image: sales chart]", testImageParts(), "", "test") {
		t.Fatal("recovery must record the image turn")
	}
	if len(s.LastImages) != 1 {
		t.Fatalf("journal must keep the parts for duplicate-cache: %+v", s)
	}
	last := s.Messages[len(s.Messages)-1]
	raw, _ := json.Marshal(last.Content)
	if !strings.Contains(string(raw), "image_url") {
		t.Fatalf("transcript must hold the multipart message: %s", string(raw))
	}
	// Unretrievable images (no parts) still record text + note, never fail.
	s2 := &session{Pending: true, PendingPrompt: "DRAW2", URL: "https://gemini.google.com/app/abc",
		PendingMessages: req.Messages, PendingReqHash: reqHash(&req)}
	h.sessions["s2"] = s2
	if !h.recordCompletedTurn("s2", s2, "[Image 1: visual content not retrievable from web UI]", nil, "", "test") {
		t.Fatal("fallback text must record without parts")
	}
	if c, ok := s2.Messages[len(s2.Messages)-1].Content.(string); !ok || !strings.Contains(c, "not retrievable") {
		t.Fatalf("fallback stays a plain string: %#v", s2.Messages[len(s2.Messages)-1].Content)
	}
}
