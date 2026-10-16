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
	for _, pending := range []bool{false, true} {
		h := &handler{gate: make(chan struct{}, 1), sessions: map[string]*session{}}
		want := "browser turn busy"
		if pending {
			h.gate <- struct{}{}
			h.sessions["owned"] = &session{Pending: true}
			want = "pending uncertain turn"
		}
		w := httptest.NewRecorder()
		h.routes().ServeHTTP(w, httptest.NewRequest("GET", "/readyz", nil))
		if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), want) || strings.Contains(w.Body.String(), "not logged in") {
			t.Fatalf("pending=%v: %d %s", pending, w.Code, w.Body.String())
		}
	}
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

func TestPendingTurnNeverCallsBrowser(t *testing.T) {
	h := &handler{model: "gemini-web", sessions: map[string]*session{"s": {Pending: true}}, gate: make(chan struct{}, 1)}
	h.gate <- struct{}{}
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gemini-web","messages":[{"role":"user","content":"hello"}]}`))
	r.Header.Set("X-Session-Id", "s")
	w := httptest.NewRecorder()
	h.routes().ServeHTTP(w, r)
	if w.Code != http.StatusConflict {
		t.Fatalf("pending must fail closed: %d %s", w.Code, w.Body.String())
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
