package gemini

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestAttemptedDraftCannotCrossAdapterRequests(t *testing.T) {
	c := &Client{draftPrompt: "same text", draftIdentity: "session-A/request-1", draftHash: draftHash("same text"), draftSubmitted: true}
	for _, identity := range []string{"session-B/request-1", "session-A/request-2"} {
		_, err := c.Ask(WithTurnIdentity(context.Background(), identity), "same text", true, time.Second)
		if !errors.Is(err, ErrDraftOwned) {
			t.Fatalf("attempted draft crossed adapter requests: %v", err)
		}
		if c.draftPrompt != "same text" || !c.draftSubmitted {
			t.Fatal("attempted draft record discarded")
		}
	}
}

// TestDecideGeminiDraftOwnership pins the Quill ownership table without a
// browser: every branch that can erase or preserve text is decided here.
func TestDecideGeminiDraftOwnership(t *testing.T) {
	const prompt = "the prompt we typed"
	const target = "T1"
	const thread = "https://gemini.google.com/app/abc"
	rec := geminiDraftRecord{Prompt: prompt, Doc: prompt, Hash: draftHash(prompt), Identity: "id1", Target: target, Thread: thread}
	own := geminiDraftState{OK: true, Text: prompt, Marked: true}
	cases := []struct {
		name       string
		state      geminiDraftState
		identity   string
		target     string
		thread     string
		prompt     string
		rec        geminiDraftRecord
		wantOwned  bool
		wantReuse  bool
		wantReason string
	}{
		{"unreadable editor refuses", geminiDraftState{}, "id1", target, thread, prompt, rec, false, false, "no-quill"},
		{"empty composer types", geminiDraftState{OK: true}, "id1", target, thread, prompt, rec, true, false, "empty"},
		{"submission attempted refuses", own, "id2", target, thread, prompt, geminiDraftRecord{Prompt: prompt, Doc: prompt, Hash: draftHash(prompt), Identity: "id1", Target: target, Thread: thread, Submitted: true}, false, false, "submission-attempted"},
		{"same turn reuses", own, "id1", target, thread, prompt, rec, true, true, "same-turn"},
		{"superseded same tab and thread replaces", own, "id2", target, thread, prompt, rec, true, false, "superseded-owned-draft"},
		// The page marker alone is NOT ownership: without a matching Go record
		// the document cannot be attributed to this bridge.
		{"marker without a record refuses", geminiDraftState{OK: true, Text: prompt, Marked: true}, "id1", target, thread, prompt, geminiDraftRecord{}, false, false, "unverified-document"},
		// A matching document on a different tab or conversation is not proof.
		{"different target refuses", own, "id2", "T2", thread, prompt, rec, false, false, "other-target"},
		{"different thread refuses", own, "id2", target, "https://gemini.google.com/app/other", prompt, rec, false, false, "other-thread"},
		{"record without a target refuses", own, "id2", target, thread, prompt, geminiDraftRecord{Prompt: prompt, Doc: prompt, Hash: draftHash(prompt), Identity: "id1"}, false, false, "unverified-document"},
		// Identity reuse: the same adapter request asking for DIFFERENT text must
		// not be treated as an exact reuse.
		{"same identity different request text is superseded", own, "id1", target, thread, "a newer request", rec, true, false, "superseded-owned-draft"},
		{"same document different identity is superseded", own, "id9", target, thread, prompt, rec, true, false, "superseded-owned-draft"},
		{"user edit refuses", geminiDraftState{OK: true, Text: prompt + " plus"}, "id2", target, thread, prompt, rec, false, false, "unverified-document"},
		{"user shortened to a prefix refuses", geminiDraftState{OK: true, Text: "the prompt"}, "id2", target, thread, prompt, rec, false, false, "unverified-document"},
		{"stale record for other text refuses", geminiDraftState{OK: true, Text: "other text"}, "id2", target, thread, prompt, geminiDraftRecord{Prompt: prompt, Doc: "other different text", Hash: draftHash("other different text"), Identity: "id1", Target: target, Thread: thread}, false, false, "unverified-document"},
		{"record whose digest disagrees with its document refuses", geminiDraftState{OK: true, Text: prompt}, "id2", target, thread, prompt, geminiDraftRecord{Prompt: prompt, Doc: prompt, Hash: draftHash("something else"), Identity: "id1", Target: target, Thread: thread}, false, false, "unverified-document"},
		// A lost CDP reply leaves no Go record. The page token is sound only
		// because it names this target, this conversation and this exact text.
		{"bound page token proves ownership after a lost reply", geminiDraftState{OK: true, Text: prompt, Marked: true, MarkedText: prompt, MarkedTarget: target, MarkedThread: thread}, "id1", target, thread, prompt, geminiDraftRecord{}, true, false, "superseded-owned-draft"},
		{"bound page token reuses the same request", geminiDraftState{OK: true, Text: prompt, Marked: true, MarkedText: prompt, MarkedTarget: target, MarkedThread: thread}, "id1", target, thread, prompt, geminiDraftRecord{Prompt: prompt, Identity: "id1"}, true, true, "same-turn"},
		{"page token on another target refuses", geminiDraftState{OK: true, Text: prompt, Marked: true, MarkedText: prompt, MarkedTarget: "T2", MarkedThread: thread}, "id1", target, thread, prompt, geminiDraftRecord{}, false, false, "other-target"},
		{"page token on another thread refuses", geminiDraftState{OK: true, Text: prompt, Marked: true, MarkedText: prompt, MarkedTarget: target, MarkedThread: "https://gemini.google.com/app/other"}, "id1", target, thread, prompt, geminiDraftRecord{}, false, false, "other-thread"},
		{"page token whose text no longer matches refuses", geminiDraftState{OK: true, Text: prompt + " plus", Marked: true, MarkedText: prompt, MarkedTarget: target, MarkedThread: thread}, "id1", target, thread, prompt, geminiDraftRecord{}, false, false, "unverified-document"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			owned, reuse, why := decideGeminiDraft(tc.state, tc.identity, tc.target, tc.thread, tc.prompt, tc.rec)
			if owned != tc.wantOwned || reuse != tc.wantReuse || why != tc.wantReason {
				t.Fatalf("decideGeminiDraft = (%v,%v,%q), want (%v,%v,%q)", owned, reuse, why, tc.wantOwned, tc.wantReuse, tc.wantReason)
			}
		})
	}
}

// A truncated digest must never authorize an overwrite.
func TestDraftDigestsAreNotTruncated(t *testing.T) {
	doc := "some document text"
	if got := draftHash(doc); len(got) != 64 {
		t.Fatalf("ownership digest is %d hex chars, want 64", len(got))
	}
	if got := draftHashShort(doc); len(got) != 8 {
		t.Fatalf("log digest is %d hex chars, want 8", len(got))
	}
	if draftHashShort(doc) == draftHash(doc) {
		t.Fatal("short and ownership digests must not be interchangeable")
	}
}

func TestOperationContext(t *testing.T) {
	for _, cause := range []string{"caller", "tab", "budget", "already-canceled"} {
		t.Run(cause, func(t *testing.T) {
			tab, closeTab := context.WithCancel(context.Background())
			defer closeTab()
			caller, closeCaller := context.WithCancel(context.Background())
			defer closeCaller()
			budget := time.Minute
			if cause == "budget" {
				budget = time.Millisecond
			}
			if cause == "already-canceled" {
				closeCaller()
			}
			ctx, cancel := operationContext(tab, caller, budget)
			defer cancel()
			if cause == "caller" {
				closeCaller()
			}
			if cause == "tab" {
				closeTab()
			}
			select {
			case <-ctx.Done():
			case <-time.After(time.Second):
				t.Fatal("operation ignored cancellation")
			}
			if cause != "tab" && tab.Err() != nil {
				t.Fatal("operation poisoned long-lived tab")
			}
		})
	}
}

func TestOperationCleanupPreservesRoots(t *testing.T) {
	tab := context.Background()
	caller := context.Background()
	ctx, cancel := operationContext(tab, caller, time.Minute)
	cancel()
	if ctx.Err() == nil || tab.Err() != nil || caller.Err() != nil {
		t.Fatal("incorrect context ownership")
	}
}

func TestFillComposerEncodesEntireLargePrompt(t *testing.T) {
	prompt := strings.Repeat("public catalog \"quoted\" 日本語\n", 5000) + "\n"
	script := fillComposerJS(`div.ql-editor`, prompt, "", true, "T1")
	if !strings.Contains(script, "var text="+jsQuote(prompt)) {
		t.Fatal("prompt was truncated or improperly quoted")
	}
	for _, required := range []string{"q.setText(expected,'user')", "expected=text+'\\n'", "q.getText()===expected", "q.__geminiProxyDraft={text:text,target:", "current!==''", "if(current!==prev)return 'changed'"} {
		if !strings.Contains(script, required) {
			t.Fatalf("missing guard %q", required)
		}
	}
	// The exact document Go decided about is part of the in-page transaction:
	// an edit between the Go read and this mutation refuses instead of losing it.
	prev := "text the user typed between the read and the fill"
	guarded := fillComposerJS(`div.ql-editor`, prompt, prev, true, "T1")
	if !strings.Contains(guarded, "var text="+jsQuote(prompt)+", prev="+jsQuote(prev)) {
		t.Fatal("previous logical document is not revalidated in the page")
	}
	// The marker is set before the early return, so a reused document is never
	// left unmarked for a later Send.
	if strings.Index(script, "q.__geminiProxyDraft={text:text,target:") > strings.Index(script, "if(raw===expected)return 'reused'") {
		t.Fatal("marker is not set before the reuse early return")
	}
	// The Go-side ownership verdict is a hard gate inside the page too: without
	// it a non-empty document is never overwritten.
	unowned := fillComposerJS(`div.ql-editor`, prompt, prev, false, "T1")
	if !strings.Contains(unowned, "return 'foreign-draft'") || strings.Contains(unowned, "current!==''&&!true") {
		t.Fatalf("unowned fill must refuse a non-empty document: %s", unowned)
	}
	for _, forbidden := range []string{".click(", "textContent=", "innerHTML=", "setTimeout", "clipboard"} {
		if strings.Contains(script, forbidden) {
			t.Fatalf("unsafe input side effect %q", forbidden)
		}
	}
}
