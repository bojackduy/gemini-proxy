package gemini

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestPendingDraftCannotCrossAdapterRequests(t *testing.T) {
	c := &Client{draftPrompt: "same text", draftIdentity: "session-A/request-1"}
	for _, identity := range []string{"session-B/request-1", "session-A/request-2"} {
		_, err := c.Ask(WithTurnIdentity(context.Background(), identity), "same text", true, time.Second)
		if !errors.Is(err, ErrNotSubmitted) {
			t.Fatalf("cross-request draft accepted: %v", err)
		}
		if c.draftPrompt != "same text" {
			t.Fatal("pending draft discarded")
		}
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
	script := fillComposerJS(`div.ql-editor`, prompt)
	if !strings.Contains(script, "var text="+jsQuote(prompt)) {
		t.Fatal("prompt was truncated or improperly quoted")
	}
	for _, required := range []string{"q.setText(expected,'user')", "expected=text+'\\n'", "q.getText()===expected", "q.__geminiProxyDraft===text", "current!=='\\n'"} {
		if !strings.Contains(script, required) {
			t.Fatalf("missing guard %q", required)
		}
	}
	for _, forbidden := range []string{".click(", "textContent=", "innerHTML=", "setTimeout", "clipboard"} {
		if strings.Contains(script, forbidden) {
			t.Fatalf("unsafe input side effect %q", forbidden)
		}
	}
}
