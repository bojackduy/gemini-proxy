package gemini

import (
	"strings"
	"testing"
)

func TestConvertEmpty(t *testing.T) {
	if got := ConvertResponseHTML(""); got != "" {
		t.Fatalf("empty fragment: got %q", got)
	}
	if got := ConvertResponseHTML("<model-response></model-response>"); got != "" {
		t.Fatalf("empty model-response: got %q", got)
	}
}

func TestCodeSourceAngleBracketsSurviveProseCleanup(t *testing.T) {
	for _, fragment := range []string{
		`<pre><code class="language-html"><span>&lt;div&gt;</span>source&lt;/div&gt;</code></pre>`,
		`<pre><code><span>&lt;div&gt;</span>source&lt;/div&gt;</code></pre>`,
		`<p>Use <code>&lt;div&gt;source&lt;/div&gt;</code>.</p>`,
	} {
		if got := ConvertResponseHTML(fragment); !strings.Contains(got, "<div>source</div>") {
			t.Fatalf("source markup was lost: %q", got)
		}
	}
}

func TestLiveGeminiCodeWrapperAfterDOMCleanup(t *testing.T) {
	// Live XML code-block content after clone removal of decoration/toolbars.
	fragment := `<div class="markdown"><code-block><pre><code><span>&lt;<span>opencode_bridge_call</span>&gt;</span>{"name":"lookup_marker","arguments":{"marker":"TEST-ONLY"}}<span>&lt;/<span>opencode_bridge_call</span>&gt;</span>` + "\n" + `</code></pre></code-block></div>`
	want := `<opencode_bridge_call>{"name":"lookup_marker","arguments":{"marker":"TEST-ONLY"}}</opencode_bridge_call>`
	if got := ConvertResponseHTML(fragment); got != want {
		t.Fatalf("protocol changed: %q", got)
	}
}

func TestConvertParagraphsAndHeadings(t *testing.T) {
	html := `<model-response><div class="model-response-text"><h2>Title</h2><p>Hello <strong>world</strong> and <em>moon</em>.</p></div></model-response>`
	got := ConvertResponseHTML(html)
	if !strings.Contains(got, "## Title") {
		t.Fatalf("missing heading: %q", got)
	}
	if !strings.Contains(got, "Hello **world** and *moon*.") {
		t.Fatalf("missing formatted paragraph: %q", got)
	}
}

func TestConvertCodeBlockWithLanguage(t *testing.T) {
	html := `<model-response><div class="markdown-container"><pre><code class="language-go">package main
</code></pre></div></model-response>`
	got := ConvertResponseHTML(html)
	if !strings.Contains(got, "```go") || !strings.Contains(got, "package main") {
		t.Fatalf("code block not converted: %q", got)
	}
}

func TestConvertListTableQuoteLink(t *testing.T) {
	html := `<model-response><div class="markdown-container"><ul><li>one</li><li>two</li></ul>` +
		`<blockquote><p>cite</p></blockquote>` +
		`<p>see <a href="https://example.com">docs</a></p>` +
		`<table><tr><th>H1</th><th>H2</th></tr><tr><td>a|b</td><td>c</td></tr></table>` +
		`</div></model-response>`
	got := ConvertResponseHTML(html)
	for _, want := range []string{"- one", "- two", "> cite", "[docs](https://example.com)", "| H1 | H2 |", `a\|b`} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %q", want, got)
		}
	}
}

func TestConvertStripsButtons(t *testing.T) {
	html := `<model-response><div class="model-response-text"><p>answer</p><button>Copy code</button></div></model-response>`
	got := ConvertResponseHTML(html)
	if strings.Contains(got, "Copy code") {
		t.Fatalf("button text leaked: %q", got)
	}
	if !strings.Contains(got, "answer") {
		t.Fatalf("answer lost: %q", got)
	}
}

func TestWireProtocolPreserved(t *testing.T) {
	// In the real DOM the model's angle brackets arrive entity-escaped
	// inside outerHTML; stripTags must not eat them.
	raw := "&lt;opencode_bridge_call&gt;\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Hanoi\"}}\n&lt;/opencode_bridge_call&gt;"
	html := "<model-response><div class=\"model-response-text\"><p>" + raw + "</p></div></model-response>"
	got := ConvertResponseHTML(html)
	if !strings.Contains(got, "<opencode_bridge_call>") {
		t.Fatalf("wrapper not preserved: %q", got)
	}
	if strings.Contains(got, "**") || strings.Contains(got, "```") {
		t.Fatalf("wrapper was markdown-processed: %q", got)
	}
}

func TestACKPassthrough(t *testing.T) {
	html := `<model-response><div class="model-response-text"><p>ACK</p></div></model-response>`
	if got := ConvertResponseHTML(html); got != "ACK" {
		t.Fatalf("ACK not verbatim: %q", got)
	}
}

func TestStripTags(t *testing.T) {
	if got := stripTags(`<div>a &amp; b</div>`); got != "a & b" {
		t.Fatalf("stripTags: %q", got)
	}
}

func TestNormalizePromptCanonical(t *testing.T) {
	if NormalizePrompt("  hello\r\nworld\r ") != "hello\nworld" {
		t.Fatal("CRLF/CR normalize to LF and edges trim")
	}
	if NormalizePrompt("a  b") != "a  b" {
		t.Fatal("interior spacing is significant")
	}
}

func TestPromptHashLineage(t *testing.T) {
	// The sent-log and the adoption comparison share this hash: any two
	// strings equal after normalization must share lineage.
	if PromptHash("hello\r\nworld") != PromptHash("hello\nworld") {
		t.Fatal("normalization-equal prompts must share hash lineage")
	}
	if PromptHash("hello") == PromptHash("world") {
		t.Fatal("distinct prompts must hash distinctly")
	}
	if AnswerHash("done") == AnswerHash("done!") {
		t.Fatal("answer hashes must distinguish outcomes")
	}
}

func TestDiffSpansBoundedVsStructural(t *testing.T) {
	if got := DiffSpans("same text", "same text"); got != "" {
		t.Fatalf("identical inputs must have no spans: %q", got)
	}
	shared := strings.Repeat("0123456789abcdef", 4) // 64B anchor either side
	bounded := DiffSpans(shared+"SUFFIX"+shared, shared+"DIFFER"+shared)
	if bounded == "" || strings.Contains(bounded, "+rest") || strings.Contains(bounded, ",") {
		t.Fatalf("small difference must be one re-converging span: %q", bounded)
	}
	huge := DiffSpans(strings.Repeat("a", 9000), strings.Repeat("b", 9000))
	if !strings.Contains(huge, "+rest") {
		t.Fatalf("multi-KB divergence with no anchor must never re-converge: %q", huge)
	}
}
