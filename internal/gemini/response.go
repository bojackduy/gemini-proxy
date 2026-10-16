// HTML-to-Markdown extraction for Gemini Web model answers.
//
// Gemini renders assistant answers as HTML inside <model-response> elements.
// This file converts that rendered DOM to GitHub-flavored Markdown, mirroring
// the contract of chatgpt-proxy's response extractor (same supported
// constructs, same wire-protocol preservation) while matching Gemini's DOM:
//
//   - primary scope: the newest <model-response> outerHTML (passed in by the
//     CDP client, which already scoped it past the last user message)
//   - rendered text: div.model-response-text / div.markdown-container
//   - code: <pre><code class="language-x"> plus Gemini's copy-code buttons
//     (removed before serialization, like ChatGPT's copy controls)
//
// Wire protocol preservation: if the extracted answer consists ONLY of one or
// more <opencode_bridge_call>{...}</opencode_bridge_call> wrappers, the raw
// text is returned unchanged — no Markdown processing, JSON whitespace and
// backslashes preserved byte-for-byte. The historical ACK acknowledgement is
// also returned verbatim.
package gemini

import (
	"html"
	"regexp"
	"strings"
)

// bridgeCallRe matches <opencode_bridge_call>...</opencode_bridge_call>
// wrappers (non-greedy, dot-all emulated with [\s\S]).
var bridgeCallRe = regexp.MustCompile(`(?s)<opencode_bridge_call>[\s\S]*?</opencode_bridge_call>`)

// tagRe strips any HTML tag.
var tagRe = regexp.MustCompile(`<[^>]*>`)

// wsRe collapses 3+ newlines to two.
var wsRe = regexp.MustCompile(`\n{3,}`)

// stripTags removes all HTML tags and unescapes entities. Used for
// emptiness checks and for plain-text fallbacks.
func stripTags(s string) string {
	s = tagRe.ReplaceAllString(s, "")
	return strings.TrimSpace(html.UnescapeString(s))
}

// ConvertResponseHTML converts one model-response outerHTML fragment to
// Markdown. Pure function — unit-tested with fixture HTML, no browser needed.
func ConvertResponseHTML(fragment string) string {
	text := strings.TrimSpace(stripTags(fragment))
	if text == "" {
		return ""
	}
	// Wire-protocol preservation: only bridge wrappers (and whitespace)
	// means the model's raw text is the protocol — return it untouched.
	without := bridgeCallRe.ReplaceAllString(text, "")
	if strings.TrimSpace(without) == "" {
		return text
	}
	// Historical ACK acknowledgement passes through verbatim.
	if text == "ACK" {
		return text
	}
	return htmlToMarkdown(fragment)
}

// htmlToMarkdown converts rendered answer HTML to GitHub-flavored Markdown.
// It handles the constructs Gemini's renderer emits; anything unrecognized
// degrades to its text content (never dropped silently, never raw HTML kept
// except inside fenced code spans which are text by definition).
func htmlToMarkdown(fragment string) string {
	s := fragment

	// Remove non-content first: scripts, styles, buttons, toolbars, hidden
	// nodes, copy controls, thought blocks.
	s = regexp.MustCompile(`(?s)<(script|style)[^>]*>.*?</(script|style)>`).ReplaceAllString(s, "")
	s = regexp.MustCompile(`(?s)<button[^>]*>.*?</button>`).ReplaceAllString(s, "")
	s = regexp.MustCompile(`(?s)<[^>]*role="toolbar"[^>]*>.*?</[^>]+>`).ReplaceAllString(s, "")
	s = regexp.MustCompile(`(?s)<[^>]*(aria-hidden="true"|data-thought|data-thinking)[^>]*>.*?</[^>]+>`).ReplaceAllString(s, "")

	// Code blocks first (whitespace-sensitive): <pre><code class="language-x">
	// or Gemini's copy container. Language from class="language-x".
	s = regexp.MustCompile(`(?s)<pre[^>]*>\s*<code[^>]*class="[^"]*language-([a-zA-Z0-9+#-]+)[^"]*"[^>]*>(.*?)</code>\s*</pre>`).
		ReplaceAllStringFunc(s, func(m string) string {
			parts := regexp.MustCompile(`(?s)<pre[^>]*>\s*<code[^>]*class="[^"]*language-([a-zA-Z0-9+#-]+)[^"]*"[^>]*>(.*?)</code>\s*</pre>`).FindStringSubmatch(m)
			// Strip syntax-highlight spans BEFORE decoding source entities, then
			// keep source escaped until the final prose-tag cleanup has run.
			lang, code := parts[1], escapeCodeSource(parts[2])
			return "\n```" + lang + "\n" + strings.Trim(code, "\n") + "\n```\n"
		})
	s = regexp.MustCompile(`(?s)<pre[^>]*>(.*?)</pre>`).ReplaceAllStringFunc(s, func(m string) string {
		code := escapeCodeSource(regexp.MustCompile(`(?s)<pre[^>]*>(.*?)</pre>`).FindStringSubmatch(m)[1])
		return "\n```\n" + strings.Trim(code, "\n") + "\n```\n"
	})

	// Headings h1..h6.
	for i := 6; i >= 1; i-- {
		tag := string(rune('0' + i))
		re := regexp.MustCompile(`(?s)<h` + tag + `[^>]*>(.*?)</h` + tag + `>`)
		prefix := strings.Repeat("#", i) + " "
		s = re.ReplaceAllStringFunc(s, func(m string) string {
			inner := stripTags(re.FindStringSubmatch(m)[1])
			return "\n" + prefix + inner + "\n"
		})
	}

	// Bold / italic / strikethrough / inline code.
	s = regexp.MustCompile(`(?s)<(strong|b)[^>]*>(.*?)</(strong|b)>`).ReplaceAllString(s, "**$2**")
	s = regexp.MustCompile(`(?s)<(em|i)[^>]*>(.*?)</(em|i)>`).ReplaceAllString(s, "*$2*")
	s = regexp.MustCompile(`(?s)<(del|s|strike)[^>]*>(.*?)</(del|s|strike)>`).ReplaceAllString(s, "~~$2~~")
	s = regexp.MustCompile(`(?s)<code[^>]*>(.*?)</code>`).ReplaceAllStringFunc(s, func(m string) string {
		inner := html.EscapeString(stripTags(regexp.MustCompile(`(?s)<code[^>]*>(.*?)</code>`).FindStringSubmatch(m)[1]))
		if strings.Contains(inner, "`") {
			return "`` " + inner + " ``"
		}
		return "`" + inner + "`"
	})

	// Links: [text](url "title").
	s = regexp.MustCompile(`<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>`).ReplaceAllStringFunc(s, func(m string) string {
		parts := regexp.MustCompile(`<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>`).FindStringSubmatch(m)
		text := strings.TrimSpace(stripTags(parts[2]))
		url := strings.TrimSpace(parts[1])
		if text == "" {
			return url
		}
		return "[" + text + "](" + url + ")"
	})

	// Lists: li items get "- " (ul) or "1. " (ol); nested depth by ancestor
	// count is approximated — Gemini flattens to one level in practice.
	s = regexp.MustCompile(`(?s)<li[^>]*>(.*?)</li>`).ReplaceAllStringFunc(s, func(m string) string {
		inner := strings.TrimSpace(stripTags(regexp.MustCompile(`(?s)<li[^>]*>(.*?)</li>`).FindStringSubmatch(m)[1]))
		return "\n- " + inner
	})
	s = regexp.MustCompile(`(?s)<(ul|ol)[^>]*>(.*?)</(ul|ol)>`).ReplaceAllString(s, "\n$2\n")

	// Blockquotes.
	s = regexp.MustCompile(`(?s)<blockquote[^>]*>(.*?)</blockquote>`).ReplaceAllStringFunc(s, func(m string) string {
		inner := stripTags(regexp.MustCompile(`(?s)<blockquote[^>]*>(.*?)</blockquote>`).FindStringSubmatch(m)[1])
		lines := strings.Split(strings.TrimSpace(html.UnescapeString(inner)), "\n")
		for i, l := range lines {
			lines[i] = "> " + strings.TrimSpace(l)
		}
		return "\n" + strings.Join(lines, "\n") + "\n"
	})

	// Tables: header row + delimiter + body rows.
	s = regexp.MustCompile(`(?s)<table[^>]*>(.*?)</table>`).ReplaceAllStringFunc(s, func(m string) string {
		body := regexp.MustCompile(`(?s)<table[^>]*>(.*?)</table>`).FindStringSubmatch(m)[1]
		rowRe := regexp.MustCompile(`(?s)<tr[^>]*>(.*?)</tr>`)
		cellRe := regexp.MustCompile(`(?s)<t[hd][^>]*>(.*?)</t[hd]>`)
		var rows [][]string
		for _, row := range rowRe.FindAllStringSubmatch(body, -1) {
			var cells []string
			for _, cell := range cellRe.FindAllStringSubmatch(row[1], -1) {
				c := strings.ReplaceAll(stripTags(cell[1]), "|", `\|`)
				c = strings.ReplaceAll(strings.TrimSpace(html.UnescapeString(c)), "\n", "<br>")
				cells = append(cells, c)
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
		}
		if len(rows) == 0 {
			return ""
		}
		var b strings.Builder
		b.WriteString("\n| " + strings.Join(rows[0], " | ") + " |\n|")
		for range rows[0] {
			b.WriteString(" --- |")
		}
		b.WriteString("\n")
		for _, r := range rows[1:] {
			b.WriteString("| " + strings.Join(r, " | ") + " |\n")
		}
		return b.String()
	})

	// Horizontal rules, line breaks, paragraphs/divs.
	s = regexp.MustCompile(`<hr[^>]*/?>`).ReplaceAllString(s, "\n---\n")
	s = regexp.MustCompile(`(?i)<br[^>]*/?>`).ReplaceAllString(s, "  \n")
	s = regexp.MustCompile(`(?s)<(p|div|section|article)[^>]*>(.*?)</(p|div|section|article)>`).ReplaceAllString(s, "\n$2\n")

	// Final: strip leftovers, unescape, normalize blank lines.
	s = tagRe.ReplaceAllString(s, "")
	s = html.UnescapeString(s)
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimRight(l, " \t")
	}
	s = strings.Join(lines, "\n")
	s = wsRe.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

func escapeCodeSource(fragment string) string {
	return html.EscapeString(html.UnescapeString(tagRe.ReplaceAllString(fragment, "")))
}
