package coding

import (
	"bytes"
	"fmt"
	"strings"
	"unicode/utf8"
)

const (
	defaultMaxLines = 2000
	defaultMaxBytes = 50 * 1024
)

func formatSize(bytes int) string {
	switch {
	case bytes < 1024:
		return fmt.Sprintf("%dB", bytes)
	case bytes < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(bytes)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(bytes)/(1024*1024))
	}
}

func splitCountLines(content string) []string {
	if content == "" {
		return nil
	}
	lines := strings.Split(content, "\n")
	if strings.HasSuffix(content, "\n") {
		lines = lines[:len(lines)-1]
	}
	return lines
}

type truncation struct {
	content     string
	truncated   bool
	totalLines  int
	outputLines int
}

type tailBuffer struct {
	data       []byte
	totalBytes int64
	totalLines int64
}

func (b *tailBuffer) Write(p []byte) (int, error) {
	b.totalBytes += int64(len(p))
	b.totalLines += int64(bytes.Count(p, []byte("\n")))
	b.data = append(b.data, p...)
	const keep = defaultMaxBytes * 2
	if len(b.data) > keep {
		start := len(b.data) - keep
		for start < len(b.data) && !utf8.RuneStart(b.data[start]) {
			start++
		}
		b.data = append(b.data[:0], b.data[start:]...)
	}
	return len(p), nil
}

func (b *tailBuffer) totalLineCount() int {
	if b.totalBytes == 0 {
		return 0
	}
	lines := b.totalLines
	if len(b.data) > 0 && b.data[len(b.data)-1] != '\n' {
		lines++
	}
	return int(lines)
}

func (b *tailBuffer) truncate() truncation {
	totalLines := b.totalLineCount()
	if b.totalBytes <= defaultMaxBytes && totalLines <= defaultMaxLines {
		content := string(b.data)
		return truncation{content: content, totalLines: totalLines, outputLines: len(splitCountLines(content))}
	}

	data := b.data
	if len(data) > defaultMaxBytes {
		start := len(data) - defaultMaxBytes
		for start < len(data) && !utf8.RuneStart(data[start]) {
			start++
		}
		data = data[start:]
	}
	text := string(data)
	lines := strings.Split(text, "\n")
	if len(lines) > defaultMaxLines {
		lines = lines[len(lines)-defaultMaxLines:]
	}
	text = strings.Join(lines, "\n")
	return truncation{
		content:     text,
		truncated:   true,
		totalLines:  totalLines,
		outputLines: len(lines),
	}
}
