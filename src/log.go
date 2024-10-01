package src

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

func setupLogging(cfg *MindletConfig) (string, error) {
	logDir := filepath.Join(cfg.ProjectRoot, cfg.OutputDir, cfg.LogDir)

	timestamp := time.Now().Format("2006-01-02-15-04-05")
	runLogDir := filepath.Join(logDir, fmt.Sprintf("run-%s", timestamp))

	if err := os.MkdirAll(runLogDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create log directory: %v", err)
	}

	return runLogDir, nil
}

type MultiWriter struct {
	writers []io.Writer
}

func (mw *MultiWriter) Write(p []byte) (n int, err error) {
	for _, w := range mw.writers {
		n, err = w.Write(p)
		if err != nil {
			return
		}
	}
	return len(p), nil
}

type PrefixedWriter struct {
	prefix string
	writer io.Writer
	buffer []byte
}

func NewPrefixedWriter(prefix string, writer io.Writer) *PrefixedWriter {
	return &PrefixedWriter{
		prefix: prefix,
		writer: writer,
	}
}

func (pw *PrefixedWriter) Write(p []byte) (n int, err error) {
	n = len(p)
	lines := bytes.Split(append(pw.buffer, p...), []byte("\n"))
	pw.buffer = []byte{}

	for i, line := range lines {
		if i == len(lines)-1 && len(line) > 0 {
			// This is an incomplete line, store it in the buffer
			pw.buffer = line
			continue
		}
		if len(line) > 0 || i < len(lines)-1 { // Write empty lines except for the last one
			_, err = pw.writer.Write([]byte(pw.prefix + " "))
			if err != nil {
				return n, err
			}
			_, err = pw.writer.Write(line)
			if err != nil {
				return n, err
			}
			if i < len(lines)-1 {
				_, err = pw.writer.Write([]byte("\n"))
				if err != nil {
					return n, err
				}
			}
		}
	}
	return n, nil
}
