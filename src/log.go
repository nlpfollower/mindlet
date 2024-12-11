package src

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type LogLevel string

const (
	InfoLevel  LogLevel = "INFO"
	WarnLevel  LogLevel = "WARN"
	ErrorLevel LogLevel = "ERROR"

	infoColor  = "\033[34m" // Blue
	warnColor  = "\033[33m" // Yellow
	errorColor = "\033[31m" // Red
	resetColor = "\033[0m"

	defaultLogsDir = "../logs" // Default directory for logs
)

type Logger struct {
	logFile *os.File
	mu      sync.Mutex
}

// NewLogger creates a new logger that writes to a single file in the run directory
func NewLogger() (*Logger, error) {
	return NewLoggerWithDir(defaultLogsDir)
}

// NewLoggerWithDir creates a new logger that writes to a specific directory
func NewLoggerWithDir(logDir string) (*Logger, error) {
	// Create run directory with timestamp
	timestamp := time.Now().Format("2006-01-02-15-04-05")
	runDir := filepath.Join(logDir, fmt.Sprintf("run-%s", timestamp))

	if err := os.MkdirAll(runDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %v", err)
	}

	// Single log file for all components
	logPath := filepath.Join(runDir, "mindlet.log")
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %v", err)
	}

	return &Logger{
		logFile: logFile,
	}, nil
}

func (l *Logger) log(level LogLevel, label string, format string, args ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()

	timestamp := time.Now().Format("2006-01-02 15:04:05")
	message := fmt.Sprintf(format, args...)

	var color string
	switch level {
	case InfoLevel:
		color = infoColor
	case WarnLevel:
		color = warnColor
	case ErrorLevel:
		color = errorColor
	}

	// Colored output for terminal
	coloredEntry := fmt.Sprintf("%s%s [%s] [%s]%s %s\n",
		color, timestamp, level, label, resetColor, message)
	fmt.Print(coloredEntry)

	// Plain output for file
	plainEntry := fmt.Sprintf("%s [%s] [%s] %s\n",
		timestamp, level, label, message)
	l.logFile.WriteString(plainEntry)
}

func (l *Logger) Info(label string, format string, args ...interface{}) {
	l.log(InfoLevel, label, format, args...)
}

func (l *Logger) Warn(label string, format string, args ...interface{}) {
	l.log(WarnLevel, label, format, args...)
}

func (l *Logger) Error(label string, format string, args ...interface{}) {
	l.log(ErrorLevel, label, format, args...)
}

func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.logFile != nil {
		return l.logFile.Close()
	}
	return nil
}
