package src

import (
	"os"
	"path/filepath"
	"time"
)

// ModelType enum for model types
type ModelType string

const (
	Model8B   ModelType = "8b"
	Model70B  ModelType = "70b"
	Model405B ModelType = "405b"
)

// MindletConfig contains application configuration
type MindletConfig struct {
	// Paths
	ProjectRoot string // Base directory of the project
	PythonPath  string // Path to Python executable
	LogDir      string // Directory for logs
	RamFsRoot   string // Root directory for RAM filesystem

	// Server settings
	Port int // Server port

	// Model settings
	ModelType           ModelType     // Type of model
	BatchSize           int           // Batch size for inference
	NumCheckpointsAhead int           // Number of checkpoints to preload
	ModelCacheTick      time.Duration // How often to check for model updates
	ModelCleanupTick    time.Duration // How often to clean up old models
	ModelTTL            time.Duration // Time-to-live for unused models
}

func DefaultMindletConfig() *MindletConfig {
	// Try to determine where we are running
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}

	// Use current working directory as project root by default
	workDir, err := os.Getwd()
	if err != nil {
		workDir = "."
	}

	// Look for Python executable
	pythonPath := "python" // Default to system Python

	// Check common Python paths
	possiblePaths := []string{
		filepath.Join(homeDir, "anaconda3", "bin", "python"),
		filepath.Join(homeDir, "miniconda3", "bin", "python"),
		"/usr/bin/python3",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			pythonPath = path
			break
		}
	}

	return &MindletConfig{
		ProjectRoot:         workDir,
		PythonPath:          pythonPath,
		LogDir:              filepath.Join(workDir, "logs"),
		RamFsRoot:           filepath.Join(workDir, "ram"),
		Port:                8001,
		ModelType:           Model8B,
		BatchSize:           8,
		NumCheckpointsAhead: 4,
		ModelCacheTick:      50 * time.Millisecond,
		ModelCleanupTick:    time.Minute,
		ModelTTL:            30 * time.Minute,
	}
}
