package src

import "time"

type ModelType string

const (
	Model8B   ModelType = "8b"
	Model70B  ModelType = "70b"
	Model405B ModelType = "405b"
)

type MindletConfig struct {
	Port                int
	MaxSeqLength        int
	StreamLogs          bool
	ModelType           ModelType
	SystemPrompt        string
	NumCheckpointsAhead int
	BatchSize           int
	ModelTTL            time.Duration
	ModelCleanupTick    time.Duration
	ModelCacheTick      time.Duration

	// Root directories
	ProjectRoot string
	RamFsRoot   string

	// ProjectRoot Subdirectories
	BootDir   string
	OutputDir string
	ModelPath string

	// ProjectRoot/BootDir Subdirectories
	TrainingDir     string
	ModelManagerDir string
	PythonPath      string

	// ProjectRoot/Output Subdirectories
	LogDir string
}

func DefaultMindletConfig() *MindletConfig {
	return &MindletConfig{
		Port:                8001,
		MaxSeqLength:        2048,
		StreamLogs:          false,
		ModelType:           Model8B,
		SystemPrompt:        "You are a helpful AI assistant",
		NumCheckpointsAhead: 4,
		BatchSize:           8,
		ModelTTL:            30 * time.Minute,
		ModelCleanupTick:    time.Minute,
		ModelCacheTick:      50 * time.Millisecond,
		ProjectRoot:         "/mnt/hot_storage",
		RamFsRoot:           "/mnt/hotter_storage",
		BootDir:             "/boot",
		OutputDir:           "/output",
		ModelPath:           "/model",
		TrainingDir:         "/training",
		ModelManagerDir:     "/model_manager",
		PythonPath:          "/inference-server-venv/bin/python",
		LogDir:              "/mindlet_logs",
	}
}
