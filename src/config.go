package src

type ModelType string

const (
	Model8B   ModelType = "8b"
	Model70B  ModelType = "70b"
	Model405B ModelType = "405b"
)

type MindletConfig struct {
	MaxSeqLength        int
	StreamLogs          bool
	ModelType           ModelType
	NumCheckpointsAhead int

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
		MaxSeqLength:        2048,
		StreamLogs:          false,
		ModelType:           Model8B,
		NumCheckpointsAhead: 4,
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
