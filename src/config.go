// mindlet/src/config.go
package src

import (
	"os"
	"path/filepath"
)

// ModelType enum for model types
type ModelType string

const (
	Model8B   ModelType = "8b"
	Model70B  ModelType = "70b"
	Model405B ModelType = "405b"
)

type MindletConfig struct {
	// Server configuration
	Port int

	// VLLM configuration
	UseVLLM            bool
	TensorParallelSize int
	VLLMHost           string
	VLLMPort           int     // Base port for VLLM servers (incremented per model)
	GPUMemoryUtil      float64 // GPU memory utilization (0-1)
	MaxModelLen        int     // Maximum model context length

	// Model paths
	ConvertedModelsDir string // Where to store converted safetensors models
	DCPModelsDir       string // Where to find DCP checkpoints

	// Conversion script configuration
	ConversionScriptPath string // Path to dcp_to_safetensors.py
	PythonPath           string // Path to python executable
	DefaultTokenizerPath string // Default tokenizer path
}

func DefaultMindletConfig() *MindletConfig {
	// Try to determine where we are running
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}

	// Look for Python executable
	pythonPath := "python3" // Default to system Python

	// Check common Python paths
	possiblePaths := []string{
		filepath.Join(homeDir, "anaconda3", "bin", "python"),
		filepath.Join(homeDir, "miniconda3", "bin", "python"),
		"/usr/bin/python3",
		"python3",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			pythonPath = path
			break
		}
	}

	return &MindletConfig{
		// Server configuration
		Port: 9090,

		// VLLM configuration
		UseVLLM:            true,
		TensorParallelSize: 8,
		VLLMHost:           "0.0.0.0",
		VLLMPort:           8000,
		GPUMemoryUtil:      0, // 0 means use VLLM default
		MaxModelLen:        0, // 0 means use VLLM default

		// Model paths
		ConvertedModelsDir:   "/opt/dlami/nvme/converted_models",
		DCPModelsDir:         "/mnt/cold/contents/dcp",
		ConversionScriptPath: "/home/ec2-user/workspace/torchchat/dcp_to_safetensors.py",
		PythonPath:           pythonPath,
		DefaultTokenizerPath: "/mnt/cold/contents/checkpoints/Llama3.1-8B-Instruct",
	}
}

// LocalTestConfig returns config for local testing
func LocalTestConfig() *MindletConfig {

	return &MindletConfig{
		// Server configuration
		Port: 19090,

		// VLLM configuration
		UseVLLM:            true,
		TensorParallelSize: 1,
		VLLMHost:           "0.0.0.0",
		VLLMPort:           18000,
		GPUMemoryUtil:      0.8,
		MaxModelLen:        16000,

		// Model paths
		ConvertedModelsDir:   "/home/nlpfollower/Desktop/deltamind/torchtitan/outputs/converted_models",
		DCPModelsDir:         "/home/nlpfollower/Desktop/deltamind/torchtitan/outputs",
		ConversionScriptPath: "/home/nlpfollower/Desktop/deltamind/torchchat/dcp_to_safetensors.py",
		PythonPath:           "/home/nlpfollower/anaconda3/envs/llama-3/bin/python",
		DefaultTokenizerPath: "/home/nlpfollower/Desktop/deltamind/torchtitan/models/Llama3.1-8B-Instruct",
	}
}
