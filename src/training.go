package src

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type TrainingConfig struct {
	ModelPath     string              `json:"model_path"`
	DatasetPath   string              `json:"dataset_path"`
	TokenizerPath string              `json:"tokenizer_path"`
	ModelName     string              `json:"model_name"`
	TensorPreload TensorPreloadCfg    `json:"tensor_preload"`
	TorchTitanCfg TorchTitanConfigCfg `json:"torchtitan_config"`
	NodeTopology  NodeTopologyCfg     `json:"node_topology"`
	OutputDir     string              `json:"output_dir"`
	NFSPath       string              `json:"nfs_path"`
	Rank          int                 `json:"rank"`
	WorldSize     int                 `json:"world_size"`
	NodeIP        string              `json:"node_ip"`
}

type TrainingStatus struct {
	Status    string    `json:"status"` // "running", "completed", "error"
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time,omitempty"`
	Error     string    `json:"error,omitempty"`
	RunID     string    `json:"run_id"`
	Rank      int       `json:"rank"`
}

// New struct to handle TorchTitan config
type TorchTitanConfigCfg struct {
	ConfigPath     string   `json:"config_path"`
	OverrideParams []string `json:"override_params"`
}

type TensorPreloadCfg struct {
	Enabled   bool   `json:"enabled"`
	Threads   int    `json:"threads"`
	RedisHost string `json:"redis_host"`
	RedisPort int    `json:"redis_port"`
	RunID     string `json:"run_id"`
}

type NodeTopologyCfg struct {
	Head    string   `json:"head"`
	Workers []string `json:"workers"`
}

// preloaderCmd stores the preloader process for cleanup
var preloaderCmd *exec.Cmd
var preloaderMutex sync.Mutex

// TrainingManager handles the execution of training tasks
type TrainingManager struct {
	config          *MindletConfig
	logger          *Logger
	runID           string
	scriptPath      string
	modelLoaderPath string
	torchtitanPath  string
}

// NewTrainingManager creates a new training manager
func NewTrainingManager(config *MindletConfig, logger *Logger) (*TrainingManager, error) {
	return &TrainingManager{
		config:          config,
		logger:          logger,
		scriptPath:      "./scripts/run_training.sh",
		modelLoaderPath: "../torchtitan/model_loader.py",
	}, nil
}

func (t *TrainingManager) SetScriptPath(path string) {
	t.scriptPath = path
}

// SetModelLoaderPath sets the path to the model_loader.py script
func (t *TrainingManager) SetModelLoaderPath(path string) {
	t.modelLoaderPath = path
}

// SetTorchTitanPath sets the path to the torchtitan directory
func (t *TrainingManager) SetTorchTitanPath(path string) {
	t.torchtitanPath = path
}

// processConfigOverrides processes the TorchTitan config override parameters and
// replaces any templated variables with their actual values
func processConfigOverrides(overrides []string, cfg *TrainingConfig) []string {
	result := make([]string, 0, len(overrides))
	for _, param := range overrides {
		// Replace template variables
		processed := param
		processed = strings.ReplaceAll(processed, "${MODEL_PATH}", cfg.ModelPath)
		processed = strings.ReplaceAll(processed, "${DATASET_PATH}", cfg.DatasetPath)
		processed = strings.ReplaceAll(processed, "${TOKENIZER_PATH}", cfg.TokenizerPath) // Add this line
		processed = strings.ReplaceAll(processed, "${OUTPUT_DIR}", cfg.OutputDir)
		processed = strings.ReplaceAll(processed, "${RUN_ID}", cfg.TensorPreload.RunID)

		// Handle boolean flags properly (those ending with =true or =false)
		if strings.HasSuffix(processed, "=true") {
			// For boolean flags that are true, just include the flag name without the =true part
			processed = strings.TrimSuffix(processed, "=true")
			result = append(result, processed)
		} else if strings.HasSuffix(processed, "=false") {
			// For boolean flags that are false, skip them entirely
			// Do nothing - don't add to result
		} else {
			// For normal parameters, keep as is
			result = append(result, processed)
		}
	}
	return result
}

// LoadTrainingConfig loads a training configuration from a file
func (t *TrainingManager) LoadTrainingConfig(path string) (*TrainingConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read training config: %v", err)
	}

	var config TrainingConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse training config: %v", err)
	}

	// Check for required fields
	if config.ModelPath == "" {
		return nil, fmt.Errorf("model_path is required")
	}
	if config.DatasetPath == "" {
		return nil, fmt.Errorf("dataset_path is required")
	}
	if config.TokenizerPath == "" {
		return nil, fmt.Errorf("tokenizer_path is required")
	}
	if config.ModelName == "" {
		return nil, fmt.Errorf("model_name is required")
	}
	if config.NodeIP == "" {
		return nil, fmt.Errorf("node_ip is required")
	}
	if config.Rank < 0 {
		return nil, fmt.Errorf("rank must be >= 0")
	}
	if config.WorldSize <= 0 {
		return nil, fmt.Errorf("world_size must be > 0")
	}
	if config.TorchTitanCfg.ConfigPath == "" {
		return nil, fmt.Errorf("torchtitan_config.config_path is required")
	}

	// Set defaults if not specified
	if config.TensorPreload.RedisHost == "" {
		config.TensorPreload.RedisHost = config.NodeTopology.Head
	}
	if config.TensorPreload.RedisPort == 0 {
		config.TensorPreload.RedisPort = 6379
	}
	if config.TensorPreload.Threads == 0 {
		config.TensorPreload.Threads = 64
	}
	if config.OutputDir == "" {
		config.OutputDir = "/mnt/nfs_shared/output"
	}
	if config.NFSPath == "" {
		config.NFSPath = "/mnt/nfs_shared"
	}

	// Verify TorchTitan config file exists
	if _, err := os.Stat(config.TorchTitanCfg.ConfigPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("TorchTitan config file not found: %s", config.TorchTitanCfg.ConfigPath)
	}

	// Log the configuration details
	t.logger.Info("Training", "Using TorchTitan config file: %s", config.TorchTitanCfg.ConfigPath)

	// Process the override parameters
	processedParams := processConfigOverrides(config.TorchTitanCfg.OverrideParams, &config)
	t.logger.Info("Training", "TorchTitan override parameters: %s", strings.Join(processedParams, " "))

	return &config, nil
}

// copyMetadataToNFS copies the metadata file from model path to NFS shared location
func (t *TrainingManager) copyMetadataToNFS(trainingCfg *TrainingConfig) error {
	// Source metadata file
	sourceMetadata := filepath.Join(trainingCfg.ModelPath, ".metadata")

	// Destination metadata file on NFS
	destMetadata := filepath.Join(trainingCfg.NFSPath, "model", ".metadata")

	t.logger.Info("Training", "Copying metadata from %s to %s", sourceMetadata, destMetadata)

	// Check if source metadata exists
	if _, err := os.Stat(sourceMetadata); os.IsNotExist(err) {
		return fmt.Errorf("source metadata file not found: %s", sourceMetadata)
	}

	// Create destination directory if it doesn't exist
	destDir := filepath.Dir(destMetadata)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory %s: %v", destDir, err)
	}

	// Copy the metadata file
	sourceFile, err := os.Open(sourceMetadata)
	if err != nil {
		return fmt.Errorf("failed to open source metadata file: %v", err)
	}
	defer sourceFile.Close()

	destFile, err := os.Create(destMetadata)
	if err != nil {
		return fmt.Errorf("failed to create destination metadata file: %v", err)
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, sourceFile); err != nil {
		return fmt.Errorf("failed to copy metadata file: %v", err)
	}

	t.logger.Info("Training", "Successfully copied metadata to NFS shared location")
	return nil
}

// StartTensorPreloader starts the tensor preloader process in a managed goroutine
func (t *TrainingManager) StartTensorPreloader(ctx context.Context, trainingCfg *TrainingConfig) error {
	if !trainingCfg.TensorPreload.Enabled {
		t.logger.Info("Training", "Tensor preloading is disabled")
		return nil
	}

	// Use the run ID from the config (generated by orchestration)
	runID := trainingCfg.TensorPreload.RunID
	if runID == "" {
		runID = fmt.Sprintf("train-%s", GenerateUUID())
		t.logger.Warn("Training", "No run ID provided, generated: %s", runID)
	}
	t.runID = runID

	// If this is the head node, ensure Redis is running and copy metadata to NFS
	if trainingCfg.Rank == 0 {
		// Start Redis first
		if err := ensureRedisRunning(t.logger, trainingCfg.TensorPreload.RedisPort); err != nil {
			return fmt.Errorf("failed to ensure Redis is running: %w", err)
		}
		t.logger.Info("Training", "Redis server is ready on port %d", trainingCfg.TensorPreload.RedisPort)

		// Copy metadata file to NFS shared location
		if err := t.copyMetadataToNFS(trainingCfg); err != nil {
			return fmt.Errorf("failed to copy metadata to NFS: %w", err)
		}
	} else {
		// For worker nodes, wait to ensure head node has Redis up and running
		t.logger.Info("Training", "Worker node: waiting for Redis to be available on %s:%d",
			trainingCfg.TensorPreload.RedisHost, trainingCfg.TensorPreload.RedisPort)

		// Try to ping Redis to ensure it's ready
		maxRetries := 30 // 30 seconds max wait
		for i := 0; i < maxRetries; i++ {
			if isRedisAvailable(trainingCfg.TensorPreload.RedisHost, trainingCfg.TensorPreload.RedisPort) {
				t.logger.Info("Training", "Successfully connected to Redis on %s:%d",
					trainingCfg.TensorPreload.RedisHost, trainingCfg.TensorPreload.RedisPort)
				break
			}
			if i == maxRetries-1 {
				return fmt.Errorf("failed to connect to Redis after %d attempts", maxRetries)
			}
			time.Sleep(1 * time.Second)
			t.logger.Info("Training", "Waiting for Redis server... (attempt %d/%d)", i+1, maxRetries)
		}

		// Wait for metadata file to be available on NFS
		metadataPath := filepath.Join(trainingCfg.NFSPath, "model", ".metadata")
		t.logger.Info("Training", "Worker node: waiting for metadata file at %s", metadataPath)

		maxMetadataRetries := 30 // 30 seconds max wait
		for i := 0; i < maxMetadataRetries; i++ {
			if _, err := os.Stat(metadataPath); err == nil {
				t.logger.Info("Training", "Metadata file found on NFS")
				break
			}
			if i == maxMetadataRetries-1 {
				return fmt.Errorf("metadata file not found on NFS after %d attempts", maxMetadataRetries)
			}
			time.Sleep(1 * time.Second)
			t.logger.Info("Training", "Waiting for metadata file... (attempt %d/%d)", i+1, maxMetadataRetries)
		}
	}

	// Use a specific path for model_loader.py
	modelLoaderPath := t.modelLoaderPath

	// Check if the model loader exists
	if _, err := os.Stat(modelLoaderPath); os.IsNotExist(err) {
		return fmt.Errorf("model_loader.py not found at %s - please ensure it exists", modelLoaderPath)
	}

	t.logger.Info("Training", "Using model loader at: %s", modelLoaderPath)

	// Construct the preloader command
	preloaderMutex.Lock()

	// Build command arguments
	args := []string{
		modelLoaderPath,
		trainingCfg.ModelPath,
		"--threads", fmt.Sprintf("%d", trainingCfg.TensorPreload.Threads),
		"--rank", fmt.Sprintf("%d", trainingCfg.Rank),
		"--world-size", fmt.Sprintf("%d", trainingCfg.WorldSize),
		"--redis-host", trainingCfg.TensorPreload.RedisHost,
		"--redis-port", fmt.Sprintf("%d", trainingCfg.TensorPreload.RedisPort),
		"--run-id", runID,
	}

	// Add metadata-path for worker nodes only
	if trainingCfg.Rank > 0 {
		metadataPath := filepath.Join(trainingCfg.NFSPath, "model", ".metadata")
		args = append(args, "--metadata-path", metadataPath)
		t.logger.Info("Training", "Worker node using NFS metadata path: %s", metadataPath)
	}

	preloaderCmd = exec.Command(t.config.PythonPath, args...)
	preloaderMutex.Unlock()

	// Connect stdout and stderr for logging
	stdout, err := preloaderCmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderr, err := preloaderCmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	t.logger.Info("Training", "Starting tensor preloader with command: %s", preloaderCmd.String())

	// Start command
	if err := preloaderCmd.Start(); err != nil {
		return fmt.Errorf("failed to start tensor preloader: %v", err)
	}

	// Create a goroutine to monitor the process
	go func() {
		// Create channels for the process completion and context cancellation
		done := make(chan error, 1)
		go func() {
			done <- preloaderCmd.Wait()
		}()

		// Process stdout/stderr
		go processOutput(stdout, func(line string) {
			t.logger.Info("TensorLoader", "%s", line)
		})
		go processOutput(stderr, func(line string) {
			t.logger.Error("TensorLoader", "%s", line)
		})

		// Wait for either process completion or context cancellation
		select {
		case err := <-done:
			if err != nil {
				t.logger.Error("Training", "Tensor preloader exited with error: %v", err)
			} else {
				t.logger.Info("Training", "Tensor preloader completed successfully")
			}
		case <-ctx.Done():
			t.logger.Info("Training", "Context cancelled, killing tensor preloader")
			preloaderMutex.Lock()
			if preloaderCmd != nil && preloaderCmd.Process != nil {
				preloaderCmd.Process.Kill()
			}
			preloaderMutex.Unlock()
		}
	}()

	t.logger.Info("Training", "Tensor preloader started with PID %d", preloaderCmd.Process.Pid)
	return nil
}

// moveCheckpointToDCP moves the final checkpoint to the DCP directory
func (t *TrainingManager) moveCheckpointToDCP(trainingCfg *TrainingConfig) error {
	if trainingCfg.ModelName == "" {
		t.logger.Warn("Training", "No model name provided, skipping checkpoint move")
		return nil
	}

	// Only perform on head node (rank 0)
	if trainingCfg.Rank != 0 {
		t.logger.Info("Training", "Skipping checkpoint move on worker node (rank %d)", trainingCfg.Rank)
		return nil
	}

	// Extract base DCP directory from model path
	// Model path looks like: /mnt/cold/contents/dcp/llama-8b/checkpoint/step-0
	// We need to go up 3 levels to get to /mnt/cold/contents/dcp
	modelDir := filepath.Dir(trainingCfg.ModelPath) // removes step-0
	modelDir = filepath.Dir(modelDir)               // removes checkpoint
	dcpBaseDir := filepath.Dir(modelDir)            // removes model name, gives us /mnt/cold/contents/dcp

	// Define source and destination paths
	sourcePath := filepath.Join(trainingCfg.OutputDir, "checkpoint", "step-final")
	destBaseDir := filepath.Join(dcpBaseDir, trainingCfg.ModelName, "checkpoint")
	destPath := filepath.Join(destBaseDir, "step-0")

	t.logger.Info("Training", "Moving checkpoint from %s to %s", sourcePath, destPath)

	// Check if source exists
	if _, err := os.Stat(sourcePath); os.IsNotExist(err) {
		return fmt.Errorf("source checkpoint not found: %s", sourcePath)
	}

	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(destBaseDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %v", err)
	}

	// Check if destination already exists and remove it
	if _, err := os.Stat(destPath); err == nil {
		t.logger.Info("Training", "Removing existing checkpoint at: %s", destPath)
		if err := os.RemoveAll(destPath); err != nil {
			return fmt.Errorf("failed to remove existing checkpoint: %v", err)
		}
	}

	// Move the checkpoint directory
	if err := os.Rename(sourcePath, destPath); err != nil {
		return fmt.Errorf("failed to move checkpoint: %v", err)
	}

	t.logger.Info("Training", "Successfully moved checkpoint to %s", destPath)
	return nil
}

// StartTraining starts the training process
func (t *TrainingManager) StartTraining(ctx context.Context, trainingCfg *TrainingConfig) error {
	// Write initial status
	if err := writeTrainingStatus("running", trainingCfg.TensorPreload.RunID, trainingCfg.Rank, ""); err != nil {
		t.logger.Warn("Training", "Failed to write initial status: %v", err)
	}

	// Find the training script - use a direct path
	scriptPath := t.scriptPath

	// Check if script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("training script not found at %s", scriptPath)
	}

	// Process config override parameters
	processedParams := processConfigOverrides(trainingCfg.TorchTitanCfg.OverrideParams, trainingCfg)

	// Save the config to a temporary file
	tmpConfigPath := filepath.Join(os.TempDir(), "training_config_tmp.json")
	configData, err := json.MarshalIndent(trainingCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}
	if err := os.WriteFile(tmpConfigPath, configData, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %v", err)
	}

	// Extract Python directory to find torchrun
	pythonDir := filepath.Dir(t.config.PythonPath)
	torchrunPath := filepath.Join(pythonDir, "torchrun")

	// Prepare a temporary script to run TorchTitan with the correct parameters
	tmpScriptPath := filepath.Join(os.TempDir(), "run_torchtitan_tmp.sh")
	// In the orchestration mindlet.go, update the torchtitanCmd in startTraining function:

	torchtitanCmd := fmt.Sprintf(`#!/bin/bash
set -e

# Export environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_IB_DISABLE=1

# Use the head node IP for rendezvous endpoint
head_node_ip="%s"

# Log the torchrun parameters for debugging
echo "=== TORCHRUN CONFIGURATION ==="
echo "Head Node IP: $head_node_ip"
echo "Number of Nodes: %d"
echo "Processes per Node: 8"
echo "Rendezvous ID: 101"
echo "Rendezvous Backend: c10d"
echo "Rendezvous Endpoint: $head_node_ip:29500"
echo "Config File: %s"
echo "Override Parameters: %s"
echo "TorchTitan Path: %s"
echo "================================"

# Change to TorchTitan directory - use the path from configuration
cd %s

# Run torchrun with dynamic rendezvous (matching the working approach)
%s \
    --nnodes=%d \
    --nproc_per_node=8 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$head_node_ip:29500" \
    train.py \
    --job.config_file="%s" \
    %s
`, trainingCfg.NodeTopology.Head, trainingCfg.WorldSize, trainingCfg.TorchTitanCfg.ConfigPath,
		strings.Join(processedParams, " "), t.torchtitanPath, t.torchtitanPath, torchrunPath,
		trainingCfg.WorldSize, trainingCfg.TorchTitanCfg.ConfigPath, strings.Join(processedParams, " "))

	if err := os.WriteFile(tmpScriptPath, []byte(torchtitanCmd), 0755); err != nil {
		return fmt.Errorf("failed to write temporary script: %v", err)
	}

	// Construct command to run the training script
	preloadFlag := "true"
	if !trainingCfg.TensorPreload.Enabled {
		preloadFlag = "false"
	}

	cmd := exec.Command(scriptPath,
		"--config", tmpConfigPath,
		"--preload", preloadFlag,
		"--script", tmpScriptPath)

	// Connect stdout and stderr for logging
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	t.logger.Info("Training", "Starting training script: %s", cmd.String())

	// Start the script
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start training script: %v", err)
	}

	// Process stdout/stderr
	go processOutput(stdout, func(line string) {
		t.logger.Info("Training", "%s", line)
	})
	go processOutput(stderr, func(line string) {
		t.logger.Error("Training", "%s", line)
	})

	// Get the process ID for monitoring
	pid := cmd.Process.Pid
	t.logger.Info("Training", "Training script started with PID %d", pid)

	// Setup a goroutine to monitor the training process
	go func() {
		// Wait for either process completion or context cancellation
		done := make(chan error, 1)
		go func() {
			done <- cmd.Wait()
		}()

		select {
		case err := <-done:
			if err != nil {
				t.logger.Error("Training", "Training script exited with error: %v", err)
				// Write error status
				if statusErr := writeTrainingStatus("error", trainingCfg.TensorPreload.RunID, trainingCfg.Rank, err.Error()); statusErr != nil {
					t.logger.Warn("Training", "Failed to write error status: %v", statusErr)
				}
			} else {
				t.logger.Info("Training", "Training script completed successfully")

				// Move checkpoint after successful completion
				if moveErr := t.moveCheckpointToDCP(trainingCfg); moveErr != nil {
					t.logger.Error("Training", "Failed to move checkpoint: %v", moveErr)
					// Write error status for checkpoint move failure
					if statusErr := writeTrainingStatus("error", trainingCfg.TensorPreload.RunID, trainingCfg.Rank, fmt.Sprintf("checkpoint move failed: %v", moveErr)); statusErr != nil {
						t.logger.Warn("Training", "Failed to write error status: %v", statusErr)
					}
				} else {
					// Write completed status only after successful checkpoint move
					if statusErr := writeTrainingStatus("completed", trainingCfg.TensorPreload.RunID, trainingCfg.Rank, ""); statusErr != nil {
						t.logger.Warn("Training", "Failed to write completed status: %v", statusErr)
					}
				}
			}

			// Clean up tensor preloader if it's still running
			preloaderMutex.Lock()
			if preloaderCmd != nil && preloaderCmd.Process != nil {
				t.logger.Info("Training", "Stopping tensor preloader after training completion")
				preloaderCmd.Process.Kill()
				preloaderCmd = nil
			}
			preloaderMutex.Unlock()

			// Clean up the temporary files
			os.Remove(tmpConfigPath)
			os.Remove(tmpScriptPath)

		case <-ctx.Done():
			t.logger.Info("Training", "Context cancelled, killing training process")
			if cmd.Process != nil {
				cmd.Process.Kill()
			}
			// Write error status for cancellation
			if statusErr := writeTrainingStatus("error", trainingCfg.TensorPreload.RunID, trainingCfg.Rank, "training cancelled"); statusErr != nil {
				t.logger.Warn("Training", "Failed to write cancelled status: %v", statusErr)
			}

			// Clean up the temporary files
			os.Remove(tmpConfigPath)
			os.Remove(tmpScriptPath)
		}
	}()

	return nil
}

// Helper function to process command output
func processOutput(r io.Reader, logFn func(string)) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		logFn(scanner.Text())
	}
}

// Helper function to ensure Redis is running
func ensureRedisRunning(logger *Logger, port int) error {
	// First check if Redis container already exists but is stopped
	checkExistingCmd := exec.Command("docker", "ps", "-a", "--filter", "name=redis", "--format", "{{.Status}}")
	output, err := checkExistingCmd.Output()
	if err == nil && len(output) > 0 {
		// If it exists, check if it's already running
		statusStr := strings.TrimSpace(string(output))
		logger.Info("Training", "Existing Redis container found with status: %s", statusStr)

		if !strings.HasPrefix(statusStr, "Up") {
			// Container exists but is not running
			// Try to remove it first
			logger.Info("Training", "Removing stopped Redis container...")
			removeCmd := exec.Command("docker", "rm", "redis")
			if removeErr := removeCmd.Run(); removeErr != nil {
				logger.Info("Training", "Failed to remove Redis container: %v", removeErr)
				// If we can't remove it, try to start it
				logger.Info("Training", "Attempting to start the existing container...")
				startCmd := exec.Command("docker", "start", "redis")
				if startErr := startCmd.Run(); startErr != nil {
					return fmt.Errorf("could not remove or start existing Redis container: %v", startErr)
				}
			}
		}
	}

	// Check if Redis is now running
	checkCmd := exec.Command("docker", "ps", "--filter", "name=redis", "--format", "{{.Names}}")
	output, err = checkCmd.Output()
	if err != nil {
		return fmt.Errorf("failed to check Redis status: %v", err)
	}

	if len(output) == 0 {
		// Start Redis
		logger.Info("Training", "Starting new Redis container...")
		startCmd := exec.Command("docker", "run", "--name", "redis", "-p", fmt.Sprintf("%d:6379", port), "-d", "redis")
		startOutput, err := startCmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to start Redis: %v (output: %s)", err, string(startOutput))
		}

		// Give Redis a moment to initialize
		time.Sleep(2 * time.Second)
	} else {
		// Flush Redis
		logger.Info("Training", "Flushing existing Redis database...")
		flushCmd := exec.Command("docker", "exec", "redis", "redis-cli", "FLUSHALL")
		if err := flushCmd.Run(); err != nil {
			return fmt.Errorf("failed to flush Redis: %v", err)
		}
	}

	// Verify Redis is actually running by attempting to connect
	return checkRedisConnection(logger, port)
}

// Helper function to check if Redis is running locally
func checkRedisConnection(logger *Logger, port int) error {
	// Try to ping Redis
	maxRetries := 5
	var lastError error

	for retry := 0; retry < maxRetries; retry++ {
		pingCmd := exec.Command("docker", "exec", "redis", "redis-cli", "ping")
		output, err := pingCmd.CombinedOutput()
		outputStr := strings.TrimSpace(string(output))

		if err != nil {
			lastError = fmt.Errorf("failed to ping Redis (attempt %d/%d): %v (output: %s)",
				retry+1, maxRetries, err, outputStr)
			logger.Info("Training", "%v", lastError)
			time.Sleep(1 * time.Second)
			continue
		}

		// "PONG" response indicates Redis is running
		if outputStr != "PONG" {
			lastError = fmt.Errorf("unexpected Redis ping response (attempt %d/%d): %s",
				retry+1, maxRetries, outputStr)
			logger.Info("Training", "%v", lastError)
			time.Sleep(1 * time.Second)
			continue
		}

		// Success
		logger.Info("Training", "Redis is responding correctly (PONG)")
		return nil
	}

	return lastError
}

// Helper function to check if Redis is available at the specified host and port
func isRedisAvailable(host string, port int) bool {
	// Use netcat to check if Redis port is open
	cmd := exec.Command("nc", "-z", "-w", "1", host, fmt.Sprintf("%d", port))
	err := cmd.Run()
	return err == nil
}

// writeTrainingStatus writes the current training status to a file
func writeTrainingStatus(status string, runID string, rank int, errorMsg string) error {
	statusData := TrainingStatus{
		Status:    status,
		StartTime: time.Now(),
		RunID:     runID,
		Rank:      rank,
	}

	if status == "completed" || status == "error" {
		statusData.EndTime = time.Now()
	}

	if errorMsg != "" {
		statusData.Error = errorMsg
	}

	// Read existing status file to preserve start time if updating
	existingFile := filepath.Join(os.Getenv("HOME"), "training_status.json")
	if existing, err := os.ReadFile(existingFile); err == nil {
		var existingStatus TrainingStatus
		if json.Unmarshal(existing, &existingStatus) == nil && existingStatus.RunID == runID {
			// Preserve original start time
			statusData.StartTime = existingStatus.StartTime
		}
	}

	data, err := json.MarshalIndent(statusData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal status: %v", err)
	}

	return os.WriteFile(existingFile, data, 0644)
}
