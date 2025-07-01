package cmd

import (
	"context"
	"fmt"
	"github.com/spf13/cobra"
	"mindlet/src"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
)

func newTrainCommand() *cobra.Command {
	var (
		configFile      string
		usePreload      bool   = true // Enabled by default
		pythonPath      string        // Allow overriding the Python path
		scriptPath      string        // Path to run_training.sh
		modelLoaderPath string        // Path to model_loader.py
		torchtitanPath  string        // Path to torchtitan directory
		modelName       string        // Model name for checkpoint directory
		nfsPath         string        // NFS shared path for metadata
	)

	cmd := &cobra.Command{
		Use:   "train",
		Short: "Start a TorchTitan training job",
		Long: `Start a TorchTitan training job with tensor preloading.
This command loads training configuration from a JSON file and executes
the training process with optional tensor preloading.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Create base config
			cfg := src.DefaultMindletConfig()

			if pythonPath != "" {
				cfg.PythonPath = pythonPath
			}

			// Check if config file exists
			if _, err := os.Stat(configFile); os.IsNotExist(err) {
				return fmt.Errorf("config file not found: %s", configFile)
			}

			// Create logger
			logger, err := src.NewLogger()
			if err != nil {
				return fmt.Errorf("failed to create logger: %v", err)
			}

			// Log configuration for debugging
			logger.Info("Training", "Using Python path: %s", cfg.PythonPath)
			logger.Info("Training", "Using config file: %s", configFile)

			if scriptPath != "" {
				logger.Info("Training", "Using custom training script: %s", scriptPath)
			}

			if modelLoaderPath != "" {
				logger.Info("Training", "Using custom model loader: %s", modelLoaderPath)
			}

			if torchtitanPath != "" {
				logger.Info("Training", "Using custom torchtitan path: %s", torchtitanPath)
			}

			if modelName != "" {
				logger.Info("Training", "Model name for checkpoint: %s", modelName)
			}

			if nfsPath != "" {
				logger.Info("Training", "Using NFS path: %s", nfsPath)
			}

			// Create training manager
			trainingManager, err := src.NewTrainingManager(cfg, logger)
			if err != nil {
				return fmt.Errorf("failed to create training manager: %v", err)
			}

			// Load training configuration
			trainingCfg, err := trainingManager.LoadTrainingConfig(configFile)
			if err != nil {
				return fmt.Errorf("failed to load training config: %v", err)
			}

			// Set model name if provided via CLI (overrides config)
			if modelName != "" {
				trainingCfg.ModelName = modelName
			}

			// Set NFS path if provided via CLI (overrides config)
			if nfsPath != "" {
				trainingCfg.NFSPath = nfsPath
			}

			// Set up context with cancellation for cleanup
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel() // Ensure resources are cleaned up when we exit

			// Set custom paths if provided
			if scriptPath != "" {
				trainingManager.SetScriptPath(scriptPath)
			}

			if modelLoaderPath != "" {
				trainingManager.SetModelLoaderPath(modelLoaderPath)
			}

			if torchtitanPath != "" {
				trainingManager.SetTorchTitanPath(torchtitanPath)
			}

			// Start tensor preloader if enabled
			if usePreload && trainingCfg.TensorPreload.Enabled {
				if err := trainingManager.StartTensorPreloader(ctx, trainingCfg); err != nil {
					return fmt.Errorf("failed to start tensor preloader: %v", err)
				}
				logger.Info("Training", "Tensor preloader started in background")
			} else {
				logger.Info("Training", "Tensor preloading is disabled")
				// Disable in config to ensure training doesn't try to use it
				trainingCfg.TensorPreload.Enabled = false
			}

			// Start training
			if err := trainingManager.StartTraining(ctx, trainingCfg); err != nil {
				return fmt.Errorf("failed to start training: %v", err)
			}

			logger.Info("Training", "Training started successfully")

			// Set up signal handling for graceful shutdown
			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

			// Wait for signal
			sig := <-sigChan
			logger.Info("Training", "Received signal %v, shutting down", sig)
			cancel() // Cancel the context to trigger cleanup

			// Give a moment for cleanup
			logger.Info("Training", "Cleanup complete")
			return nil
		},
	}

	// Get home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}
	defaultConfigPath := filepath.Join(homeDir, "training_config.json")

	// Add flags
	cmd.Flags().StringVar(&configFile, "config", defaultConfigPath, "Path to training configuration file")
	cmd.Flags().BoolVar(&usePreload, "preload", true, "Enable tensor preloading (true) or disable it (false)")
	cmd.Flags().StringVar(&pythonPath, "python-path", "", "Override Python executable path")
	cmd.Flags().StringVar(&scriptPath, "script-path", "./scripts/run_training.sh", "Path to run_training.sh script")
	cmd.Flags().StringVar(&modelLoaderPath, "model-loader-path", "../torchtitan/model_loader.py", "Path to model_loader.py script")
	cmd.Flags().StringVar(&torchtitanPath, "torchtitan-path", "/home/nlpfollower/Desktop/deltamind/torchtitan", "Path to torchtitan directory")
	cmd.Flags().StringVar(&modelName, "model-name", "", "Model name for checkpoint directory (e.g., llama-8b-u1-c2)")
	cmd.Flags().StringVar(&nfsPath, "nfs-path", "/mnt/nfs_shared", "NFS shared path for metadata sharing between nodes")

	return cmd
}
