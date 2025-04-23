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
		configFile string
		usePreload bool = true // Enabled by default
	)

	cmd := &cobra.Command{
		Use:   "train",
		Short: "Start a TorchTitan training job",
		Long: `Start a TorchTitan training job with tensor preloading.
This command loads training configuration from a JSON file and executes
the training process with optional tensor preloading.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg := src.DefaultMindletConfig()

			// Check if config file exists
			if _, err := os.Stat(configFile); os.IsNotExist(err) {
				return fmt.Errorf("config file not found: %s", configFile)
			}

			// Create logger
			logger, err := src.NewLogger()
			if err != nil {
				return fmt.Errorf("failed to create logger: %v", err)
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

			// Set up context with cancellation for cleanup
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel() // Ensure resources are cleaned up when we exit

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

	return cmd
}

func init() {
	rootCmd.AddCommand(newTrainCommand())
}
