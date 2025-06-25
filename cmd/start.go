package cmd

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
	"mindlet/src"
)

func newStartCommand() *cobra.Command {
	var (
		port                 int
		useVLLM              bool
		tensorParallelSize   int
		vllmHost             string
		vllmPort             int
		convertedModelsDir   string
		dcpModelsDir         string
		conversionScriptPath string
		pythonPath           string
		defaultTokenizerPath string
	)

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start the mindlet inference server",
		Long: `Start the mindlet server that manages model loading and inference.
Models are loaded automatically on-demand when inference requests are received.
DCP checkpoints are converted to safetensors format automatically when needed.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg := &src.MindletConfig{
				Port:                 port,
				UseVLLM:              useVLLM,
				TensorParallelSize:   tensorParallelSize,
				VLLMHost:             vllmHost,
				VLLMPort:             vllmPort,
				ConvertedModelsDir:   convertedModelsDir,
				DCPModelsDir:         dcpModelsDir,
				ConversionScriptPath: conversionScriptPath,
				PythonPath:           pythonPath,
				DefaultTokenizerPath: defaultTokenizerPath,
			}

			server, err := src.NewMindletServer(cfg)
			if err != nil {
				return fmt.Errorf("failed to create server: %w", err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// Handle shutdown gracefully
			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

			go func() {
				<-sigChan
				log.Println("Shutdown signal received")
				cancel()
			}()

			log.Printf("Starting mindlet server on port %d", port)
			if useVLLM {
				log.Printf("VLLM enabled with tensor parallel size %d", tensorParallelSize)
			}

			return server.Start(ctx)
		},
	}

	// Server configuration
	cmd.Flags().IntVar(&port, "port", 9090, "Mindlet server port")

	// VLLM configuration
	cmd.Flags().BoolVar(&useVLLM, "use-vllm", true, "Use VLLM for inference (recommended)")
	cmd.Flags().IntVar(&tensorParallelSize, "tensor-parallel-size", 8, "Tensor parallel size for VLLM")
	cmd.Flags().StringVar(&vllmHost, "host", "0.0.0.0", "VLLM server host")
	cmd.Flags().IntVar(&vllmPort, "vllm-port", 8000, "Base port for VLLM servers")

	// Model configuration
	cmd.Flags().StringVar(&convertedModelsDir, "converted-models-dir", "/opt/dlami/nvme/converted_models", "Directory for converted models")
	cmd.Flags().StringVar(&dcpModelsDir, "dcp-models-dir", "/mnt/cold/contents/dcp", "Directory containing DCP checkpoints")

	// Conversion configuration
	cmd.Flags().StringVar(&conversionScriptPath, "conversion-script", "/home/ec2-user/workspace/torchchat/dcp_to_safetensors.py", "Path to dcp_to_safetensors.py script")
	cmd.Flags().StringVar(&pythonPath, "python-path", "python3", "Path to Python executable")
	cmd.Flags().StringVar(&defaultTokenizerPath, "tokenizer-path", "/mnt/cold/contents/checkpoints/Llama3.1-8B-Instruct", "Default tokenizer path")

	return cmd
}
