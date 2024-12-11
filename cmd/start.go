package cmd

import (
	"fmt"
	"github.com/spf13/cobra"
	"mindlet/src"
)

func newStartCommand() *cobra.Command {
	cfg := src.DefaultMindletConfig()
	var modelTypeStr string

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start inference and model servers",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg.ModelType = src.ModelType(modelTypeStr)
			switch cfg.ModelType {
			case src.Model8B, src.Model70B, src.Model405B:
				break
			default:
				return fmt.Errorf("invalid model type: %s", modelTypeStr)
			}
			return nil
		},
	}

	cmd.Flags().IntVar(&cfg.MaxSeqLength, "max-seq-length", cfg.MaxSeqLength, "Maximum sequence length")
	cmd.Flags().IntVar(&cfg.NumCheckpointsAhead, "num-checkpoints-ahead", cfg.NumCheckpointsAhead, "Number of checkpoints ahead")
	cmd.Flags().BoolVar(&cfg.StreamLogs, "stream-logs", cfg.StreamLogs, "Stream logs from servers")
	cmd.Flags().StringVar(&cfg.ProjectRoot, "project-root", cfg.ProjectRoot, "Project root directory")
	cmd.Flags().StringVar(&cfg.RamFsRoot, "ramfs-root", cfg.RamFsRoot, "RamFS root directory")
	cmd.Flags().StringVar(&cfg.BootDir, "boot-dir", cfg.BootDir, "Boot directory")
	cmd.Flags().StringVar(&cfg.OutputDir, "output-dir", cfg.OutputDir, "Output directory")
	cmd.Flags().StringVar(&cfg.ModelPath, "model-path", cfg.ModelPath, "Model path")
	cmd.Flags().StringVar(&cfg.TrainingDir, "training-dir", cfg.TrainingDir, "Training directory")
	cmd.Flags().StringVar(&cfg.ModelManagerDir, "model-manager-dir", cfg.ModelManagerDir, "Model manager directory")
	cmd.Flags().StringVar(&cfg.LogDir, "log-dir", cfg.LogDir, "Log directory")
	cmd.Flags().StringVar(&modelTypeStr, "model-type", "8b", "Model type (8b, 70b, or 405b)")

	return cmd
}
