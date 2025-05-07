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

	// Include only the flags that match the fields available in the simplified MindletConfig
	cmd.Flags().IntVar(&cfg.Port, "port", cfg.Port, "Server port")
	cmd.Flags().IntVar(&cfg.NumCheckpointsAhead, "num-checkpoints-ahead", cfg.NumCheckpointsAhead, "Number of checkpoints ahead")
	cmd.Flags().IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "Batch size for inference")
	cmd.Flags().StringVar(&cfg.ProjectRoot, "project-root", cfg.ProjectRoot, "Project root directory")
	cmd.Flags().StringVar(&cfg.RamFsRoot, "ramfs-root", cfg.RamFsRoot, "RamFS root directory")
	cmd.Flags().StringVar(&cfg.LogDir, "log-dir", cfg.LogDir, "Log directory")
	cmd.Flags().StringVar(&cfg.PythonPath, "python-path", cfg.PythonPath, "Python executable path")
	cmd.Flags().StringVar(&modelTypeStr, "model-type", "8b", "Model type (8b, 70b, or 405b)")

	return cmd
}
