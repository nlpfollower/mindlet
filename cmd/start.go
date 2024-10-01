package cmd

import (
	"github.com/spf13/cobra"
	"mindlet/src"
)

func newStartCommand() *cobra.Command {
	cfg := src.DefaultMindletConfig()

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start inference and model servers",
		RunE: func(cmd *cobra.Command, args []string) error {
			return src.StartServers(cfg)
		},
	}

	cmd.Flags().StringVar(&cfg.ProjectRoot, "project-root", cfg.ProjectRoot, "Project root directory")
	cmd.Flags().StringVar(&cfg.ChatModelDir, "chat-model-dir", cfg.ChatModelDir, "Chat model directory name")
	cmd.Flags().StringVar(&cfg.ModelManagerDir, "model-manager-dir", cfg.ModelManagerDir, "Model manager directory name")
	cmd.Flags().StringVar(&cfg.LogDir, "log-dir", cfg.LogDir, "Directory for storing logs")
	cmd.Flags().BoolVar(&cfg.StreamLogs, "stream-logs", false, "Stream logs from servers")

	return cmd
}
