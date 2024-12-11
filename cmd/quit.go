package cmd

import (
	"github.com/spf13/cobra"
)

func newQuitCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "quit",
		Short: "Shutdown all servers",
		RunE: func(cmd *cobra.Command, args []string) error {
			return nil
		},
	}
}
