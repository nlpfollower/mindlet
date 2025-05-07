package cmd

import "github.com/spf13/cobra"

var rootCmd = &cobra.Command{
	Use:   "server-manager",
	Short: "Manage inference and model servers",
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.AddCommand(newStartCommand())
	rootCmd.AddCommand(newQuitCommand())
	rootCmd.AddCommand(newTrainCommand())
}
