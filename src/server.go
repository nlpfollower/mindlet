package src

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

type ServerInfo struct {
	cmd        *exec.Cmd
	port       int
	name       string
	workingDir string
}

func StartServers(cfg *MindletConfig) error {
	runLogDir, err := setupLogging(cfg)
	if err != nil {
		return fmt.Errorf("error setting up logging: %v", err)
	}

	pythonPath := filepath.Join(cfg.ProjectRoot, cfg.BootDir, cfg.PythonPath)
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		return fmt.Errorf("python executable not found at %s", pythonPath)
	}

	modelManagerPath := filepath.Join(cfg.ProjectRoot, cfg.BootDir, cfg.ModelManagerDir)
	trainingPath := filepath.Join(cfg.ProjectRoot, cfg.BootDir, cfg.TrainingDir)

	serverPath := filepath.Join(modelManagerPath, "model_server.py")
	if _, err := os.Stat(serverPath); os.IsNotExist(err) {
		return fmt.Errorf("model_server.py not found at %s", serverPath)
	}

	// Create a context that we'll use to manage the lifecycle of our servers
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a WaitGroup to wait for all servers to finish
	var wg sync.WaitGroup

	// Start the model_manager server
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := startServer(ctx, cfg, ServerInfo{
			name:       cfg.ModelManagerDir,
			port:       8001,
			workingDir: modelManagerPath,
			cmd: exec.Command(pythonPath, "model_server.py",
				"--type", string(cfg.ModelType),
				"--num-checkpoints-ahead", fmt.Sprintf("%d", cfg.NumCheckpointsAhead),
				"--log-dir", filepath.Join(cfg.ProjectRoot, cfg.OutputDir, cfg.LogDir),
				"--src-dir", filepath.Join(cfg.ProjectRoot, cfg.ModelPath),
				"--dst-dir", cfg.RamFsRoot),
		}, runLogDir, cfg.StreamLogs); err != nil {
			fmt.Printf("Error starting model_manager server: %v\n", err)
			cancel()
			return
		}
	}()

	fmt.Println("Waiting for model_manager server to initialize...")
	for i := 0; i < 30; i++ {
		if isServerAvailable(8001) {
			fmt.Println("Model_manager server is available.")
			break
		}
		if i == 29 {
			return fmt.Errorf("timeout waiting for model_manager server to initialize")
		}
		time.Sleep(1 * time.Second)
	}

	// Start the inference server
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := startServer(ctx, cfg, ServerInfo{
			name:       cfg.TrainingDir,
			port:       8000,
			workingDir: trainingPath,
			cmd: exec.Command(pythonPath, "-m", "scripts.inference_server",
				"--model", "llama3",
				"--model_path", cfg.RamFsRoot,
				"--max_sequence_length", fmt.Sprintf("%d", cfg.MaxSeqLength),
				"--system_prompt", "You are a helpful AI assistant",
				"--debug",
				"--output_dir", filepath.Join(cfg.ProjectRoot, cfg.OutputDir, cfg.LogDir)),
		}, runLogDir, cfg.StreamLogs); err != nil {
			fmt.Printf("Error starting inference server: %v\n", err)
			cancel()
			return
		}
	}()

	fmt.Println("Waiting for inference server to initialize...")
	for i := 0; i < 30; i++ {
		if isServerAvailable(8000) {
			fmt.Println("Inference server is available.")
			break
		}
		if i == 29 {
			return fmt.Errorf("timeout waiting for inference server to initialize")
		}
		time.Sleep(1 * time.Second)
	}

	fmt.Println("Servers started in the background. Use 'quit' command to shutdown.")

	if cfg.StreamLogs {
		// Set up signal handling
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

		// Use a channel to signal when we should exit
		done := make(chan bool)

		go func() {
			for {
				select {
				case <-sigChan:
					fmt.Printf("\nReceived interrupt signal. Stopping log streaming. Servers continue to run in the background.\n")
					fmt.Println("Use 'quit' command to shutdown servers.")
					done <- true
					return
				case <-ctx.Done():
					fmt.Println("Context cancelled. Stopping log streaming.")
					done <- true
					return
				}
			}
		}()

		// Wait for the done signal
		<-done
	}

	return nil
}

func startServer(ctx context.Context, cfg *MindletConfig, info ServerInfo, runLogDir string, streamLogs bool) error {
	logFile, err := os.Create(filepath.Join(runLogDir, fmt.Sprintf("%s.log", info.name)))
	if err != nil {
		return fmt.Errorf("error creating log file for %s: %v", strings.ReplaceAll(info.name, ".py", ""), err)
	}
	defer logFile.Close()

	info.cmd.Dir = info.workingDir

	var prefix string
	if info.name == cfg.TrainingDir {
		prefix = fmt.Sprintf("[%s]", cfg.TrainingDir)
	} else if info.name == cfg.ModelManagerDir {
		prefix = fmt.Sprintf("[%s]", cfg.ModelManagerDir)
	} else {
		prefix = "[unknown]"
	}

	var writers []io.Writer
	writers = append(writers, NewPrefixedWriter(prefix, logFile))
	if streamLogs {
		writers = append(writers, NewPrefixedWriter(prefix, os.Stdout))
	}

	multiOut := &MultiWriter{writers: writers}
	info.cmd.Stdout = multiOut
	info.cmd.Stderr = multiOut

	fmt.Printf("Starting %s in directory: %s\n", info.name, info.workingDir)
	fmt.Printf("Command: %v\n", info.cmd.Args)

	info.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	err = info.cmd.Start()
	if err != nil {
		return fmt.Errorf("error starting %s: %v", info.name, err)
	}

	fmt.Printf("%s started with PID %d\n", info.name, info.cmd.Process.Pid)

	go func() {
		<-ctx.Done()
		if info.cmd.Process != nil {
			info.cmd.Process.Signal(syscall.SIGTERM)
		}
	}()

	err = info.cmd.Wait()
	if err != nil && err.Error() != "signal: terminated" {
		fmt.Printf("%s exited with error: %v\n", info.name, err)
	} else {
		fmt.Printf("%s exited successfully\n", info.name)
	}

	return nil
}

func ShutdownServers() error {
	client := &http.Client{Timeout: 5 * time.Second}

	servers := []ServerInfo{
		{name: "model_server.py", port: 8001},
		{name: "inference_server.py", port: 8000},
	}

	for _, server := range servers {
		if isServerRunning(server.port) {
			fmt.Printf("Sending shutdown request to %s...\n", server.name)
			_, err := client.Post(fmt.Sprintf("http://localhost:%d/shutdown", server.port), "application/json", nil)
			if err != nil && !strings.Contains(err.Error(), "EOF") {
				fmt.Printf("Error sending shutdown request to %s: %v\n", server.name, err)
			} else {
				fmt.Printf("Shutdown request sent to %s", server.name)
			}
		} else {
			fmt.Printf("%s is not running.\n", server.name)
		}
	}

	fmt.Println("Waiting for servers to shut down...")
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	for {
		allShutdown := true
		for _, server := range servers {
			if isServerRunning(server.port) {
				allShutdown = false
				break
			}
		}

		if allShutdown {
			fmt.Println("All servers have been shut down.")
			return nil
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for servers to shut down")
		case <-time.After(1 * time.Second):
			// Continue waiting
		}
	}
}
