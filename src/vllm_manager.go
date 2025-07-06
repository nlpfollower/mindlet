// mindlet/src/vllm_manager.go
package src

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// TokenizeRequest represents a request to the VLLM tokenize endpoint
type TokenizeRequest struct {
	Messages []Message `json:"messages"`
}

// TokenizeResponse represents the response from VLLM tokenize endpoint
type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

// DetokenizeRequest represents a request to the VLLM detokenize endpoint
type DetokenizeRequest struct {
	Tokens []int `json:"tokens"`
}

// DetokenizeResponse represents the response from VLLM detokenize endpoint
type DetokenizeResponse struct {
	Prompt string `json:"prompt"`
}

type VLLMConfig struct {
	ModelPath          string
	ModelID            string
	Host               string
	Port               int
	TensorParallelSize int
	Dtype              string
	ChatTemplate       string
	GPUMemoryUtil      float64
	MaxModelLen        int
}

type VLLMServer struct {
	Config      VLLMConfig
	Process     *exec.Cmd
	Port        int
	Status      string
	ReadyChan   chan bool
	StartupLogs []string
	mu          sync.Mutex
	ctx         context.Context
	cancel      context.CancelFunc
}

type VLLMInfo struct {
	ModelID  string `json:"model_id"`
	Endpoint string `json:"endpoint"`
	Status   string `json:"status"`
	Port     int    `json:"port"`
}

type VLLMManager struct {
	config  *MindletConfig
	servers map[string]*VLLMServer
	mu      sync.RWMutex
	logger  *Logger
}

func NewVLLMManager(cfg *MindletConfig, logger *Logger) *VLLMManager {
	return &VLLMManager{
		config:  cfg,
		servers: make(map[string]*VLLMServer),
		logger:  logger,
	}
}

func (m *VLLMManager) StartServer(modelID string, config VLLMConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if server already running
	if server, exists := m.servers[modelID]; exists {
		if server.Status == "running" {
			return fmt.Errorf("VLLM server already running for model %s", modelID)
		}
	}

	// Find chat template
	chatTemplate := config.ChatTemplate
	if chatTemplate == "" {
		chatTemplate = m.findChatTemplate()
	}

	// Build VLLM command
	args := []string{
		"serve",
		config.ModelPath,
		"--dtype", config.Dtype,
		"--host", config.Host,
		"--port", fmt.Sprintf("%d", config.Port),
		"--tensor-parallel-size", fmt.Sprintf("%d", config.TensorParallelSize),
	}

	// Only add GPU memory util if specified (not 0)
	if config.GPUMemoryUtil > 0 {
		args = append(args, "--gpu-memory-utilization", fmt.Sprintf("%.2f", config.GPUMemoryUtil))
	}

	// Only add max model len if specified (not 0)
	if config.MaxModelLen > 0 {
		args = append(args, "--max-model-len", fmt.Sprintf("%d", config.MaxModelLen))
	}

	if chatTemplate != "" {
		args = append(args, "--chat-template", chatTemplate)
	}

	// Determine vllm path based on environment
	vllmPath := "vllm"
	if m.config.PythonPath != "" && strings.Contains(m.config.PythonPath, "anaconda3") {
		// For anaconda environments, use the vllm from the same environment
		envPath := strings.Split(m.config.PythonPath, "/bin/python")[0]
		vllmPath = filepath.Join(envPath, "bin", "vllm")
	}

	cmd := exec.Command(vllmPath, args...)

	// Set up environment for CUDA
	cmd.Env = os.Environ()
	// Add any necessary CUDA paths if not already in environment

	// Create pipes for output
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	log.Printf("Starting VLLM server: %s %s", vllmPath, strings.Join(args, " "))
	if m.logger != nil {
		m.logger.Info("VLLMManager", "Starting VLLM server for model %s: %s %s", modelID, vllmPath, strings.Join(args, " "))
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start VLLM server: %w", err)
	}

	// Create context for this server
	ctx, cancel := context.WithCancel(context.Background())

	server := &VLLMServer{
		Config:      config,
		Process:     cmd,
		Port:        config.Port,
		Status:      "starting",
		ReadyChan:   make(chan bool, 1),
		StartupLogs: make([]string, 0),
		ctx:         ctx,
		cancel:      cancel,
	}

	m.servers[modelID] = server

	// Monitor output in background
	go m.monitorServerOutput(modelID, stdout, stderr, server)

	// Monitor server startup in background with context
	go m.monitorServerStartup(modelID, server)

	// Wait for server to be ready with timeout
	log.Printf("Waiting for VLLM server to be ready...")
	if m.logger != nil {
		m.logger.Info("VLLMManager", "Waiting for VLLM server to be ready...")
	}

	select {
	case <-server.ReadyChan:
		log.Printf("VLLM server for model %s is ready", modelID)
		if m.logger != nil {
			m.logger.Info("VLLMManager", "VLLM server for model %s is ready", modelID)
		}
		return nil
	case <-ctx.Done():
		// Server was stopped while starting
		return fmt.Errorf("VLLM server startup cancelled")
	case <-time.After(5 * time.Minute):
		// Print startup logs for debugging
		server.mu.Lock()
		logs := server.StartupLogs
		server.mu.Unlock()

		log.Printf("VLLM server startup timeout. Last logs:")
		for _, line := range logs {
			log.Printf("  %s", line)
		}

		// Cancel and kill the process
		cancel()
		cmd.Process.Kill()
		delete(m.servers, modelID)
		return fmt.Errorf("VLLM server failed to start within timeout")
	}
}

func (m *VLLMManager) monitorServerOutput(modelID string, stdout, stderr io.ReadCloser, server *VLLMServer) {
	// Monitor stdout
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			select {
			case <-server.ctx.Done():
				return
			default:
				line := scanner.Text()

				// Store startup logs
				server.mu.Lock()
				if len(server.StartupLogs) < 100 {
					server.StartupLogs = append(server.StartupLogs, line)
				}
				server.mu.Unlock()

				// Check for ready indicators
				if strings.Contains(line, "Uvicorn running on") ||
					strings.Contains(line, "Application startup complete") ||
					strings.Contains(line, "Started server process") {
					select {
					case server.ReadyChan <- true:
					default:
					}
				}

				if m.logger != nil {
					m.logger.Info("VLLM-"+modelID, "stdout: %s", line)
				} else {
					log.Printf("VLLM[%s] stdout: %s", modelID, line)
				}
			}
		}
	}()

	// Monitor stderr
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			select {
			case <-server.ctx.Done():
				return
			default:
				line := scanner.Text()

				// Store startup logs
				server.mu.Lock()
				if len(server.StartupLogs) < 100 {
					server.StartupLogs = append(server.StartupLogs, line)
				}
				server.mu.Unlock()

				// Check for ready indicators in stderr too
				if strings.Contains(line, "Uvicorn running on") ||
					strings.Contains(line, "Application startup complete") {
					select {
					case server.ReadyChan <- true:
					default:
					}
				}

				if m.logger != nil {
					m.logger.Info("VLLM-"+modelID, "stderr: %s", line)
				} else {
					log.Printf("VLLM[%s] stderr: %s", modelID, line)
				}
			}
		}
	}()
}

func (m *VLLMManager) StopServer(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	server, exists := m.servers[modelID]
	if !exists {
		return fmt.Errorf("no VLLM server running for model %s", modelID)
	}

	// Cancel the context to stop all goroutines
	if server.cancel != nil {
		server.cancel()
	}

	if server.Process != nil && server.Process.Process != nil {
		log.Printf("Stopping VLLM server for model %s", modelID)

		// Try graceful shutdown first
		server.Process.Process.Signal(os.Interrupt)

		// Wait a bit for graceful shutdown
		done := make(chan error, 1)
		go func() {
			done <- server.Process.Wait()
		}()

		select {
		case <-done:
			// Process exited gracefully
		case <-time.After(10 * time.Second):
			// Force kill after timeout
			server.Process.Process.Kill()
		}
	}

	delete(m.servers, modelID)
	return nil
}

func (m *VLLMManager) StopAll() error {
	m.mu.Lock()
	modelIDs := make([]string, 0, len(m.servers))
	for modelID := range m.servers {
		modelIDs = append(modelIDs, modelID)
	}
	m.mu.Unlock()

	var errors []error
	for _, modelID := range modelIDs {
		if err := m.StopServer(modelID); err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors stopping servers: %v", errors)
	}
	return nil
}

func (m *VLLMManager) GetServer(modelID string) *VLLMServer {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.servers[modelID]
}

func (m *VLLMManager) GetRunningServers() []VLLMInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	info := make([]VLLMInfo, 0, len(m.servers))
	for modelID, server := range m.servers {
		info = append(info, VLLMInfo{
			ModelID:  modelID,
			Endpoint: fmt.Sprintf("http://%s:%d", server.Config.Host, server.Port),
			Status:   server.Status,
			Port:     server.Port,
		})
	}
	return info
}

func (m *VLLMManager) monitorServerStartup(modelID string, server *VLLMServer) {
	// Also check health endpoint as backup
	endpoint := fmt.Sprintf("http://%s:%d/health", server.Config.Host, server.Port)

	// Wait a bit before starting health checks
	select {
	case <-time.After(10 * time.Second):
	case <-server.ctx.Done():
		return
	}

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	timeout := time.After(5 * time.Minute)

	for {
		select {
		case <-server.ctx.Done():
			// Server was stopped, exit gracefully
			return
		case <-timeout:
			// Timeout reached
			m.mu.Lock()
			// Check if server still exists (wasn't stopped)
			if _, exists := m.servers[modelID]; exists {
				server.Status = "failed"
				m.mu.Unlock()
				log.Printf("VLLM server for model %s failed to start", modelID)
			} else {
				m.mu.Unlock()
			}
			return
		case <-ticker.C:
			resp, err := http.Get(endpoint)
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					m.mu.Lock()
					server.Status = "running"
					m.mu.Unlock()

					// Signal ready
					select {
					case server.ReadyChan <- true:
					default:
					}

					log.Printf("VLLM server for model %s confirmed ready via health check", modelID)
					return
				}
			}
		}
	}
}

func (m *VLLMManager) findChatTemplate() string {
	// Look for chat template in common locations
	possiblePaths := []string{
		"/home/nlpfollower/Desktop/deltamind/vllm/examples/tool_chat_template_llama3.2_json.jinja",
		"/home/ec2-user/workspace/vllm/examples/tool_chat_template_llama3.2_json.jinja",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}

// saveRequestToFile saves the VLLM request to a temporary file
func (s *VLLMServer) saveRequestToFile(messages []Message, stream bool, maxTokens int) (string, error) {
	// Create request
	vllmReq := map[string]interface{}{
		"model":      "",
		"messages":   messages,
		"stream":     stream,
		"max_tokens": maxTokens,
	}

	jsonData, err := json.Marshal(vllmReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create temp file
	tmpFile, err := os.CreateTemp("/tmp", "vllm_request_*.json")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer tmpFile.Close()

	// Write request to file
	if _, err := tmpFile.Write(jsonData); err != nil {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to write request to file: %w", err)
	}

	return tmpFile.Name(), nil
}

// executeReplayRequest runs the replay_request binary
func (s *VLLMServer) executeReplayRequest(requestFile string, stream bool) ([]byte, error) {
	// Look for replay_request binary in common locations
	replayPaths := []string{
		"./replay_request",
		"/usr/local/bin/replay_request",
		"/tmp/replay_request",
		// Add the path where you compiled it
	}

	var replayPath string
	for _, path := range replayPaths {
		if _, err := os.Stat(path); err == nil {
			replayPath = path
			break
		}
	}

	if replayPath == "" {
		return nil, fmt.Errorf("replay_request binary not found")
	}

	// Build command
	endpoint := fmt.Sprintf("http://%s:%d/v1/chat/completions", s.Config.Host, s.Port)
	args := []string{
		"-endpoint", endpoint,
		requestFile,
	}

	if stream {
		args = append(args, "-stream")
	}

	log.Printf("Executing: %s %s", replayPath, strings.Join(args, " "))

	// Execute the command
	cmd := exec.Command(replayPath, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("replay_request failed: %w, output: %s", err, string(output))
	}

	return output, nil
}

// Forward forwards inference requests to VLLM server (non-streaming)
func (s *VLLMServer) Forward(ctx context.Context, messages []Message, maxTokens int) (map[string]interface{}, error) {
	// Check if context is already cancelled
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("context already cancelled before request: %v", ctx.Err())
	default:
	}

	log.Printf("Using hacky file-based approach for non-streaming request")

	// Save request to file
	requestFile, err := s.saveRequestToFile(messages, false, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to save request: %w", err)
	}
	defer os.Remove(requestFile)

	log.Printf("Saved request to %s", requestFile)

	// Execute replay_request
	output, err := s.executeReplayRequest(requestFile, false)
	if err != nil {
		return nil, err
	}

	// Parse the output to extract JSON response
	// The output might have some log lines before the actual JSON
	lines := strings.Split(string(output), "\n")
	var jsonStart int
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "{") {
			jsonStart = i
			break
		}
	}

	if jsonStart >= len(lines) {
		return nil, fmt.Errorf("no JSON response found in output")
	}

	// Join the JSON lines
	jsonOutput := strings.Join(lines[jsonStart:], "\n")

	// Parse JSON response
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(jsonOutput), &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result, nil
}

// ForwardStream forwards streaming inference requests to VLLM server
func (s *VLLMServer) ForwardStream(ctx context.Context, messages []Message, maxTokens int, onChunk func(map[string]interface{}) error) error {
	// Check if context is already cancelled
	select {
	case <-ctx.Done():
		return fmt.Errorf("context already cancelled before request: %v", ctx.Err())
	default:
	}

	log.Printf("Using hacky file-based approach for streaming request")

	// Save request to file
	requestFile, err := s.saveRequestToFile(messages, true, maxTokens)
	if err != nil {
		return fmt.Errorf("failed to save request: %w", err)
	}
	defer os.Remove(requestFile)

	log.Printf("Saved streaming request to %s", requestFile)

	// For streaming, we need to execute replay_request and parse its output
	// This is more complex because we need to parse the streaming output

	// Build command
	endpoint := fmt.Sprintf("http://%s:%d/v1/chat/completions", s.Config.Host, s.Port)
	replayPath := "./replay_request" // Adjust path as needed

	cmd := exec.Command(replayPath, "-endpoint", endpoint, "-stream", requestFile)

	// Get stdout pipe
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start replay_request: %w", err)
	}

	// Read the streaming output
	scanner := bufio.NewScanner(stdout)
	inStreamSection := false
	currentContent := ""

	for scanner.Scan() {
		line := scanner.Text()

		// Look for the streaming response marker
		if strings.Contains(line, "--- Streaming Response ---") {
			inStreamSection = true
			continue
		}

		if strings.Contains(line, "[Stream completed]") {
			break
		}

		// If we're in the stream section, accumulate content
		if inStreamSection && line != "" {
			currentContent += line

			// Create a fake chunk that matches VLLM format
			chunk := map[string]interface{}{
				"choices": []interface{}{
					map[string]interface{}{
						"index": 0,
						"delta": map[string]interface{}{
							"content": line,
						},
					},
				},
			}

			// Send the chunk
			if err := onChunk(chunk); err != nil {
				cmd.Process.Kill()
				return fmt.Errorf("chunk handler error: %w", err)
			}
		}
	}

	// Wait for command to finish
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("replay_request failed: %w", err)
	}

	return nil
}
