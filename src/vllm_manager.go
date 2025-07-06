// mindlet/src/vllm_manager.go
package src

import (
	"bufio"
	"bytes"
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

// tokenizeMessages sends a tokenize request to get token counts for messages
func (s *VLLMServer) tokenizeMessages(ctx context.Context, messages []Message) ([]int, error) {
	// Create tokenize request
	tokenReq := TokenizeRequest{
		Messages: messages,
	}

	jsonData, err := json.Marshal(tokenReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tokenize request: %w", err)
	}

	// Make request to VLLM tokenize endpoint
	endpoint := fmt.Sprintf("http://%s:%d/tokenize", s.Config.Host, s.Port)
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenize request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make tokenize request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("tokenize request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var tokenResp TokenizeResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return nil, fmt.Errorf("failed to decode tokenize response: %w", err)
	}

	return tokenResp.Tokens, nil
}

// logTokenCounts tokenizes messages individually and logs token counts
func (s *VLLMServer) logTokenCounts(ctx context.Context, messages []Message) (int, error) {
	log.Printf("=== Token Count Analysis ===")

	totalTokens := 0

	// Tokenize each message individually to get per-message counts
	for i, msg := range messages {
		tokens, err := s.tokenizeMessages(ctx, []Message{msg})
		if err != nil {
			log.Printf("Failed to tokenize message %d (role: %s): %v", i+1, msg.Role, err)
			continue
		}

		tokenCount := len(tokens)
		totalTokens += tokenCount

		// Truncate content for logging if it's too long
		contentPreview := msg.Content
		if len(contentPreview) > 100 {
			contentPreview = contentPreview[:97] + "..."
		}

		log.Printf("Message %d (role: %s): %d tokens - \"%s\"",
			i+1, msg.Role, tokenCount, contentPreview)
	}

	// Also tokenize all messages together to get the actual total
	// (which might be different due to special tokens between messages)
	allTokens, err := s.tokenizeMessages(ctx, messages)
	if err != nil {
		log.Printf("Failed to tokenize all messages together: %v", err)
		// Return the sum of individual messages as fallback
		return totalTokens, err
	}

	actualTotal := len(allTokens)
	log.Printf("Total tokens (all messages): %d", actualTotal)
	if actualTotal != totalTokens {
		log.Printf("Note: Combined tokenization differs by %d tokens (likely due to message separators)",
			actualTotal-totalTokens)
	}

	log.Printf("=========================")

	// Return the actual total from combined tokenization
	return actualTotal, nil
}

// Forward forwards inference requests to VLLM server (non-streaming)
func (s *VLLMServer) Forward(ctx context.Context, messages []Message, maxTokens int) (map[string]interface{}, error) {
	// Check if context is already cancelled
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("context already cancelled before request: %v", ctx.Err())
	default:
	}

	// Log token counts before sending inference request
	_, err := s.logTokenCounts(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("token count check failed: %w", err)
	}

	// Convert messages to VLLM format
	vllmReq := map[string]interface{}{
		"model":      "",
		"messages":   messages,
		"stream":     false,
		"max_tokens": maxTokens,
	}

	jsonData, err := json.Marshal(vllmReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Log request details
	log.Printf("Non-streaming inference request size: %d bytes (%.2f MB)",
		len(jsonData), float64(len(jsonData))/(1024*1024))

	// Make request to VLLM
	endpoint := fmt.Sprintf("http://%s:%d/v1/chat/completions", s.Config.Host, s.Port)

	// CRITICAL: Use context.Background() for HTTP request to prevent cancellation
	// Keep original context for monitoring
	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.ContentLength = int64(len(jsonData))

	// Create client with proper settings
	client := &http.Client{
		Timeout: 5 * time.Minute,
		Transport: &http.Transport{
			MaxIdleConns:           100,
			MaxIdleConnsPerHost:    100,
			IdleConnTimeout:        90 * time.Second,
			ResponseHeaderTimeout:  5 * time.Minute,
			ExpectContinueTimeout:  1 * time.Second,
			MaxResponseHeaderBytes: 1 << 20, // 1 MB
			WriteBufferSize:        1 << 20, // 1 MB
			ReadBufferSize:         1 << 20, // 1 MB
			DisableCompression:     true,
		},
	}

	log.Printf("Sending non-streaming request to VLLM at %s", endpoint)
	startTime := time.Now()

	// Monitor original context cancellation in background
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			log.Printf("WARNING: Original context cancelled during non-streaming request: %v", ctx.Err())
		case <-done:
			// Request completed normally
		}
	}()

	resp, err := client.Do(httpReq)
	close(done) // Signal that request completed

	if err != nil {
		log.Printf("Non-streaming request failed after %v: %v", time.Since(startTime), err)
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("Got non-streaming response after %v, status: %d", time.Since(startTime), resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("VLLM server returned %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
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

	// Log token counts before sending inference request
	_, err := s.logTokenCounts(ctx, messages)
	if err != nil {
		return fmt.Errorf("token count check failed: %w", err)
	}

	// Convert messages to VLLM format
	vllmReq := map[string]interface{}{
		"model":      "",
		"messages":   messages,
		"stream":     true,
		"max_tokens": maxTokens,
	}

	jsonData, err := json.Marshal(vllmReq)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Log request details
	log.Printf("Streaming inference request size: %d bytes (%.2f MB)",
		len(jsonData), float64(len(jsonData))/(1024*1024))

	// Make request to VLLM
	endpoint := fmt.Sprintf("http://%s:%d/v1/chat/completions", s.Config.Host, s.Port)

	// CRITICAL: Use context.Background() for HTTP request to prevent cancellation
	// This matches the pattern in session.go and prevents context cancellation issues
	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.ContentLength = int64(len(jsonData))

	// Create client with same settings as standalone version
	client := &http.Client{
		Timeout: 10 * time.Minute,
		Transport: &http.Transport{
			MaxIdleConns:           100,
			MaxIdleConnsPerHost:    100,
			IdleConnTimeout:        90 * time.Second,
			ResponseHeaderTimeout:  5 * time.Minute,
			ExpectContinueTimeout:  1 * time.Second,
			MaxResponseHeaderBytes: 1 << 20, // 1 MB
			WriteBufferSize:        1 << 20, // 1 MB
			ReadBufferSize:         1 << 20, // 1 MB
			DisableCompression:     true,
		},
	}

	log.Printf("Sending streaming request to VLLM at %s", endpoint)
	startTime := time.Now()

	// Optional: Save request for debugging
	if debugPath := os.Getenv("VLLM_DEBUG_PATH"); debugPath != "" {
		debugFile := fmt.Sprintf("%s/vllm_request_%d.json", debugPath, time.Now().Unix())
		if err := os.WriteFile(debugFile, jsonData, 0644); err == nil {
			log.Printf("DEBUG: Saved request to %s", debugFile)
		}
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Streaming request failed after %v: %v", time.Since(startTime), err)
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("Got streaming response after %v, status: %d", time.Since(startTime), resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("VLLM server returned %d: %s", resp.StatusCode, string(body))
	}

	// Read SSE stream
	reader := bufio.NewReader(resp.Body)
	chunkCount := 0

	for {
		// Check if original context is cancelled (for clean shutdown)
		select {
		case <-ctx.Done():
			log.Printf("Original context cancelled during streaming (after %d chunks)", chunkCount)
			return ctx.Err()
		default:
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				log.Printf("Stream ended normally after %d chunks", chunkCount)
				break
			}
			return fmt.Errorf("error reading stream after %d chunks: %w", chunkCount, err)
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Parse SSE data
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			// Check for end of stream
			if data == "[DONE]" {
				log.Printf("Received [DONE] signal after %d chunks", chunkCount)
				break
			}

			// Parse JSON chunk
			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				log.Printf("Error parsing chunk %d: %v, data: %s", chunkCount, err, data)
				continue
			}

			chunkCount++

			// Call the callback with the chunk
			if err := onChunk(chunk); err != nil {
				return fmt.Errorf("chunk handler error at chunk %d: %w", chunkCount, err)
			}
		}
	}

	log.Printf("Stream processing completed successfully after %d chunks, total time: %v",
		chunkCount, time.Since(startTime))
	return nil
}
