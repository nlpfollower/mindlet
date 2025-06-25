package src

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// Helper function to get model paths based on environment
func getModelPath(modelName string) string {
	if os.Getenv("RUN_LOCAL") != "" {
		// Use local paths when RUN_LOCAL is set
		return fmt.Sprintf("/home/nlpfollower/Desktop/deltamind/torchtitan/outputs/models/%s/checkpoint", modelName)
	}
	// Use default paths for remote/production environment
	return fmt.Sprintf("/mnt/cold/contents/dcp/%s/checkpoint", modelName)
}

// Helper function to get model names and sizes based on environment
func getTestModels() (model1Name, model2Name, model1Size, model2Size string) {
	if os.Getenv("RUN_LOCAL") != "" {
		return "llama-8b", "llama-3b", "8B", "3B"
	}
	// Production uses 8B and 70B models
	return "llama-8b", "llama-70b", "8B", "70B"
}

// Helper function to get appropriate config based on environment
func getTestConfig() *MindletConfig {
	if os.Getenv("RUN_LOCAL") != "" {
		return LocalTestConfig()
	}
	// For production/remote testing, use default config but with test-specific adjustments
	cfg := DefaultMindletConfig()
	// Override some settings for testing
	cfg.Port = 19090     // Use different port to avoid conflicts
	cfg.VLLMPort = 18000 // Use different VLLM base port
	return cfg
}

// TestMindletServerIntegration tests the full mindlet server functionality
func TestMindletServerIntegration(t *testing.T) {
	// Use appropriate config based on environment
	cfg := getTestConfig()

	// Create server
	server, err := NewMindletServer(cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start server in a goroutine
	serverErr := make(chan error, 1)
	go func() {
		t.Log("Starting mindlet server...")
		serverErr <- server.Start(ctx)
	}()

	// Wait for server to be ready
	serverURL := fmt.Sprintf("http://localhost:%d", cfg.Port)
	if err := waitForServer(serverURL+"/health", 30*time.Second); err != nil {
		t.Fatalf("Server failed to start: %v", err)
	}
	t.Log("Server is ready")

	// Log which environment we're testing in
	if os.Getenv("RUN_LOCAL") != "" {
		t.Log("Running tests with LOCAL paths")
		t.Logf("Config: Port=%d, TensorParallelSize=%d, ConvertedModelsDir=%s",
			cfg.Port, cfg.TensorParallelSize, cfg.ConvertedModelsDir)
	} else {
		t.Log("Running tests with PRODUCTION paths")
		t.Logf("Config: Port=%d, TensorParallelSize=%d, ConvertedModelsDir=%s",
			cfg.Port, cfg.TensorParallelSize, cfg.ConvertedModelsDir)
	}

	// Run all tests
	t.Run("HealthCheck", func(t *testing.T) {
		testHealthCheck(t, serverURL)
	})

	t.Run("InferenceWithAutoLoad", func(t *testing.T) {
		// Test that model is auto-loaded on first inference
		testModelAutoLoad(t, serverURL)
	})

	// Run multiple models test. In production with TensorParallelSize=8, we need sufficient GPUs
	t.Run("MultipleModels", func(t *testing.T) {
		// Test loading multiple models
		testMultipleModels(t, serverURL)
	})

	// Test streaming
	t.Run("StreamingInference", func(t *testing.T) {
		testStreamingInference(t, serverURL)
	})

	// Test streaming with model switching
	t.Run("StreamingWithModelSwitch", func(t *testing.T) {
		testStreamingWithModelSwitch(t, serverURL)
	})

	// Cleanup
	t.Log("Shutting down server...")
	cancel()

	// Wait for server to shut down
	select {
	case err := <-serverErr:
		if err != nil && err != context.Canceled {
			t.Errorf("Server error: %v", err)
		}
	case <-time.After(10 * time.Second): // Longer timeout for production
		t.Error("Server failed to shut down within timeout")
	}
}

func waitForServer(healthURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}

	for time.Now().Before(deadline) {
		resp, err := client.Get(healthURL)
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("server failed to become ready within %v", timeout)
}

func testHealthCheck(t *testing.T, serverURL string) {
	resp, err := http.Get(serverURL + "/health")
	if err != nil {
		t.Fatalf("Failed to get health: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Health check failed with status: %d", resp.StatusCode)
	}

	var health map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatalf("Failed to decode health response: %v", err)
	}

	// Verify health response
	if status, ok := health["status"].(string); !ok || status != "healthy" {
		t.Errorf("Expected status 'healthy', got: %v", health["status"])
	}

	t.Logf("Health check passed: %+v", health)
}

func testModelAutoLoad(t *testing.T, serverURL string) {
	// First, verify no models are loaded
	health := getHealth(t, serverURL)
	models := health["models"].([]interface{})
	if len(models) != 0 {
		t.Errorf("Expected 0 models initially, got: %d", len(models))
	}

	// Make inference request - this should auto-load the model
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Say 'Hello, World!' and nothing else."},
	}

	// Get model info based on environment
	model1Name, _, _, _ := getTestModels()

	// Use environment-aware path
	modelPath := getModelPath(model1Name)

	t.Logf("Using model path: %s", modelPath)

	result := makeInferenceRequest(t, serverURL, model1Name, modelPath, messages)

	// Verify we got a response
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					t.Logf("Model response: %s", content)
				}
			}
		}
	}

	// Verify model is now loaded
	health = getHealth(t, serverURL)
	models = health["models"].([]interface{})
	if len(models) != 1 {
		t.Errorf("Expected 1 model after inference, got: %d", len(models))
	}

	// Verify VLLM server is running
	vllmServers := health["vllm"].([]interface{})
	if len(vllmServers) != 1 {
		t.Errorf("Expected 1 VLLM server, got: %d", len(vllmServers))
	}

	t.Log("Model auto-loading test passed")
}

func testMultipleModels(t *testing.T, serverURL string) {
	// Load first model if not already loaded
	messages := []Message{
		{Role: "user", Content: "Hello from test"},
	}

	// Get model info based on environment
	model1Name, model2Name, model1Size, model2Size := getTestModels()

	// Get model paths based on environment
	model1Path := getModelPath(model1Name)
	model2Path := getModelPath(model2Name)

	t.Logf("Testing with models: %s (%s) and %s (%s)", model1Name, model1Size, model2Name, model2Size)
	t.Logf("Loading first model from: %s", model1Path)
	t.Logf("Loading second model from: %s", model2Path)

	// Check current state
	health := getHealth(t, serverURL)
	initialModels := health["models"].([]interface{})
	t.Logf("Initial models loaded: %d", len(initialModels))

	// First model might already be loaded from previous test, but VLLM should switch
	result1 := makeInferenceRequestWithSize(t, serverURL, model1Name, model1Path, model1Size, messages)
	if _, ok := result1["choices"]; !ok {
		t.Errorf("Failed to get response from %s", model1Name)
	}

	// Give VLLM time to stabilize
	time.Sleep(1 * time.Second)

	// Load second model - this should stop the first VLLM and start a new one
	result2 := makeInferenceRequestWithSize(t, serverURL, model2Name, model2Path, model2Size, messages)
	if _, ok := result2["choices"]; !ok {
		t.Errorf("Failed to get response from %s", model2Name)
	}

	// Wait a bit for models to fully load
	time.Sleep(2 * time.Second)

	// Verify both models are loaded in model manager
	health = getHealth(t, serverURL)
	models := health["models"].([]interface{})
	t.Logf("Models loaded after requests: %d", len(models))

	// We expect at least 2 models in the model manager
	if len(models) < 2 {
		t.Errorf("Expected at least 2 models loaded, got: %d", len(models))
	}

	// Verify model IDs
	modelIDs := make(map[string]bool)
	for _, m := range models {
		if model, ok := m.(map[string]interface{}); ok {
			if id, ok := model["id"].(string); ok {
				modelIDs[id] = true
				t.Logf("Found loaded model: %s", id)
			}
		}
	}

	if !modelIDs[model1Name] || !modelIDs[model2Name] {
		t.Errorf("Expected both %s and %s to be loaded, got: %v", model1Name, model2Name, modelIDs)
	}

	// Check current model
	if currentModel, ok := health["current_model"].(string); ok {
		t.Logf("Current active model: %s", currentModel)
		if currentModel != model2Name {
			t.Errorf("Expected current model to be %s, got: %s", model2Name, currentModel)
		}
	}

	// Verify only ONE VLLM server is running (for the current model)
	vllmServers := health["vllm"].([]interface{})
	t.Logf("VLLM servers running: %d", len(vllmServers))

	if len(vllmServers) != 1 {
		t.Errorf("Expected exactly 1 VLLM server (for current model), got: %d", len(vllmServers))
		// Log VLLM server details for debugging
		for _, s := range vllmServers {
			if server, ok := s.(map[string]interface{}); ok {
				t.Logf("VLLM Server: %+v", server)
			}
		}
	}

	// Verify the running VLLM server is for the current model
	if len(vllmServers) > 0 {
		if server, ok := vllmServers[0].(map[string]interface{}); ok {
			if modelID, ok := server["model_id"].(string); ok && modelID != model2Name {
				t.Errorf("Expected VLLM server for %s, but found: %s", model2Name, modelID)
			}
		}
	}

	// Test switching back to first model
	t.Log("Testing switch back to first model...")
	result3 := makeInferenceRequestWithSize(t, serverURL, model1Name, model1Path, model1Size, messages)
	if _, ok := result3["choices"]; !ok {
		t.Errorf("Failed to get response from %s after switching back", model1Name)
	}

	// Verify VLLM switched
	time.Sleep(1 * time.Second)
	health = getHealth(t, serverURL)
	if currentModel, ok := health["current_model"].(string); ok {
		if currentModel != model1Name {
			t.Errorf("Expected current model to be %s after switch, got: %s", model1Name, currentModel)
		}
	}

	// Test switching again to second model (testing multiple switches)
	t.Log("Testing switch back to second model...")
	result4 := makeInferenceRequestWithSize(t, serverURL, model2Name, model2Path, model2Size, messages)
	if _, ok := result4["choices"]; !ok {
		t.Errorf("Failed to get response from %s on second switch", model2Name)
	}

	// Verify VLLM switched again
	time.Sleep(1 * time.Second)
	health = getHealth(t, serverURL)
	if currentModel, ok := health["current_model"].(string); ok {
		if currentModel != model2Name {
			t.Errorf("Expected current model to be %s after second switch, got: %s", model2Name, currentModel)
		}
	}

	t.Log("Multiple models test passed")
}

func testStreamingInference(t *testing.T, serverURL string) {
	// Get model info based on environment
	model1Name, _, _, _ := getTestModels()
	modelPath := getModelPath(model1Name)

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Count from 1 to 5, saying each number on a new line."},
	}

	t.Logf("Testing streaming with model: %s", model1Name)

	// Collect streamed chunks
	chunks := make([]map[string]interface{}, 0)
	err := makeStreamingRequest(t, serverURL, model1Name, modelPath, "8B", messages, func(chunk map[string]interface{}) {
		chunks = append(chunks, chunk)
		// Log chunk for debugging
		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok && content != "" {
						t.Logf("Streamed content: %q", content)
					}
				}
			}
		}
	})

	if err != nil {
		t.Fatalf("Streaming request failed: %v", err)
	}

	// Verify we got multiple chunks
	if len(chunks) < 2 {
		t.Errorf("Expected multiple chunks in stream, got: %d", len(chunks))
	}

	// Reconstruct full response
	var fullContent strings.Builder
	for _, chunk := range chunks {
		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok {
						fullContent.WriteString(content)
					}
				}
			}
		}
	}

	t.Logf("Full streamed response: %s", fullContent.String())
	t.Log("Streaming inference test passed")
}

func testStreamingWithModelSwitch(t *testing.T, serverURL string) {
	// Get model info based on environment
	model1Name, model2Name, model1Size, model2Size := getTestModels()
	model1Path := getModelPath(model1Name)
	model2Path := getModelPath(model2Name)

	messages := []Message{
		{Role: "user", Content: "Say 'hello' and nothing else."},
	}

	t.Logf("Testing streaming with model switching between %s and %s", model1Name, model2Name)

	// Stream from first model
	chunks1 := make([]string, 0)
	err := makeStreamingRequest(t, serverURL, model1Name, model1Path, model1Size, messages, func(chunk map[string]interface{}) {
		if content := extractStreamContent(chunk); content != "" {
			chunks1 = append(chunks1, content)
		}
	})

	if err != nil {
		t.Fatalf("Streaming request to %s failed: %v", model1Name, err)
	}

	// Verify we got response from first model
	if len(chunks1) == 0 {
		t.Errorf("No chunks received from %s", model1Name)
	}

	// Give time for switch
	time.Sleep(1 * time.Second)

	// Stream from second model (should trigger model switch)
	chunks2 := make([]string, 0)
	err = makeStreamingRequest(t, serverURL, model2Name, model2Path, model2Size, messages, func(chunk map[string]interface{}) {
		if content := extractStreamContent(chunk); content != "" {
			chunks2 = append(chunks2, content)
		}
	})

	if err != nil {
		t.Fatalf("Streaming request to %s failed: %v", model2Name, err)
	}

	// Verify we got response from second model
	if len(chunks2) == 0 {
		t.Errorf("No chunks received from %s", model2Name)
	}

	// Verify current model switched
	health := getHealth(t, serverURL)
	if currentModel, ok := health["current_model"].(string); ok {
		if currentModel != model2Name {
			t.Errorf("Expected current model to be %s after streaming, got: %s", model2Name, currentModel)
		}
	}

	t.Log("Streaming with model switch test passed")
}

// Helper function to extract content from a streaming chunk
func extractStreamContent(chunk map[string]interface{}) string {
	if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if delta, ok := choice["delta"].(map[string]interface{}); ok {
				if content, ok := delta["content"].(string); ok {
					return content
				}
			}
		}
	}
	return ""
}

func getHealth(t *testing.T, serverURL string) map[string]interface{} {
	resp, err := http.Get(serverURL + "/health")
	if err != nil {
		t.Fatalf("Failed to get health: %v", err)
	}
	defer resp.Body.Close()

	var health map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatalf("Failed to decode health: %v", err)
	}
	return health
}

func makeInferenceRequest(t *testing.T, serverURL, modelID, checkpointPath string, messages []Message) map[string]interface{} {
	return makeInferenceRequestWithSize(t, serverURL, modelID, checkpointPath, "8B", messages)
}

func makeInferenceRequestWithSize(t *testing.T, serverURL, modelID, checkpointPath, modelSize string, messages []Message) map[string]interface{} {
	reqBody := map[string]interface{}{
		"model_id":        modelID,
		"messages":        messages,
		"checkpoint_path": checkpointPath,
		"model_size":      modelSize,
		"max_tokens":      300,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	resp, err := http.Post(
		serverURL+"/api/inference",
		"application/json",
		bytes.NewReader(jsonData),
	)
	if err != nil {
		t.Fatalf("Failed to make inference request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("Inference failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	return result
}

func makeStreamingRequest(t *testing.T, serverURL, modelID, checkpointPath, modelSize string, messages []Message, onChunk func(map[string]interface{})) error {
	reqBody := map[string]interface{}{
		"model_id":        modelID,
		"messages":        messages,
		"checkpoint_path": checkpointPath,
		"model_size":      modelSize,
		"max_tokens":      300,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := http.Post(
		serverURL+"/api/inference/stream",
		"application/json",
		bytes.NewReader(jsonData),
	)
	if err != nil {
		return fmt.Errorf("failed to make streaming request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("streaming failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Verify we got SSE content type
	contentType := resp.Header.Get("Content-Type")
	if contentType != "text/event-stream" {
		return fmt.Errorf("expected content-type text/event-stream, got: %s", contentType)
	}

	// Read SSE stream
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error reading stream: %w", err)
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
				t.Log("Received stream termination signal")
				break
			}

			// Parse JSON chunk
			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				// Check if it's an error message
				if strings.Contains(data, "error") {
					return fmt.Errorf("stream error: %s", data)
				}
				t.Logf("Warning: failed to parse chunk: %v, data: %s", err, data)
				continue
			}

			// Call the chunk handler
			onChunk(chunk)
		}
	}

	return nil
}
