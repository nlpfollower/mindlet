package src

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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
	} else {
		t.Log("Running tests with PRODUCTION paths")
	}

	// Run all tests
	t.Run("HealthCheck", func(t *testing.T) {
		testHealthCheck(t, serverURL)
	})

	t.Run("InferenceWithAutoLoad", func(t *testing.T) {
		// Test that model is auto-loaded on first inference
		testModelAutoLoad(t, serverURL)
	})

	t.Run("MultipleModels", func(t *testing.T) {
		// Test loading multiple models
		testMultipleModels(t, serverURL)
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
	case <-time.After(5 * time.Second):
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

	// Use environment-aware path
	modelPath := getModelPath("llama-8b")

	t.Logf("Using model path: %s", modelPath)

	result := makeInferenceRequest(t, serverURL, "llama-8b", modelPath, messages)

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

	// Get model paths based on environment
	model1Path := getModelPath("llama-8b")
	model2Path := getModelPath("llama-3b")

	t.Logf("Loading first model from: %s", model1Path)
	t.Logf("Loading second model from: %s", model2Path)

	// Check current state
	health := getHealth(t, serverURL)
	initialModels := health["models"].([]interface{})
	t.Logf("Initial models loaded: %d", len(initialModels))

	// First model might already be loaded from previous test, but VLLM should switch
	result1 := makeInferenceRequest(t, serverURL, "llama-8b", model1Path, messages)
	if _, ok := result1["choices"]; !ok {
		t.Error("Failed to get response from llama-8b")
	}

	// Give VLLM time to stabilize
	time.Sleep(1 * time.Second)

	// Load second model - this should stop the first VLLM and start a new one
	result2 := makeInferenceRequestWithSize(t, serverURL, "llama-3b", model2Path, "3B", messages)
	if _, ok := result2["choices"]; !ok {
		t.Error("Failed to get response from llama-3b")
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

	if !modelIDs["llama-8b"] || !modelIDs["llama-3b"] {
		t.Errorf("Expected both llama-8b and llama-3b to be loaded, got: %v", modelIDs)
	}

	// Check current model
	if currentModel, ok := health["current_model"].(string); ok {
		t.Logf("Current active model: %s", currentModel)
		if currentModel != "llama-3b" {
			t.Errorf("Expected current model to be llama-3b, got: %s", currentModel)
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
			if modelID, ok := server["model_id"].(string); ok && modelID != "llama-3b" {
				t.Errorf("Expected VLLM server for llama-3b, but found: %s", modelID)
			}
		}
	}

	// Test switching back to first model
	t.Log("Testing switch back to first model...")
	result3 := makeInferenceRequest(t, serverURL, "llama-8b", model1Path, messages)
	if _, ok := result3["choices"]; !ok {
		t.Error("Failed to get response from llama-8b after switching back")
	}

	// Verify VLLM switched
	time.Sleep(1 * time.Second)
	health = getHealth(t, serverURL)
	if currentModel, ok := health["current_model"].(string); ok {
		if currentModel != "llama-8b" {
			t.Errorf("Expected current model to be llama-8b after switch, got: %s", currentModel)
		}
	}

	t.Log("Multiple models test passed")
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
