package src

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"
)

func TestDatasetProcessing(t *testing.T) {
	// Use test config
	cfg := getTestConfig()
	cfg.Port = 19091
	cfg.VLLMPort = 18001

	// Create server
	server, err := NewMindletServer(cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start server
	go func() {
		server.Start(ctx)
	}()

	// Wait for server to be ready
	serverURL := fmt.Sprintf("http://localhost:%d", cfg.Port)
	if err := waitForServer(serverURL+"/health", 30*time.Second); err != nil {
		t.Fatalf("Server failed to start: %v", err)
	}

	// Load a model to ensure VLLM is running
	model1Name, _, model1Size, _ := getTestModels()
	modelPath := getModelPath(model1Name)

	// Make inference request to load model and start VLLM
	messages := []Message{{Role: "user", Content: "Hello"}}
	resultMap := makeInferenceRequestWithSize(t, serverURL, model1Name, modelPath, model1Size, messages)
	if _, ok := resultMap["choices"]; !ok {
		t.Fatalf("Failed to load model")
	}

	// Wait for VLLM to be ready
	time.Sleep(3 * time.Second)

	// Create a large dataset
	contextMessages := []Message{
		{Role: "system", Content: "You are an AI researcher."},
	}

	// Ask the model to generate diverse content to create a rich dataset
	generationPrompts := []string{
		"Write a detailed explanation of how photosynthesis works in plants, including the light and dark reactions.",
		"Create a short story about a robot who discovers emotions for the first time while working in a factory.",
		"Explain the history and cultural significance of the Great Wall of China.",
		"Describe the process of making traditional sourdough bread from scratch, including the science behind fermentation.",
		"Write about the discovery of penicillin and how it revolutionized medicine.",
		"Explain quantum entanglement in simple terms that a high school student could understand.",
		"Create a recipe for a three-course meal using only ingredients that would have been available in medieval times.",
		"Describe the lifecycle of a star from birth to death, including different possible endings.",
	}

	// Generate content for each prompt
	for i, prompt := range generationPrompts {
		t.Logf("Generating content %d/%d...", i+1, len(generationPrompts))

		// Ask the model to generate content
		genMessages := []Message{
			{Role: "user", Content: prompt},
		}

		genResult := makeInferenceRequestWithSize(t, serverURL, model1Name, modelPath, model1Size, genMessages)

		// Extract the generated content
		generatedContent := ""
		if choices, ok := genResult["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if message, ok := choice["message"].(map[string]interface{}); ok {
					if content, ok := message["content"].(string); ok {
						generatedContent = content
					}
				}
			}
		}

		if generatedContent == "" {
			t.Fatalf("Failed to generate content for prompt %d", i+1)
		}

		// Add as a user message with the generated content
		contextMessages = append(contextMessages, Message{
			Role:    "user",
			Content: generatedContent,
		})

		// Small delay to avoid overwhelming the server
		time.Sleep(500 * time.Millisecond)
	}

	dataset := TrainingDataset{
		ContextMessages: contextMessages,
		TrainingPrompt:  "Comprehensive understanding of modern machine learning techniques",
	}

	datasetName := fmt.Sprintf("test-dataset-%d", time.Now().Unix())

	// Create the request
	req := ProcessDatasetRequest{
		Dataset:     dataset,
		ModelID:     model1Name,
		DatasetName: datasetName,
	}

	// Measure processing time
	start := time.Now()

	// Send the request
	jsonData, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	client := &http.Client{Timeout: 15 * time.Minute}
	resp, err := client.Post(
		serverURL+"/api/dataset/process",
		"application/json",
		bytes.NewReader(jsonData),
	)
	if err != nil {
		t.Fatalf("Failed to send dataset processing request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response: %v", err)
	}

	duration := time.Since(start)
	t.Logf("Dataset processing took: %v", duration)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Dataset processing failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result ProcessDatasetResponse
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if !result.Success {
		t.Fatalf("Dataset processing failed: %s", result.Error)
	}

	t.Logf("Dataset processed successfully at: %s", result.DatasetPath)
}
