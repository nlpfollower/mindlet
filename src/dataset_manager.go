package src

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DatasetEpochs defines how many times to duplicate the dataset
	DatasetEpochs = 3
)

// DatasetManager handles dataset processing for training
type DatasetManager struct {
	vllmManager *VLLMManager
	logger      *Logger
	vllmHost    string
	vllmPort    int
}

// ProcessDatasetRequest represents the request to process a dataset
type ProcessDatasetRequest struct {
	Dataset     TrainingDataset `json:"dataset"`
	ModelID     string          `json:"model_id"`
	DatasetName string          `json:"dataset_name"`
}

// ProcessDatasetResponse represents the response from dataset processing
type ProcessDatasetResponse struct {
	Success     bool   `json:"success"`
	DatasetPath string `json:"dataset_path"`
	Error       string `json:"error,omitempty"`
}

// TrainingDataset represents the parsed training dataset
type TrainingDataset struct {
	ContextMessages []Message `json:"context_messages"`
	TrainingPrompt  string    `json:"training_prompt"`
}

// TrainingSample represents a single training sample
type TrainingSample struct {
	Prompt   string
	Response string
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

// NewDatasetProcessor creates a new dataset processor
func NewDatasetProcessor(vllmManager *VLLMManager, logger *Logger, vllmHost string, vllmPort int) *DatasetManager {
	// Initialize random seed for shuffling
	rand.Seed(time.Now().UnixNano())

	return &DatasetManager{
		vllmManager: vllmManager,
		logger:      logger,
		vllmHost:    vllmHost,
		vllmPort:    vllmPort,
	}
}

// ProcessDataset processes a dataset for training
func (dp *DatasetManager) ProcessDataset(ctx context.Context, req *ProcessDatasetRequest) (*ProcessDatasetResponse, error) {
	dp.logger.Info("DatasetManager", "Starting dataset processing for %s", req.DatasetName)

	// Extract user messages
	userMessages := dp.extractUserMessages(req.Dataset.ContextMessages)
	if len(userMessages) == 0 {
		return nil, fmt.Errorf("no user messages found in dataset")
	}

	dp.logger.Info("DatasetManager", "Extracted %d user messages", len(userMessages))

	// Tokenize all user messages and merge tokens
	var allTokens []int
	for i, msg := range userMessages {
		dp.logger.Info("DatasetManager", "Tokenizing user message %d/%d", i+1, len(userMessages))

		// Tokenize the message
		tokens, err := dp.tokenizeMessage(ctx, msg)
		if err != nil {
			dp.logger.Error("DatasetManager", "Failed to tokenize message: %v", err)
			continue
		}

		// Filter out tokens >= 128000
		filteredTokens := dp.filterTokens(tokens)

		// Merge tokens into the combined list
		allTokens = append(allTokens, filteredTokens...)
	}

	dp.logger.Info("DatasetManager", "Total tokens after merging: %d", len(allTokens))

	// Create chunks with sliding windows from the merged tokens
	chunks := dp.createChunks(allTokens)
	dp.logger.Info("DatasetManager", "Created %d chunks from merged tokens", len(chunks))

	// Process each chunk
	var allSamples []TrainingSample
	for j, chunk := range chunks {
		if len(chunk) == 0 {
			continue
		}

		dp.logger.Info("DatasetManager", "Processing chunk %d/%d", j+1, len(chunks))

		// Generate training sample for this chunk
		sample, err := dp.generateTrainingSample(ctx, chunk, req.Dataset.TrainingPrompt)
		if err != nil {
			dp.logger.Error("DatasetManager", "Failed to generate training sample for chunk %d: %v", j, err)
			continue
		}

		allSamples = append(allSamples, *sample)
	}

	dp.logger.Info("DatasetManager", "Generated %d training samples", len(allSamples))

	// Write samples to CSV files
	datasetPath, err := dp.writeSamplesToCSV(req.DatasetName, allSamples)
	if err != nil {
		return nil, fmt.Errorf("failed to write samples to CSV: %w", err)
	}

	dp.logger.Info("DatasetManager", "Successfully wrote dataset to %s", datasetPath)

	return &ProcessDatasetResponse{
		Success:     true,
		DatasetPath: datasetPath,
	}, nil
}

// extractUserMessages extracts only messages with role "user"
func (dp *DatasetManager) extractUserMessages(messages []Message) []Message {
	var userMessages []Message
	for _, msg := range messages {
		if msg.Role == "user" {
			userMessages = append(userMessages, msg)
		}
	}
	return userMessages
}

// tokenizeMessage tokenizes a single message using VLLM
func (dp *DatasetManager) tokenizeMessage(ctx context.Context, msg Message) ([]int, error) {
	// Create tokenize request
	tokenReq := TokenizeRequest{
		Messages: []Message{msg},
	}

	jsonData, err := json.Marshal(tokenReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tokenize request: %w", err)
	}

	// Make request to VLLM tokenize endpoint
	endpoint := fmt.Sprintf("http://%s:%d/tokenize", dp.vllmHost, dp.vllmPort)
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

// filterTokens removes tokens >= 128000 and removes the redundant prefix if present
func (dp *DatasetManager) filterTokens(tokens []int) []int {
	// Define the redundant prefix to remove
	prefix := []int{
		128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25,
		6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 2839, 10263,
		220, 2366, 20, 271, 128009, 128006, 882, 128007, 271,
	}

	// Check if tokens start with the prefix
	startIndex := 0
	if len(tokens) >= len(prefix) {
		hasPrefix := true
		for i, p := range prefix {
			if tokens[i] != p {
				hasPrefix = false
				break
			}
		}
		if hasPrefix {
			startIndex = len(prefix)
			dp.logger.Info("DatasetManager", "Removed redundant prefix of %d tokens", len(prefix))
		}
	}

	// Filter out tokens >= 128000, starting after the prefix if it was found
	var filtered []int
	for i := startIndex; i < len(tokens); i++ {
		if tokens[i] < 128000 {
			filtered = append(filtered, tokens[i])
		}
	}

	return filtered
}

// createChunks creates sliding window chunks of various sizes
func (dp *DatasetManager) createChunks(tokens []int) [][]int {
	var chunks [][]int

	// Define chunk sizes and their step sizes
	chunkConfigs := []struct {
		size int
		step int
	}{
		{1024, 512},
		{512, 256},
		{256, 128},
		{128, 64},
	}

	for _, config := range chunkConfigs {
		for start := 0; start+config.size <= len(tokens); start += config.step {
			end := start + config.size
			chunk := make([]int, config.size)
			copy(chunk, tokens[start:end])
			chunks = append(chunks, chunk)
		}
	}

	return chunks
}

// detokenize converts tokens back to text using VLLM
func (dp *DatasetManager) detokenize(ctx context.Context, tokens []int) (string, error) {
	detokenReq := DetokenizeRequest{
		Tokens: tokens,
	}

	jsonData, err := json.Marshal(detokenReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal detokenize request: %w", err)
	}

	endpoint := fmt.Sprintf("http://%s:%d/detokenize", dp.vllmHost, dp.vllmPort)
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create detokenize request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make detokenize request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("detokenize request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var detokenResp DetokenizeResponse
	if err := json.NewDecoder(resp.Body).Decode(&detokenResp); err != nil {
		return "", fmt.Errorf("failed to decode detokenize response: %w", err)
	}

	return detokenResp.Prompt, nil
}

// generateTrainingSample generates a training sample from a chunk
func (dp *DatasetManager) generateTrainingSample(ctx context.Context, chunk []int, trainingPrompt string) (*TrainingSample, error) {
	// First, detokenize the chunk to get the dataset text
	datasetText, err := dp.detokenize(ctx, chunk)
	if err != nil {
		return nil, fmt.Errorf("failed to detokenize chunk: %w", err)
	}

	// Create the learning prompt
	learningPrompt := fmt.Sprintf(
		"Your task is to create a short list of notes that relate the attached learning prompt. "+
			"The list of notes is meant to help you recall the dataset, which is also attached, "+
			"in the context of the learning prompt.\n\n"+
			"Training Prompt: %s\n\n"+
			"Dataset: %s",
		trainingPrompt, datasetText)

	// Get the learning notes from VLLM
	learningNotes, err := dp.generateLearningNotes(ctx, learningPrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate learning notes: %w", err)
	}

	// Shuffle the tokens for the training prompt
	shuffledTokens := make([]int, len(chunk))
	copy(shuffledTokens, chunk)
	rand.Shuffle(len(shuffledTokens), func(i, j int) {
		shuffledTokens[i], shuffledTokens[j] = shuffledTokens[j], shuffledTokens[i]
	})

	// Detokenize the shuffled tokens
	shuffledData, err := dp.detokenize(ctx, shuffledTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to detokenize shuffled tokens: %w", err)
	}

	// Create the training sample
	prompt := fmt.Sprintf(
		"This message contains shuffled data that includes a piece of information, "+
			"followed by a learning list that you created based on this Training Prompt: %s. "+
			"Your task is to unshuffle the data and your notes. "+
			"Do so in the format [Data] unshuffled data\n\n[Notes] unshuffled notes.\n\n"+
			"[Shuffled Data]: %s\n\n[Shuffled Notes]: %s",
		trainingPrompt, shuffledData, dp.shuffleText(learningNotes))

	response := fmt.Sprintf(
		"[Data] %s\n\n[Notes] %s",
		datasetText, learningNotes)

	return &TrainingSample{
		Prompt:   prompt,
		Response: response,
	}, nil
}

// shuffleText shuffles the words in a text
func (dp *DatasetManager) shuffleText(text string) string {
	words := strings.Fields(text)
	rand.Shuffle(len(words), func(i, j int) {
		words[i], words[j] = words[j], words[i]
	})
	return strings.Join(words, " ")
}

// generateLearningNotes generates learning notes using VLLM inference
func (dp *DatasetManager) generateLearningNotes(ctx context.Context, prompt string) (string, error) {
	// Create inference request
	inferenceReq := map[string]interface{}{
		"messages": []Message{
			{Role: "user", Content: prompt},
		},
		"max_tokens": 200,
		"stream":     false,
	}

	jsonData, err := json.Marshal(inferenceReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal inference request: %w", err)
	}

	endpoint := fmt.Sprintf("http://%s:%d/v1/chat/completions", dp.vllmHost, dp.vllmPort)
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create inference request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make inference request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("inference request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode inference response: %w", err)
	}

	// Extract the generated text
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					return content, nil
				}
			}
		}
	}

	return "", fmt.Errorf("failed to extract content from inference response")
}

// writeSamplesToCSV writes training samples to CSV files
func (dp *DatasetManager) writeSamplesToCSV(datasetName string, samples []TrainingSample) (string, error) {
	// Determine base directory based on environment
	baseDir := "/mnt/cold/contents/datasets"
	if os.Getenv("RUN_LOCAL") != "" {
		baseDir = "/home/nlpfollower/Desktop/deltamind/torchtitan/outputs/datasets"
	}

	// Create dataset directory
	datasetDir := filepath.Join(baseDir, datasetName)
	if err := os.MkdirAll(datasetDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create dataset directory: %w", err)
	}

	// Write file_test.csv (header only)
	testPath := filepath.Join(datasetDir, "file_test.csv")
	testFile, err := os.Create(testPath)
	if err != nil {
		return "", fmt.Errorf("failed to create test file: %w", err)
	}
	defer testFile.Close()

	testWriter := csv.NewWriter(testFile)
	if err := testWriter.Write([]string{"prompt", "response"}); err != nil {
		return "", fmt.Errorf("failed to write test header: %w", err)
	}
	testWriter.Flush()

	// Write file_train.csv (with all samples)
	trainPath := filepath.Join(datasetDir, "file_train.csv")
	trainFile, err := os.Create(trainPath)
	if err != nil {
		return "", fmt.Errorf("failed to create train file: %w", err)
	}
	defer trainFile.Close()

	trainWriter := csv.NewWriter(trainFile)

	// Write header
	if err := trainWriter.Write([]string{"prompt", "response"}); err != nil {
		return "", fmt.Errorf("failed to write train header: %w", err)
	}

	// Write all samples, repeated for each epoch
	for epoch := 0; epoch < DatasetEpochs; epoch++ {
		dp.logger.Info("DatasetManager", "Writing epoch %d/%d", epoch+1, DatasetEpochs)

		for _, sample := range samples {
			if err := trainWriter.Write([]string{sample.Prompt, sample.Response}); err != nil {
				return "", fmt.Errorf("failed to write training sample: %w", err)
			}
		}
	}

	trainWriter.Flush()

	dp.logger.Info("DatasetManager", "Total samples written: %d (base samples: %d × %d epochs)",
		len(samples)*DatasetEpochs, len(samples), DatasetEpochs)

	return datasetDir, nil
}
