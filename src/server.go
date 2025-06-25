// mindlet/src/server.go
package src

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type MindletServer struct {
	config       *MindletConfig
	modelManager *ModelManager
	vllmManager  *VLLMManager
	httpServer   *http.Server
	logger       *Logger
	mu           sync.RWMutex
	currentModel string // Track currently loaded model
}

func NewMindletServer(cfg *MindletConfig) (*MindletServer, error) {
	// Create logger
	logger, err := NewLogger()
	if err != nil {
		// Fall back to console logging
		logger = nil
		log.Printf("Warning: Failed to create logger: %v", err)
	}

	modelManager := NewModelManager(cfg, logger)
	vllmManager := NewVLLMManager(cfg, logger)

	server := &MindletServer{
		config:       cfg,
		modelManager: modelManager,
		vllmManager:  vllmManager,
		logger:       logger,
		currentModel: "",
	}

	// Setup HTTP server for health checks and API
	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.handleHealth)
	mux.HandleFunc("/api/inference", server.handleInference)

	server.httpServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Port),
		Handler: mux,
	}

	return server, nil
}

func (s *MindletServer) Start(ctx context.Context) error {
	// Log startup
	if s.logger != nil {
		s.logger.Info("MindletServer", "Starting mindlet server on port %d", s.config.Port)
		if s.config.UseVLLM {
			s.logger.Info("MindletServer", "VLLM enabled with tensor parallel size %d", s.config.TensorParallelSize)
		}
		s.logger.Info("MindletServer", "DCP models directory: %s", s.config.DCPModelsDir)
		s.logger.Info("MindletServer", "Converted models directory: %s", s.config.ConvertedModelsDir)
	}

	// Start HTTP server
	go func() {
		log.Printf("Starting mindlet HTTP server on port %d", s.config.Port)
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
			if s.logger != nil {
				s.logger.Error("MindletServer", "HTTP server error: %v", err)
			}
		}
	}()

	// Wait for context cancellation
	<-ctx.Done()

	if s.logger != nil {
		s.logger.Info("MindletServer", "Shutting down mindlet server...")
	}

	// Shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Stop VLLM if running
	if err := s.vllmManager.StopAll(); err != nil {
		log.Printf("Error stopping VLLM: %v", err)
		if s.logger != nil {
			s.logger.Error("MindletServer", "Error stopping VLLM: %v", err)
		}
	}

	// Close logger
	if s.logger != nil {
		s.logger.Close()
	}

	return s.httpServer.Shutdown(shutdownCtx)
}

func (s *MindletServer) switchToModel(modelID, checkpointPath, modelSize string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// If it's the same model, nothing to do
	if s.currentModel == modelID {
		if s.logger != nil {
			s.logger.Info("MindletServer", "Model %s is already loaded and active", modelID)
		}
		return nil
	}

	// Stop any running VLLM server first
	if s.currentModel != "" && s.config.UseVLLM {
		if s.logger != nil {
			s.logger.Info("MindletServer", "Stopping VLLM server for model %s before switching to %s", s.currentModel, modelID)
		}
		if err := s.vllmManager.StopServer(s.currentModel); err != nil {
			if s.logger != nil {
				s.logger.Warn("MindletServer", "Failed to stop VLLM server for %s: %v", s.currentModel, err)
			}
		}
		// Give it a moment to fully release resources
		time.Sleep(2 * time.Second)
	}

	// Load the new model
	if err := s.loadModel(modelID, checkpointPath, modelSize); err != nil {
		return err
	}

	// Update current model
	s.currentModel = modelID
	return nil
}

func (s *MindletServer) loadModel(modelID, checkpointPath, modelSize string) error {
	if s.logger != nil {
		s.logger.Info("MindletServer", "Loading model %s from %s (size: %s)", modelID, checkpointPath, modelSize)
	}

	// Check if model is already loaded in the model manager
	if s.modelManager.IsModelLoaded(modelID) {
		log.Printf("Model %s is already loaded in model manager", modelID)
		if s.logger != nil {
			s.logger.Info("MindletServer", "Model %s is already loaded in model manager", modelID)
		}

		// Still need to start VLLM if it's not running
		if s.config.UseVLLM && s.vllmManager.GetServer(modelID) == nil {
			modelInfo, _ := s.modelManager.GetModelInfo(modelID)
			return s.startVLLMForModel(modelID, modelInfo.ConvertedPath)
		}

		return nil
	}

	// Determine the source path
	sourcePath := checkpointPath
	if sourcePath == "" {
		// Default path construction
		sourcePath = filepath.Join(s.config.DCPModelsDir, modelID)
	}

	// Check if this is a DCP checkpoint that needs conversion
	isDCP := false
	if _, err := os.Stat(filepath.Join(sourcePath, "step-0")); err == nil {
		isDCP = true
		sourcePath = filepath.Join(sourcePath, "step-0")
	}

	if s.logger != nil {
		s.logger.Info("MindletServer", "Source path: %s, isDCP: %v", sourcePath, isDCP)
	}

	// Load the model with specified size
	modelInfo, err := s.modelManager.LoadModel(modelID, sourcePath, modelSize, isDCP)
	if err != nil {
		if s.logger != nil {
			s.logger.Error("MindletServer", "Failed to load model %s: %v", modelID, err)
		}
		return fmt.Errorf("failed to load model: %w", err)
	}

	// Start VLLM server if enabled
	if s.config.UseVLLM {
		if err := s.startVLLMForModel(modelID, modelInfo.ConvertedPath); err != nil {
			return err
		}
	}

	log.Printf("Successfully loaded model %s", modelID)
	if s.logger != nil {
		s.logger.Info("MindletServer", "Successfully loaded model %s", modelID)
	}
	return nil
}

func (s *MindletServer) startVLLMForModel(modelID, modelPath string) error {
	vllmConfig := VLLMConfig{
		ModelPath:          modelPath,
		ModelID:            modelID,
		Host:               s.config.VLLMHost,
		Port:               s.config.VLLMPort, // Always use the same port since we only run one at a time
		TensorParallelSize: s.config.TensorParallelSize,
		Dtype:              "auto",
		GPUMemoryUtil:      s.config.GPUMemoryUtil,
		MaxModelLen:        s.config.MaxModelLen,
	}

	if s.logger != nil {
		s.logger.Info("MindletServer", "Starting VLLM server for model %s on port %d", modelID, vllmConfig.Port)
	}

	if err := s.vllmManager.StartServer(modelID, vllmConfig); err != nil {
		if s.logger != nil {
			s.logger.Error("MindletServer", "Failed to start VLLM server for model %s: %v", modelID, err)
		}
		return fmt.Errorf("failed to start VLLM server: %w", err)
	}

	return nil
}

// HTTP Handlers

func (s *MindletServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	status := map[string]interface{}{
		"status":        "healthy",
		"models":        s.modelManager.GetLoadedModels(),
		"vllm":          s.vllmManager.GetRunningServers(),
		"current_model": s.currentModel,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (s *MindletServer) handleInference(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ModelID        string    `json:"model_id"`
		Messages       []Message `json:"messages"`
		CheckpointPath string    `json:"checkpoint_path,omitempty"`
		ModelSize      string    `json:"model_size,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Default model size if not specified
	if req.ModelSize == "" {
		req.ModelSize = "8B"
	}

	log.Printf("Received inference request for model: %s (size: %s)", req.ModelID, req.ModelSize)
	if s.logger != nil {
		s.logger.Info("MindletServer", "Received inference request for model: %s, checkpoint: %s, size: %s",
			req.ModelID, req.CheckpointPath, req.ModelSize)
	}

	// Use checkpoint path from request, or construct default path
	checkpointPath := req.CheckpointPath
	if checkpointPath == "" {
		checkpointPath = filepath.Join(s.config.DCPModelsDir, req.ModelID)
	}

	// Switch to the requested model (this will handle stopping old VLLM and starting new one)
	if err := s.switchToModel(req.ModelID, checkpointPath, req.ModelSize); err != nil {
		log.Printf("Failed to switch to model: %v", err)
		if s.logger != nil {
			s.logger.Error("MindletServer", "Failed to switch to model: %v", err)
		}
		http.Error(w, fmt.Sprintf("Failed to switch to model: %v", err), http.StatusInternalServerError)
		return
	}

	// Forward to VLLM if enabled
	if s.config.UseVLLM {
		server := s.vllmManager.GetServer(req.ModelID)
		if server == nil {
			http.Error(w, "VLLM server not running for model", http.StatusServiceUnavailable)
			return
		}

		log.Printf("Forwarding inference to VLLM server on port %d", server.Port)
		if s.logger != nil {
			s.logger.Info("MindletServer", "Forwarding inference to VLLM server on port %d", server.Port)
		}

		// Forward the request to VLLM
		// This is a simplified version - in production, you'd want streaming support
		response, err := server.Forward(r.Context(), req.Messages)
		if err != nil {
			log.Printf("VLLM inference failed: %v", err)
			if s.logger != nil {
				s.logger.Error("MindletServer", "VLLM inference failed: %v", err)
			}
			http.Error(w, fmt.Sprintf("Inference failed: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// Fallback response if VLLM is not enabled
	http.Error(w, "Non-VLLM inference not implemented", http.StatusNotImplemented)
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
