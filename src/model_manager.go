// mindlet/src/model_manager.go
package src

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type ModelInfo struct {
	ID            string    `json:"id"`
	SourcePath    string    `json:"source_path"`
	ConvertedPath string    `json:"converted_path,omitempty"`
	ModelSize     string    `json:"model_size"`
	IsDCP         bool      `json:"is_dcp"`
	LoadedAt      time.Time `json:"loaded_at"`
	Status        string    `json:"status"`
}

type ModelManager struct {
	config       *MindletConfig
	loadedModels map[string]*ModelInfo
	mu           sync.RWMutex
	converter    *ModelConverter
	logger       *Logger
}

func NewModelManager(cfg *MindletConfig, logger *Logger) *ModelManager {
	return &ModelManager{
		config:       cfg,
		loadedModels: make(map[string]*ModelInfo),
		converter:    NewModelConverter(cfg, logger),
		logger:       logger,
	}
}

func (m *ModelManager) LoadModel(modelID, sourcePath, modelSize string, isDCP bool) (*ModelInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if already loaded
	if info, exists := m.loadedModels[modelID]; exists {
		return info, nil
	}

	if m.logger != nil {
		m.logger.Info("ModelManager", "Loading model %s from %s (isDCP: %v, size: %s)", modelID, sourcePath, isDCP, modelSize)
	}

	modelInfo := &ModelInfo{
		ID:         modelID,
		SourcePath: sourcePath,
		IsDCP:      isDCP,
		LoadedAt:   time.Now(),
		Status:     "loading",
		ModelSize:  modelSize,
	}

	// If it's a DCP checkpoint, check if it's already converted
	if isDCP {
		outputDir := filepath.Join(m.config.ConvertedModelsDir, modelID)

		// Check if the converted model already exists
		if isModelAlreadyConverted(outputDir) {
			log.Printf("Model %s is already converted at %s, skipping conversion", modelID, outputDir)
			if m.logger != nil {
				m.logger.Info("ModelManager", "Model %s is already converted at %s, skipping conversion", modelID, outputDir)
			}
			modelInfo.ConvertedPath = outputDir
		} else {
			// Need to convert
			log.Printf("Converting DCP checkpoint for model %s", modelID)
			if m.logger != nil {
				m.logger.Info("ModelManager", "Converting DCP checkpoint for model %s", modelID)
			}

			if err := os.MkdirAll(outputDir, 0755); err != nil {
				return nil, fmt.Errorf("failed to create output directory: %w", err)
			}

			err := m.converter.ConvertDCPToSafetensors(
				sourcePath,
				outputDir,
				modelInfo.ModelSize,
				"bfloat16",
				m.config.DefaultTokenizerPath,
			)
			if err != nil {
				if m.logger != nil {
					m.logger.Error("ModelManager", "Failed to convert DCP checkpoint: %v", err)
				}
				return nil, fmt.Errorf("failed to convert DCP checkpoint: %w", err)
			}

			modelInfo.ConvertedPath = outputDir
			log.Printf("Successfully converted model %s to %s", modelID, outputDir)
			if m.logger != nil {
				m.logger.Info("ModelManager", "Successfully converted model %s to %s", modelID, outputDir)
			}
		}
	} else {
		// Already in safetensors format
		modelInfo.ConvertedPath = sourcePath
	}

	modelInfo.Status = "loaded"
	m.loadedModels[modelID] = modelInfo

	return modelInfo, nil
}

func (m *ModelManager) IsModelLoaded(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	_, exists := m.loadedModels[modelID]
	return exists
}

func (m *ModelManager) GetLoadedModels() []ModelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	models := make([]ModelInfo, 0, len(m.loadedModels))
	for _, info := range m.loadedModels {
		models = append(models, *info)
	}
	return models
}

func (m *ModelManager) GetModelInfo(modelID string) (*ModelInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	info, exists := m.loadedModels[modelID]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelID)
	}

	return info, nil
}

// isModelAlreadyConverted checks if a model has already been converted to safetensors format
func isModelAlreadyConverted(outputDir string) bool {
	// Check if the directory exists
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		return false
	}

	// Check for required files
	requiredFiles := []string{
		"config.json",
		"model.safetensors.index.json",
	}

	for _, file := range requiredFiles {
		if _, err := os.Stat(filepath.Join(outputDir, file)); os.IsNotExist(err) {
			return false
		}
	}

	// Check if there are any safetensors files
	safetensorsFiles, err := filepath.Glob(filepath.Join(outputDir, "*.safetensors"))
	if err != nil || len(safetensorsFiles) == 0 {
		return false
	}

	return true
}

// ModelConverter handles DCP to safetensors conversion
type ModelConverter struct {
	config *MindletConfig
	logger *Logger
}

func NewModelConverter(cfg *MindletConfig, logger *Logger) *ModelConverter {
	return &ModelConverter{
		config: cfg,
		logger: logger,
	}
}

func (c *ModelConverter) ConvertDCPToSafetensors(dcpDir, outputDir, modelSize, dtype, tokenizerPath string) error {
	// Check if conversion script exists
	scriptPath := c.config.ConversionScriptPath
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("conversion script not found at %s", scriptPath)
	}

	// Build command
	args := []string{
		scriptPath,
		"--dcp-dir", dcpDir,
		"--output-dir", outputDir,
		"--model-size", modelSize,
		"--dtype", dtype,
	}

	if tokenizerPath != "" {
		args = append(args, "--tokenizer-path", tokenizerPath)
	}

	cmd := exec.Command(c.config.PythonPath, args...)

	// Capture output for logging
	output, err := cmd.CombinedOutput()

	if c.logger != nil {
		c.logger.Info("ModelConverter", "Running conversion: %s %s", c.config.PythonPath, strings.Join(args, " "))
		c.logger.Info("ModelConverter", "Conversion output:\n%s", string(output))
	} else {
		log.Printf("Running conversion: %s %s", c.config.PythonPath, strings.Join(args, " "))
		if len(output) > 0 {
			log.Printf("Conversion output:\n%s", string(output))
		}
	}

	if err != nil {
		if c.logger != nil {
			c.logger.Error("ModelConverter", "Conversion failed: %v", err)
		}
		return fmt.Errorf("conversion failed: %w", err)
	}

	// Verify output exists
	outputFiles, err := filepath.Glob(filepath.Join(outputDir, "*.safetensors"))
	if err != nil || len(outputFiles) == 0 {
		return fmt.Errorf("conversion completed but no safetensors files found")
	}

	log.Printf("Conversion successful, found %d safetensors files", len(outputFiles))
	if c.logger != nil {
		c.logger.Info("ModelConverter", "Conversion successful, found %d safetensors files", len(outputFiles))
	}
	return nil
}
