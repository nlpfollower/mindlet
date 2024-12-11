package src

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ModelInstance represents a loaded model in memory
type ModelInstance struct {
	ID                string
	Type              ModelType
	SrcDir            string
	DstDir            string
	VolumePath        string
	LoadedCheckpoints map[int]*CheckpointState
	CreationTime      time.Time
	LastAccessedTime  time.Time
	mu                sync.RWMutex
}

type CheckpointState struct {
	Number    int
	Path      string
	LoadState LoadState
}

type LoadState int

const (
	StateUnloaded LoadState = iota
	StateLoading
	StateLoaded
)

type ModelManager struct {
	config *MindletConfig
	logger *Logger
	// model ID -> model instance
	models map[string]*ModelInstance
	// volume path -> model ID -> model instance
	volumeModels map[string]map[string]*ModelInstance
	// model ID -> last requested checkpoint
	lastRequested map[string]int
	done          chan struct{}
	mu            sync.RWMutex
}

func NewModelManager(cfg *MindletConfig, logger *Logger) (*ModelManager, error) {
	return &ModelManager{
		config:        cfg,
		logger:        logger,
		models:        make(map[string]*ModelInstance),
		volumeModels:  make(map[string]map[string]*ModelInstance),
		lastRequested: make(map[string]int),
		done:          make(chan struct{}),
	}, nil
}

func (m *ModelManager) Start(ctx context.Context) error {
	go m.runModelCache(ctx)
	go m.runModelCleanup(ctx)
	return nil
}

func (m *ModelManager) runModelCache(ctx context.Context) {
	ticker := time.NewTicker(m.config.ModelCacheTick)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.mu.RLock()
			for modelID, lastCheckpoint := range m.lastRequested {
				if model, exists := m.models[modelID]; exists {
					maxCheckpoint := getMaxCheckpoint(model.Type)
					endCheckpoint := min(lastCheckpoint+m.config.NumCheckpointsAhead+1, maxCheckpoint)

					for checkpoint := lastCheckpoint + 1; checkpoint < endCheckpoint; checkpoint++ {
						if err := m.LoadCheckpoint(modelID, checkpoint, true); err != nil {
							m.logger.Error("ModelManager", "Failed to preload checkpoint %d: %v", checkpoint, err)
							continue
						}
					}
				} else {
					delete(m.lastRequested, modelID)
				}
			}
			m.mu.RUnlock()
		}
	}
}

func (m *ModelManager) runModelCleanup(ctx context.Context) {
	ticker := time.NewTicker(m.config.ModelCleanupTick)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.cleanupExpiredModels()
		}
	}
}

func (m *ModelManager) cleanupExpiredModels() {
	var expiredModels []ModelSpec
	var volumePaths []string

	m.mu.RLock()
	now := time.Now()
	for _, model := range m.models {
		model.mu.RLock()
		if now.Sub(model.LastAccessedTime) > m.config.ModelTTL {
			expiredModels = append(expiredModels, ModelSpec{
				ID:   model.ID,
				Type: model.Type,
			})
			volumePaths = append(volumePaths, model.VolumePath)
		}
		model.mu.RUnlock()
	}
	m.mu.RUnlock()

	for _, spec := range expiredModels {
		if err := m.ScaleDownModel(spec.ID); err != nil {
			m.logger.Error("ModelManager", "Failed to scale down model %s: %v", spec.ID, err)
		}
	}
}

func (m *ModelManager) AttachVolume(volumePath string, modelSpecs map[string]ModelSpec) error {
	// Initialize volume entry if it doesn't exist
	m.mu.Lock()
	if _, exists := m.volumeModels[volumePath]; exists {
		return fmt.Errorf("volume already attached: %s", volumePath)
	}
	m.volumeModels[volumePath] = make(map[string]*ModelInstance)
	m.mu.Unlock()

	// Load each model specified in the request's model map.
	for _, spec := range modelSpecs {
		if err := m.loadModel(spec, volumePath); err != nil {
			return fmt.Errorf("failed to load model %s: %v", spec.ID, err)
		}
	}

	return nil
}

func (m *ModelManager) DetachVolume(volumePath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	models, exists := m.volumeModels[volumePath]
	if !exists {
		return fmt.Errorf("volume not found: %s", volumePath)
	}

	var errors []string

	// Delete all models associated with this volume
	for modelName, model := range models {
		if err := m.DeleteModel(model.ID); err != nil {
			errors = append(errors, fmt.Sprintf("model %s: %v", modelName, err))
		}
		delete(m.models, model.ID)
	}

	// Remove volume entry
	delete(m.volumeModels, volumePath)

	if len(errors) > 0 {
		return fmt.Errorf("errors deleting models: %s", strings.Join(errors, "; "))
	}

	return nil
}

func (m *ModelManager) GetModel(modelID string) (*ModelInstance, error) {
	m.mu.RLock()
	model, exists := m.models[modelID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}
	return model, nil
}

func (m *ModelManager) GetDefaultModelID() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Return the first model ID found
	for id := range m.models {
		return id
	}
	return ""
}

func (m *ModelManager) ScaleDownModel(modelID string) error {
	m.mu.Lock()
	model, exists := m.models[modelID]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("model not found: %s", modelID)
	}
	m.mu.Unlock()

	maxCheckpoint := getMaxCheckpoint(model.Type)
	model.mu.Lock()
	defer model.mu.Unlock()

	for checkpoint := m.config.NumCheckpointsAhead + 1; checkpoint < maxCheckpoint; checkpoint++ {
		filename := fmt.Sprintf("model-%05d-of-%05d.safetensors", checkpoint+1, maxCheckpoint)
		path := filepath.Join(model.DstDir, filename)
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to delete checkpoint file: %v", err)
		}
		delete(model.LoadedCheckpoints, checkpoint)
	}
	return nil
}

func (m *ModelManager) DeleteModel(modelID string) error {
	m.mu.Lock()
	model, exists := m.models[modelID]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("model not found: %s", modelID)
	}
	delete(m.models, modelID)
	delete(m.lastRequested, modelID)
	m.mu.Unlock()

	if err := os.RemoveAll(model.DstDir); err != nil {
		return fmt.Errorf("failed to delete model directory: %v", err)
	}

	return nil
}

func (m *ModelManager) LoadCheckpoint(modelID string, checkpoint int, fromCache bool) error {
	model, err := m.GetModel(modelID)
	if err != nil {
		return err
	}

	maxCheckpoint := getMaxCheckpoint(model.Type)
	if checkpoint < 0 || checkpoint >= maxCheckpoint {
		return fmt.Errorf("invalid checkpoint number: %d", checkpoint)
	}

	model.mu.Lock()
	defer model.mu.Unlock()
	if !fromCache {
		m.mu.Lock()
		m.lastRequested[modelID] = max(m.lastRequested[modelID], checkpoint)
		m.mu.Unlock()
		// Update on real access
		model.LastAccessedTime = time.Now()
	}

	state, exists := model.LoadedCheckpoints[checkpoint]
	if !exists {
		state = &CheckpointState{
			Number:    checkpoint,
			Path:      filepath.Join(model.SrcDir, fmt.Sprintf("model-%05d-of-%05d.safetensors", checkpoint+1, maxCheckpoint)),
			LoadState: StateUnloaded,
		}
		model.LoadedCheckpoints[checkpoint] = state
	}

	// If already loaded, we're done
	if state.LoadState == StateLoaded {
		return nil
	}

	// If being loaded by another goroutine, return error
	if state.LoadState == StateLoading {
		return fmt.Errorf("checkpoint %d is already being loaded", checkpoint)
	}

	// Mark as loading and copy the file
	state.LoadState = StateLoading

	checkpointFile := fmt.Sprintf("model-%05d-of-%05d.safetensors", checkpoint+1, maxCheckpoint)
	src := filepath.Join(model.SrcDir, checkpointFile)
	dst := filepath.Join(model.DstDir, checkpointFile)

	if err := m.copyFile(src, dst); err != nil {
		// Update state on failure
		state.LoadState = StateUnloaded
		return fmt.Errorf("failed to copy checkpoint file: %v", err)
	}

	// Update state on success
	state.LoadState = StateLoaded

	return nil
}

func (m *ModelManager) loadModel(spec ModelSpec, volumePath string) error {
	modelSrcPath := filepath.Join(volumePath, spec.ID)

	// Validate model path exists
	if _, err := os.Stat(modelSrcPath); err != nil {
		return fmt.Errorf("model %s: %v", spec.ID, err)
	}

	// Create destination directory
	dstDir := filepath.Join(m.config.RamFsRoot, spec.ID)
	if err := os.MkdirAll(dstDir, 0755); err != nil {
		return fmt.Errorf("model %s: %v", spec.ID, err)
	}

	model := &ModelInstance{
		ID:                spec.ID,
		Type:              spec.Type,
		SrcDir:            modelSrcPath,
		DstDir:            dstDir,
		VolumePath:        volumePath,
		LoadedCheckpoints: make(map[int]*CheckpointState),
		CreationTime:      time.Now(),
		LastAccessedTime:  time.Now(),
	}

	// Add to maps under lock
	m.mu.Lock()
	m.models[spec.ID] = model
	if volumePath != "" {
		if _, exists := m.volumeModels[volumePath]; !exists {
			m.volumeModels[volumePath] = make(map[string]*ModelInstance)
		}
		m.volumeModels[volumePath][spec.ID] = model
	}
	m.mu.Unlock()

	// Copy non-checkpoint files
	if err := m.copyNonCheckpointFiles(model); err != nil {
		m.logger.Error("ModelManager", "Failed to copy non-checkpoint files: %v", err)
		m.DeleteModel(model.ID)
		return err
	}

	// Load initial checkpoints
	if err := m.LoadCheckpoint(model.ID, 0, false); err != nil {
		m.logger.Error("ModelManager", "Failed to load initial checkpoint: %v", err)
		m.DeleteModel(model.ID)
		return err
	}

	return nil
}

func (m *ModelManager) copyFile(src, dst string) error {
	cmd := exec.Command("dd", "if="+src, "of="+dst, "bs=1024M", "status=progress")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start dd command: %v", err)
	}

	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			m.logger.Info("Copy progress: %s", scanner.Text())
		}
	}()

	return cmd.Wait()
}

func (m *ModelManager) copyNonCheckpointFiles(model *ModelInstance) error {
	entries, err := os.ReadDir(model.SrcDir)
	if err != nil {
		return fmt.Errorf("failed to read source directory: %v", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() && !isCheckpointFile(entry.Name()) {
			src := filepath.Join(model.SrcDir, entry.Name())
			dst := filepath.Join(model.DstDir, entry.Name())
			if err := m.copyFile(src, dst); err != nil {
				return err
			}
		}
	}
	return nil
}
