package src

import (
	"bufio"
	"context"
	"fmt"
	"github.com/nlpfollower/deltamind/orchestration/utils"
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
	models *utils.ConcurrentMap[string, *ModelInstance]
	// volume path -> model ID -> model instance
	volumeModels *utils.ConcurrentMap[string, map[string]*ModelInstance]
	// model ID -> last requested checkpoint
	lastRequested *utils.ConcurrentMap[string, int]
	// model ID -> mutex
	modelMutexes *utils.ConcurrentMap[string, *sync.RWMutex]
	done         chan struct{}
}

func NewModelManager(cfg *MindletConfig, logger *Logger) (*ModelManager, error) {
	return &ModelManager{
		config:        cfg,
		logger:        logger,
		models:        utils.NewConcurrentMap[string, *ModelInstance](),
		volumeModels:  utils.NewConcurrentMap[string, map[string]*ModelInstance](),
		lastRequested: utils.NewConcurrentMap[string, int](),
		modelMutexes:  utils.NewConcurrentMap[string, *sync.RWMutex](),
		done:          make(chan struct{}),
	}, nil
}

func (m *ModelManager) Start(ctx context.Context) error {
	go m.runModelCache(ctx)
	go m.runModelCleanup(ctx)
	return nil
}

func (m *ModelManager) getModelMutex(modelID string) *sync.RWMutex {
	mutex, exists := m.modelMutexes.Get(modelID)
	if !exists {
		mutex = &sync.RWMutex{}
		m.modelMutexes.Set(modelID, mutex)
	}
	return mutex
}

func (m *ModelManager) runModelCache(ctx context.Context) {
	ticker := time.NewTicker(m.config.ModelCacheTick)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			lastRequested := m.lastRequested.ToMap()
			for modelID, lastCheckpoint := range lastRequested {
				if model, exists := m.models.Get(modelID); exists {
					maxCheckpoint := getMaxCheckpoint(model.Type)
					endCheckpoint := min(lastCheckpoint+m.config.NumCheckpointsAhead+1, maxCheckpoint)

					for checkpoint := lastCheckpoint + 1; checkpoint < endCheckpoint; checkpoint++ {
						if err := m.LoadCheckpoint(modelID, checkpoint, true); err != nil {
							m.logger.Error("ModelManager", "Failed to preload checkpoint %d: %v", checkpoint, err)
							continue
						}
					}
				} else {
					m.lastRequested.Remove(modelID)
				}
			}
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
			var expiredModels []ModelSpec
			var volumePaths []string

			now := time.Now()
			for _, model := range m.models.GetAll() {
				mutex := m.getModelMutex(model.ID)
				mutex.RLock()
				if now.Sub(model.LastAccessedTime) > m.config.ModelTTL {
					expiredModels = append(expiredModels, ModelSpec{
						ID:   model.ID,
						Type: model.Type,
					})
					volumePaths = append(volumePaths, model.VolumePath)
				}
				mutex.RUnlock()
			}

			for _, spec := range expiredModels {
				if err := m.ScaleDownModel(spec.ID); err != nil {
					m.logger.Error("ModelManager", "Failed to scale down model %s: %v", spec.ID, err)
				}
			}
		}
	}
}

func (m *ModelManager) AttachVolume(volumePath string, modelSpecs map[string]ModelSpec) error {
	if _, exists := m.volumeModels.Get(volumePath); exists {
		return fmt.Errorf("volume already attached: %s", volumePath)
	}
	m.volumeModels.Set(volumePath, make(map[string]*ModelInstance))

	for _, spec := range modelSpecs {
		if err := m.LoadModel(spec, volumePath); err != nil {
			return fmt.Errorf("failed to load model %s: %v", spec.ID, err)
		}
	}

	return nil
}

func (m *ModelManager) DetachVolume(volumePath string) error {
	models, exists := m.volumeModels.Get(volumePath)
	if !exists {
		return fmt.Errorf("volume not found: %s", volumePath)
	}

	var errors []string

	for modelName, model := range models {
		if err := m.DeleteModel(model.ID); err != nil {
			errors = append(errors, fmt.Sprintf("model %s: %v", modelName, err))
		}
	}

	m.volumeModels.Remove(volumePath)

	if len(errors) > 0 {
		return fmt.Errorf("errors deleting models: %s", strings.Join(errors, "; "))
	}

	return nil
}

func (m *ModelManager) LoadModel(spec ModelSpec, volumePath string) error {
	mutex := m.getModelMutex(spec.ID)
	mutex.Lock()
	defer mutex.Unlock()

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

	// Add to maps
	m.models.Set(spec.ID, model)
	if volumePath != "" {
		volumeModels, _ := m.volumeModels.Get(volumePath)
		if volumeModels == nil {
			volumeModels = make(map[string]*ModelInstance)
		}
		volumeModels[spec.ID] = model
		m.volumeModels.Set(volumePath, volumeModels)
	}

	// Copy non-checkpoint files
	if err := m.copyNonCheckpointFiles(model); err != nil {
		m.logger.Error("ModelManager", "Failed to copy non-checkpoint files: %v", err)
		m.deleteModelNoLock(model.ID)
		return err
	}

	// Load initial checkpoints
	if err := m.loadCheckpointNoLock(model.ID, 0, false); err != nil {
		m.logger.Error("ModelManager", "Failed to load initial checkpoint: %v", err)
		m.deleteModelNoLock(model.ID)
		return err
	}

	return nil
}

func (m *ModelManager) GetModel(modelID string) (*ModelInstance, error) {
	model, exists := m.models.Get(modelID)
	if !exists {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}
	return model, nil
}

func (m *ModelManager) GetDefaultModelID() string {
	for id := range m.models.ToMap() {
		return id
	}
	return ""
}

func (m *ModelManager) ScaleDownModel(modelID string) error {
	mutex := m.getModelMutex(modelID)
	mutex.Lock()
	defer mutex.Unlock()

	model, exists := m.models.Get(modelID)
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	maxCheckpoint := getMaxCheckpoint(model.Type)

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
	mutex := m.getModelMutex(modelID)
	mutex.Lock()
	defer mutex.Unlock()

	return m.deleteModelNoLock(modelID)
}

func (m *ModelManager) deleteModelNoLock(modelID string) error {
	model, exists := m.models.Get(modelID)
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}
	m.models.Remove(modelID)
	m.lastRequested.Remove(modelID)

	if err := os.RemoveAll(model.DstDir); err != nil {
		return fmt.Errorf("failed to delete model directory: %v", err)
	}

	return nil
}

func (m *ModelManager) LoadCheckpoint(modelID string, checkpoint int, fromCache bool) error {
	mutex := m.getModelMutex(modelID)
	mutex.Lock()
	defer mutex.Unlock()

	return m.loadCheckpointNoLock(modelID, checkpoint, fromCache)
}

func (m *ModelManager) loadCheckpointNoLock(modelID string, checkpoint int, fromCache bool) error {
	model, err := m.GetModel(modelID)
	if err != nil {
		return err
	}

	maxCheckpoint := getMaxCheckpoint(model.Type)
	if checkpoint < 0 || checkpoint >= maxCheckpoint {
		return fmt.Errorf("invalid checkpoint number: %d", checkpoint)
	}

	if !fromCache {
		lastRequested, _ := m.lastRequested.Get(modelID)
		m.lastRequested.Set(modelID, max(lastRequested, checkpoint))
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
