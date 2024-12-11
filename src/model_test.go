package src

import (
	"context"
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestHelper creates and manages a temporary model environment
type ModelTestHelper struct {
	t          *testing.T
	logger     *Logger
	config     *MindletConfig
	tempDir    string
	ramfsDir   string
	volumePath string
	modelPaths map[string]string
}

func NewModelTestHelper(t *testing.T) *ModelTestHelper {
	logger, err := NewLogger()
	require.NoError(t, err)

	// Create temporary directories
	tempDir, err := os.MkdirTemp("", "model-test-*")
	require.NoError(t, err)

	ramfsDir := filepath.Join(tempDir, "ramfs")
	volumePath := filepath.Join(tempDir, "volume")
	require.NoError(t, os.MkdirAll(ramfsDir, 0755))
	require.NoError(t, os.MkdirAll(volumePath, 0755))

	// Create config with test paths
	cfg := DefaultMindletConfig()
	cfg.RamFsRoot = ramfsDir
	cfg.ModelTTL = 100 * time.Millisecond // Short TTL for testing

	return &ModelTestHelper{
		t:          t,
		logger:     logger,
		config:     cfg,
		tempDir:    tempDir,
		ramfsDir:   ramfsDir,
		volumePath: volumePath,
		modelPaths: make(map[string]string),
	}
}

func (h *ModelTestHelper) Cleanup() {
	os.RemoveAll(h.tempDir)
}

// CreateModelFiles creates a mock model directory with checkpoint and non-checkpoint files
func (h *ModelTestHelper) CreateModelFiles(modelID string, modelType ModelType) error {
	// Create both source and destination directories
	srcDir := filepath.Join(h.volumePath, modelID)
	dstDir := filepath.Join(h.ramfsDir, modelID)
	if err := os.MkdirAll(srcDir, 0755); err != nil {
		return fmt.Errorf("failed to create source directory: %v", err)
	}
	if err := os.MkdirAll(dstDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %v", err)
	}

	nonCheckpointFiles := []string{
		"config.json",
		"tokenizer.model",
		"special_tokens.txt",
	}

	for _, filename := range nonCheckpointFiles {
		path := filepath.Join(srcDir, filename)
		if err := os.WriteFile(path, []byte("mock content"), 0644); err != nil {
			return fmt.Errorf("failed to create file %s: %v", filename, err)
		}
	}

	maxCheckpoints := getMaxCheckpoint(modelType)
	for i := 0; i < maxCheckpoints; i++ {
		filename := fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, maxCheckpoints)
		path := filepath.Join(srcDir, filename)
		content := fmt.Sprintf("mock checkpoint %d content", i)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return fmt.Errorf("failed to create checkpoint file %s: %v", filename, err)
		}
	}

	h.modelPaths[modelID] = srcDir
	return nil
}

// VerifyModelFiles checks if all files were correctly copied to the destination
func (h *ModelTestHelper) VerifyModelFiles(modelDir, dstDir string, checkAll bool) {
	// Check non-checkpoint files
	nonCheckpointFiles := []string{
		"config.json",
		"tokenizer.model",
		"special_tokens.txt",
	}

	for _, filename := range nonCheckpointFiles {
		dstPath := filepath.Join(dstDir, filename)
		if _, err := os.Stat(dstPath); os.IsNotExist(err) {
			h.t.Errorf("Non-checkpoint file %s was not copied", filename)
		}
	}

	// Read source directory to get list of checkpoint files
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		h.t.Fatalf("Failed to read model directory: %v", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() && isCheckpointFile(entry.Name()) {
			dstPath := filepath.Join(dstDir, entry.Name())
			if _, err := os.Stat(dstPath); os.IsNotExist(err) {
				// For volume attachment, only check first checkpoint
				if !checkAll && !strings.HasPrefix(entry.Name(), "model-00001-") {
					continue
				}
				h.t.Errorf("Checkpoint file %s was not copied", entry.Name())
			}
		}
	}
}

func (h *ModelTestHelper) VerifyCheckpointFile(dstDir string, checkpoint int, modelType ModelType) {
	maxCheckpoint := getMaxCheckpoint(modelType)
	filename := fmt.Sprintf("model-%05d-of-%05d.safetensors", checkpoint+1, maxCheckpoint)
	dstPath := filepath.Join(dstDir, filename)

	if _, err := os.Stat(dstPath); os.IsNotExist(err) {
		h.t.Errorf("Checkpoint file %s was not copied", filename)
	}
}

func TestCheckpointLoading(t *testing.T) {
	helper := NewModelTestHelper(t)
	defer helper.Cleanup()

	manager, err := NewModelManager(helper.config, helper.logger)
	require.NoError(t, err)

	modelID := "test-model"
	modelDir := filepath.Join(helper.volumePath, modelID)
	require.NoError(t, helper.CreateModelFiles(modelID, Model8B))

	// Create a model instance directly (bypassing volume attachment)
	model := &ModelInstance{
		ID:                modelID,
		Type:              Model8B,
		SrcDir:            modelDir,
		DstDir:            filepath.Join(helper.ramfsDir, modelID),
		LoadedCheckpoints: make(map[int]*CheckpointState),
		CreationTime:      time.Now(),
		LastAccessedTime:  time.Now(),
	}
	manager.models[model.ID] = model

	// Load checkpoints sequentially
	for i := 0; i < 3; i++ {
		err := manager.LoadCheckpoint(model.ID, i, true)
		require.NoError(t, err)

		// Verify state
		model.mu.RLock()
		state := model.LoadedCheckpoints[i]
		require.Equal(t, StateLoaded, state.LoadState)
		model.mu.RUnlock()

		// Verify file
		helper.VerifyCheckpointFile(model.DstDir, i, model.Type)
	}

	// Test reloading an already loaded checkpoint
	start := time.Now()
	err = manager.LoadCheckpoint(model.ID, 0, true)
	duration := time.Since(start)

	require.NoError(t, err)
	require.Less(t, duration, 100*time.Millisecond)
}

func TestCachePreloading(t *testing.T) {
	helper := NewModelTestHelper(t)
	defer helper.Cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	manager, err := NewModelManager(helper.config, helper.logger)
	require.NoError(t, err)
	require.NoError(t, manager.Start(ctx))

	modelID := "test-model"
	modelDir := filepath.Join(helper.volumePath, modelID)
	require.NoError(t, helper.CreateModelFiles(modelID, Model8B))

	model := &ModelInstance{
		ID:                modelID,
		Type:              Model8B,
		SrcDir:            modelDir,
		DstDir:            filepath.Join(helper.ramfsDir, modelID),
		LoadedCheckpoints: make(map[int]*CheckpointState),
		CreationTime:      time.Now(),
		LastAccessedTime:  time.Now(),
	}
	manager.models[model.ID] = model

	// Load checkpoint 0 and verify cache loads 1-4
	err = manager.LoadCheckpoint(model.ID, 0, false)
	require.NoError(t, err)

	waitForCacheLoad := func(start, end int) {
		maxWait := time.After(2 * time.Second)
		ticker := time.NewTicker(50 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-maxWait:
				t.Fatalf("Timeout waiting for cache to load checkpoints %d-%d", start, end)
			case <-ticker.C:
				model.mu.RLock()
				loaded := true
				for i := start; i <= end; i++ {
					state, exists := model.LoadedCheckpoints[i]
					if !exists || state.LoadState != StateLoaded {
						loaded = false
						break
					}
				}
				model.mu.RUnlock()
				if loaded {
					return
				}
			}
		}
	}

	// Verify initial cache load (1-4)
	waitForCacheLoad(1, 4)

	// Request checkpoint 2 and verify cache loads 3-6
	err = manager.LoadCheckpoint(model.ID, 1, false)
	require.NoError(t, err)
	waitForCacheLoad(2, 5)

	// Request checkpoint 3 and verify cache loads 4-7
	err = manager.LoadCheckpoint(model.ID, 2, false)
	require.NoError(t, err)
	waitForCacheLoad(3, 6)
}

func TestModelCleanup(t *testing.T) {
	helper := NewModelTestHelper(t)
	defer helper.Cleanup()
	helper.config.ModelCleanupTick = 100 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	manager, err := NewModelManager(helper.config, helper.logger)
	require.NoError(t, err)
	require.NoError(t, manager.Start(ctx))

	modelID := "test-model"
	modelDir := filepath.Join(helper.volumePath, modelID)
	require.NoError(t, helper.CreateModelFiles(modelID, Model8B))

	model := &ModelInstance{
		ID:                modelID,
		Type:              Model8B,
		SrcDir:            modelDir,
		DstDir:            filepath.Join(helper.ramfsDir, modelID),
		LoadedCheckpoints: make(map[int]*CheckpointState),
		CreationTime:      time.Now(),
		LastAccessedTime:  time.Now(),
	}
	manager.models[model.ID] = model

	// Load several checkpoints
	for i := 0; i < 7; i++ {
		err := manager.LoadCheckpoint(model.ID, i, false)
		require.NoError(t, err)
	}

	// Set model as expired
	model.LastAccessedTime = time.Now().Add(-2 * helper.config.ModelTTL)

	// Wait for cleanup routine to scale down the model
	maxWait := time.After(2 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-maxWait:
			t.Fatal("Timeout waiting for model cleanup")
		case <-ticker.C:
			t.Logf("Checking model %s", model.ID)
			allCorrect := true

			// Check filesystem state
			for i := 0; i < 8; i++ {
				checkpointFile := fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, 8)
				path := filepath.Join(helper.ramfsDir, modelID, checkpointFile)
				exists := true
				if _, err := os.Stat(path); os.IsNotExist(err) {
					exists = false
				}

				if i <= helper.config.NumCheckpointsAhead && !exists ||
					i > helper.config.NumCheckpointsAhead && exists {
					allCorrect = false
					break
				}
			}

			if !allCorrect {
				continue
			}

			t.Logf("Model %s cleaned up correctly", model.ID)

			// Verify model manager state
			model, err := manager.GetModel(modelID)
			if err != nil {
				t.Fatal("Model was completely deleted")
			}

			model.mu.RLock()
			for i := 0; i < 8; i++ {
				_, hasCheckpoint := model.LoadedCheckpoints[i]
				if (i <= helper.config.NumCheckpointsAhead && !hasCheckpoint) ||
					(i > helper.config.NumCheckpointsAhead && hasCheckpoint) {
					allCorrect = false
					break
				}
			}
			model.mu.RUnlock()

			if allCorrect {
				t.Logf("Model %s state is correct", model.ID)
				return
			}
		}
	}
}
