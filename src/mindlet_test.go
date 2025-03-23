package src

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/stretchr/testify/require"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestHelper encapsulates common test functionality
type TestHelper struct {
	*ModelTestHelper
	mindlet  *Mindlet
	listener net.Listener
	ctx      context.Context
	cancel   context.CancelFunc
}

func NewTestHelper(t *testing.T, useMock bool) *TestHelper {
	modelHelper := NewModelTestHelper(t, useMock)

	// Create real TCP listener on random port
	listener, err := net.Listen("tcp", ":0")
	require.NoError(t, err)

	// Update config with listener port
	modelHelper.config.Port = listener.Addr().(*net.TCPAddr).Port
	modelHelper.config.StreamLogs = false

	model, err := NewModelManager(modelHelper.config, modelHelper.logger)
	require.NoError(t, err)

	connState := NewConnectionState()
	var llmServer LLMServer
	if useMock {
		llmServer = NewMockLLMServer(modelHelper.config, modelHelper.logger, connState)
	} else {
		llmServer = NewDefaultLLMServer(modelHelper.config, modelHelper.logger)
	}
	inferenceServer := NewInferenceServer(modelHelper.config, modelHelper.logger, model, connState, llmServer)

	mindlet := &Mindlet{
		config:          modelHelper.config,
		logger:          modelHelper.logger,
		model:           model,
		listener:        listener,
		connState:       connState,
		inferenceServer: inferenceServer,
		done:            make(chan struct{}),
	}

	ctx, cancel := context.WithCancel(context.Background())

	helper := &TestHelper{
		ModelTestHelper: modelHelper,
		mindlet:         mindlet,
		listener:        listener,
		ctx:             ctx,
		cancel:          cancel,
	}

	go func() {
		if err := mindlet.Start(ctx); err != nil {
			t.Errorf("Mindlet failed to start: %v", err)
		}
	}()

	return helper
}

func (h *TestHelper) CreateConnection(connType ConnectionType) *FramedConn {
	addr := h.listener.Addr().String()
	conn, err := net.Dial("tcp", addr)
	require.NoError(h.t, err)

	framedConn := NewFramedConn(conn)

	initData := InitialConnectionMessage{
		ConnectionType: connType,
		ClientID:       "test-client",
	}

	initMsg, err := NewRequestMessage(GenerateUUID(), ActionTypeInitialConnection, initData)
	require.NoError(h.t, err)

	err = framedConn.WriteMessage(initMsg)
	require.NoError(h.t, err)

	return framedConn
}

func (h *TestHelper) Cleanup() {
	h.cancel()
	h.listener.Close()
}

func (h *TestHelper) AttachVolume(engineConn *FramedConn, models map[string]ModelSpec) error {
	for modelID, spec := range models {
		h.CreateModelFiles(modelID, spec.Type)
	}

	attachReq := &VolumeAttachRequest{
		Path:   h.volumePath,
		Models: models,
	}
	msg, _ := NewRequestMessage(GenerateUUID(), ActionTypeVolumeAttach, attachReq)

	if err := engineConn.WriteMessage(msg); err != nil {
		return fmt.Errorf("failed to send attach request: %v", err)
	}

	attachResp, err := engineConn.ReadMessage()
	if err != nil {
		return fmt.Errorf("failed to read attach response: %v", err)
	}

	var attachData VolumeAttachResponse
	if err := json.Unmarshal(attachResp.Data, &attachData); err != nil {
		return fmt.Errorf("failed to unmarshal attach response: %v", err)
	}

	if attachData.Status != ResponseStatusSuccess {
		return fmt.Errorf("volume attach failed: %s", attachData.Message)
	}

	return nil
}

func (h *TestHelper) waitForConnection(connType ConnectionType, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if h.mindlet.connState.CheckConnection(connType) {
			return nil
		}
		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("connection %s not established within timeout", connType)
}

func validateMockResponsePattern(t *testing.T, responses []InferenceStreamResponse) {
	require.Len(t, responses, 3, "Expected 3 responses (STREAM, STREAM, FINAL) per request")
	require.Equal(t, ResponseTypeStream, responses[0].Type)
	require.Equal(t, "Hello", responses[0].Text)
	require.Equal(t, ResponseTypeStream, responses[1].Type)
	require.Equal(t, " ", responses[1].Text)
	require.Equal(t, ResponseTypeFinal, responses[2].Type)
	require.Equal(t, "world!", responses[2].Text)
}

func validateReconnectResponses(t *testing.T, engineConn *FramedConn) {
	responses := make([]InferenceStreamResponse, 0, 3)
	for i := 0; i < 3; i++ {
		streamResp, err := engineConn.ReadMessage()
		require.NoError(t, err)
		var sr InferenceStreamResponse
		require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
		responses = append(responses, sr)
	}

	require.Len(t, responses, 3, "Expected 3 responses")
	require.Equal(t, ResponseTypeStream, responses[0].Type)
	require.Equal(t, "Hello", responses[0].Text)
	require.Equal(t, ResponseTypeStream, responses[1].Type)
	require.Equal(t, " ", responses[1].Text)
	require.Equal(t, ResponseTypeFinal, responses[2].Type)
	require.Equal(t, "world!", responses[2].Text)
}

func TestMindletVolumeAttachInferenceAndDetach(t *testing.T) {
	helper := NewTestHelper(t, false)
	defer helper.Cleanup()

	// Attach model
	engineConn := helper.CreateConnection(EngineConnection)
	t.Logf("Attaching volume")
	require.NoError(t, helper.AttachVolume(engineConn, map[string]ModelSpec{
		"my-model": {
			ID:   "my-model",
			Type: Model8B,
		},
	}))

	t.Logf("Validating model files")
	helper.VerifyModelFiles(
		helper.modelPaths["my-model"],
		filepath.Join(helper.ramfsDir, "my-model"),
		false,
	)

	// Send LoadModel request
	loadModelReq := &LoadModelRequest{
		ModelName: "my-model",
	}
	loadModelMsg, _ := NewRequestMessage(GenerateUUID(), ActionTypeLoadModel, loadModelReq)
	require.NoError(t, engineConn.WriteMessage(loadModelMsg))

	// Read LoadModel response
	loadModelResp, err := engineConn.ReadMessage()
	require.NoError(t, err)
	require.Equal(t, MessageTypeResponse, loadModelResp.Type)
	require.Equal(t, ActionTypeLoadModel, loadModelResp.Action)

	var loadResp LoadModelResponse
	require.NoError(t, json.Unmarshal(loadModelResp.Data, &loadResp))
	require.Equal(t, ResponseStatusSuccess, loadModelResp.Status)
	require.Contains(t, loadResp.Message, "my-model loaded successfully")

	// Send multiple inference requests
	numRequests := 20
	infRequests := make([]*InferenceStreamRequest, numRequests)
	infMsgs := make([]*UnifiedMessage, numRequests)
	responsesByID := make(map[string][]InferenceStreamResponse)

	for i := 0; i < numRequests; i++ {
		infRequests[i] = &InferenceStreamRequest{
			ModelID: "my-model",
			Input:   fmt.Sprintf("Request %d", i+1),
		}
		infMsgs[i], _ = NewRequestMessage(GenerateUUID(), ActionTypeInferenceStream, infRequests[i])
		responsesByID[infMsgs[i].ID] = []InferenceStreamResponse{}
	}

	// Send all requests
	for _, msg := range infMsgs {
		require.NoError(t, engineConn.WriteMessage(msg))
	}

	// Read all responses
	responsesReceived := 0
	totalExpectedResponses := numRequests * 3 // Each request should get 3 responses (2 STREAM, 1 FINAL)

	for responsesReceived < totalExpectedResponses {
		streamResp, err := engineConn.ReadMessage()
		require.NoError(t, err)
		var sr InferenceStreamResponse
		require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
		require.Equal(t, MessageTypeResponse, streamResp.Type)
		require.Equal(t, ActionTypeInferenceStream, streamResp.Action)
		responsesByID[streamResp.ID] = append(responsesByID[streamResp.ID], sr)
		responsesReceived++
	}

	// Validate responses
	for _, responses := range responsesByID {
		validateMockResponsePattern(t, responses)
	}

	// Detach volume
	detachReq := &VolumeDetachRequest{
		Path: helper.volumePath,
	}
	detachMsg, _ := NewRequestMessage(GenerateUUID(), ActionTypeVolumeDetach, detachReq)
	require.NoError(t, engineConn.WriteMessage(detachMsg))

	// Read detach response
	detachResp, err := engineConn.ReadMessage()
	require.NoError(t, err)
	require.Equal(t, MessageTypeResponse, detachResp.Type)
	require.Equal(t, ActionTypeVolumeDetach, detachResp.Action)

	var detachData VolumeDetachResponse
	require.NoError(t, json.Unmarshal(detachResp.Data, &detachData))
	require.Equal(t, ResponseStatusSuccess, detachData.Status)
	require.Contains(t, detachData.Message, "Volume detached")

	// Verify model files are removed from ramfs
	_, err = os.Stat(filepath.Join(helper.ramfsDir, "my-model"))
	require.True(t, os.IsNotExist(err), "Model files should be removed after detach")
}

func TestMindletReconnectEngine(t *testing.T) {
	helper := NewTestHelper(t, true)
	defer helper.Cleanup()

	helper.mindlet.logger.Info("TestMindletReconnectEngine", "Starting test with volume path: %s", helper.volumePath)

	// Create model files before attaching
	require.NoError(t, helper.CreateModelFiles("reconnect-model", Model8B))

	// 1. Connect engine and send VolumeAttachRequest
	engineConn := helper.CreateConnection(EngineConnection)

	attachReq := &VolumeAttachRequest{
		Path: helper.volumePath,
		Models: map[string]ModelSpec{
			"reconnect-model": {
				ID:   "reconnect-model",
				Type: Model8B,
			},
		},
	}
	wrappedAttach, _ := NewRequestMessage("attach-req-1", ActionTypeVolumeAttach, attachReq)
	require.NoError(t, engineConn.WriteMessage(wrappedAttach))

	attachResp, err := engineConn.ReadMessage()
	require.NoError(t, err)
	var attachData VolumeAttachResponse
	require.NoError(t, json.Unmarshal(attachResp.Data, &attachData))
	require.Equal(t, ResponseStatusSuccess, attachData.Status)

	helper.mindlet.logger.Info("TestMindletReconnectEngine", "Volume attached, model files created at: %s", filepath.Join(helper.volumePath, "reconnect-model"))

	// Wait for the model to be loaded
	require.Eventually(t, func() bool {
		model, err := helper.mindlet.model.GetModel("reconnect-model")
		return err == nil && model != nil
	}, 5*time.Second, 100*time.Millisecond, "Model was not loaded within the expected time")

	// 2. Simulate engine disconnect by closing engineConn
	helper.mindlet.logger.Info("TestMindletReconnectEngine", "Simulating engine disconnect")
	engineConn.conn.Close()

	// Wait to ensure the connection is fully closed
	time.Sleep(500 * time.Millisecond)

	// 3. Reconnect engine
	helper.mindlet.logger.Info("TestMindletReconnectEngine", "Reconnecting engine")
	engineConn2 := helper.CreateConnection(EngineConnection)

	// Wait for the new connection to be established
	require.NoError(t, helper.waitForConnection(EngineConnection, 5*time.Second))

	// Verify model state
	model, err := helper.mindlet.model.GetModel("reconnect-model")
	require.NoError(t, err)
	require.NotNil(t, model)
	require.Equal(t, Model8B, model.Type)
	require.NotEmpty(t, model.DstDir)
	require.FileExists(t, filepath.Join(model.DstDir, "config.json"))

	// 4. Send another inference request and verify it still works
	infReq := &InferenceStreamRequest{
		ModelID: "reconnect-model",
		Input:   "Hello again!",
	}
	wrappedInf, _ := NewRequestMessage("inf-req-2", ActionTypeInferenceStream, infReq)
	require.NoError(t, engineConn2.WriteMessage(wrappedInf))

	// Validate responses
	validateReconnectResponses(t, engineConn2)
}
