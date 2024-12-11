package src

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/stretchr/testify/require"
	"net"
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

func NewTestHelper(t *testing.T) *TestHelper {
	modelHelper := NewModelTestHelper(t)

	// Create real TCP listener on random port
	listener, err := net.Listen("tcp", ":0")
	require.NoError(t, err)

	// Update config with listener port
	modelHelper.config.Port = listener.Addr().(*net.TCPAddr).Port
	modelHelper.config.StreamLogs = false

	model, err := NewModelManager(modelHelper.config, modelHelper.logger)
	require.NoError(t, err)

	connState := NewConnectionState()
	llmServer := NewMockLLMServer(modelHelper.config, modelHelper.logger, connState)
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

func (h *TestHelper) CreateConnection(connType ConnectionType) net.Conn {
	// Get the actual listener address
	addr := h.listener.Addr().String()

	// Create real TCP connection
	conn, err := net.Dial("tcp", addr)
	require.NoError(h.t, err)

	// Send initial connection message
	initMsg := InitialConnectionMessage{
		ConnectionType: connType,
		ClientID:       "test-client",
	}

	if err := json.NewEncoder(conn).Encode(initMsg); err != nil {
		conn.Close()
		h.t.Fatalf("Failed to encode init message: %v", err)
	}

	return conn
}

func (h *TestHelper) Cleanup() {
	h.cancel()
	h.listener.Close()
}

func (h *TestHelper) AttachVolume(engineConn net.Conn, models map[string]ModelSpec) error {
	// Create all models first
	for modelID, spec := range models {
		h.CreateModelFiles(modelID, spec.Type)
	}

	attachReq := &VolumeAttachRequest{
		Path:   h.volumePath,
		Models: models,
	}
	wrappedAttach, _ := NewWrappedRequest("attach-req-1", attachReq)

	if err := json.NewEncoder(engineConn).Encode(wrappedAttach); err != nil {
		return fmt.Errorf("failed to send attach request: %v", err)
	}

	var attachResp WrappedResponse
	if err := json.NewDecoder(engineConn).Decode(&attachResp); err != nil {
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

func validateMockResponsePattern(t *testing.T, responses []InferenceStreamResponse) {
	require.Len(t, responses, 3, "Expected 3 responses (STREAM, STREAM, FINAL) per request")
	require.Equal(t, ResponseTypeStream, responses[0].Type)
	require.Equal(t, "Hello", responses[0].Text)
	require.Equal(t, ResponseTypeStream, responses[1].Type)
	require.Equal(t, " ", responses[1].Text)
	require.Equal(t, ResponseTypeFinal, responses[2].Type)
	require.Equal(t, "world!", responses[2].Text)
}

func TestMindletVolumeAttachAndInference(t *testing.T) {
	helper := NewTestHelper(t)
	defer helper.Cleanup()

	// Attach model
	engineConn := helper.CreateConnection(EngineConnection)
	require.NoError(t, helper.AttachVolume(engineConn, map[string]ModelSpec{
		"test-model": {
			ID:   "test-model",
			Type: Model8B,
		},
	}))

	helper.VerifyModelFiles(
		helper.modelPaths["test-model"],
		filepath.Join(helper.ramfsDir, "test-model"),
		false,
	)

	encoder := json.NewEncoder(engineConn)
	decoder := json.NewDecoder(engineConn)

	// First request
	infReq1 := &InferenceStreamRequest{
		ModelID: "test-model",
		Input:   "Hello world!",
	}
	wrappedInf1, _ := NewWrappedRequest("inf-req-1", infReq1)
	require.NoError(t, encoder.Encode(wrappedInf1))

	var streamResp WrappedResponse
	var sr InferenceStreamResponse

	// First chunk (STREAM) for first request
	require.NoError(t, decoder.Decode(&streamResp))
	require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
	require.Equal(t, "inf-req-1", streamResp.RequestID)
	require.Equal(t, ResponseTypeStream, sr.Type)
	require.Equal(t, "Hello", sr.Text)

	// Now enqueue two more requests quickly to force batching
	infReq2 := &InferenceStreamRequest{
		ModelID: "test-model",
		Input:   "How are you?",
	}
	infReq3 := &InferenceStreamRequest{
		ModelID: "test-model",
		Input:   "What's the weather?",
	}

	wrappedInf2, _ := NewWrappedRequest("inf-req-2", infReq2)
	wrappedInf3, _ := NewWrappedRequest("inf-req-3", infReq3)

	require.NoError(t, encoder.Encode(wrappedInf2))
	require.NoError(t, encoder.Encode(wrappedInf3))

	// Read the remaining chunks for all three requests.
	// Each request gets 3 responses total: STREAM, STREAM, FINAL.
	// We've already read 1 for the first request, so total expected:
	// Request 1: 2 more responses
	// Request 2: 3 responses
	// Request 3: 3 responses
	// Total to read now = 2 + 3 + 3 = 8 responses.

	responsesByID := make(map[string][]InferenceStreamResponse)
	responsesByID["inf-req-1"] = []InferenceStreamResponse{sr} // already have first chunk

	for i := 0; i < 8; i++ {
		require.NoError(t, decoder.Decode(&streamResp))
		require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
		responsesByID[streamResp.RequestID] = append(responsesByID[streamResp.RequestID], sr)
	}

	// Validate all requests got the "Hello", " ", "world!" pattern
	validateMockResponsePattern(t, responsesByID["inf-req-1"])
	validateMockResponsePattern(t, responsesByID["inf-req-2"])
	validateMockResponsePattern(t, responsesByID["inf-req-3"])
}

func TestMindletReconnectEngine(t *testing.T) {
	helper := NewTestHelper(t)
	defer helper.Cleanup()

	// 1. Connect engine and send VolumeAttachRequest
	engineConn := helper.CreateConnection(EngineConnection)
	encoder := json.NewEncoder(engineConn)
	decoder := json.NewDecoder(engineConn)

	attachReq := &VolumeAttachRequest{
		Path: "/test/volume",
		Models: map[string]ModelSpec{
			"my-model": {
				ID:   "reconnect-model",
				Type: Model8B,
			},
		},
	}
	wrappedAttach, _ := NewWrappedRequest("attach-req-1", attachReq)
	require.NoError(t, encoder.Encode(wrappedAttach))

	var attachResp WrappedResponse
	require.NoError(t, decoder.Decode(&attachResp))
	var attachData VolumeAttachResponse
	require.NoError(t, json.Unmarshal(attachResp.Data, &attachData))
	require.Equal(t, ResponseStatusSuccess, attachData.Status)

	// 2. Simulate engine disconnect by closing engineConn
	engineConn.Close()

	// 3. Reconnect engine
	engineConn2 := helper.CreateConnection(EngineConnection)
	encoder2 := json.NewEncoder(engineConn2)
	decoder2 := json.NewDecoder(engineConn2)

	// Wait a bit for Mindlet to accept and set new engine connection
	time.Sleep(200 * time.Millisecond)

	// 4. Send another inference request and verify it still works
	infReq := &InferenceStreamRequest{
		ModelID: "reconnect-model",
		Input:   "Hello again!",
	}
	wrappedInf, _ := NewWrappedRequest("inf-req-2", infReq)
	require.NoError(t, encoder2.Encode(wrappedInf))

	// Expect streamed responses as before
	var streamResp WrappedResponse
	var sr InferenceStreamResponse

	require.NoError(t, decoder2.Decode(&streamResp))
	require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
	require.Equal(t, ResponseTypeStream, sr.Type)
	require.Equal(t, "Hello", sr.Text)

	require.NoError(t, decoder2.Decode(&streamResp))
	require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
	require.Equal(t, ResponseTypeStream, sr.Type)
	require.Equal(t, " ", sr.Text)

	done := make(chan struct{})
	go func() {
		require.NoError(t, decoder2.Decode(&streamResp))
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(1 * time.Second):
		t.Fatal("Timed out waiting for final response after reconnect")
	}

	require.NoError(t, json.Unmarshal(streamResp.Data, &sr))
	require.Equal(t, ResponseTypeFinal, sr.Type)
	require.Equal(t, "world!", sr.Text)
}
