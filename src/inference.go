package src

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"io"
	"net"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

type LLMServer interface {
	Start(ctx context.Context) error
	Stop() error
}

// InferenceServer handles request queueing and response routing
type InferenceServer struct {
	config          *MindletConfig
	logger          *Logger
	model           *ModelManager
	connState       *ConnectionState
	llmServer       LLMServer
	requestsMu      sync.Mutex
	requestQueue    []*internalInferenceRequest
	batchMap        *utils.ConcurrentMap[string, map[int]string]
	batchMu         sync.Mutex
	inferenceBatch  internalInferenceBatch
	lastBatchID     int
	processingBatch atomic.Bool
}

type internalInferenceRequest struct {
	RequestID string
	Request   *InferenceStreamRequest
}

type internalInferenceBatch struct {
	BatchID  string
	Requests []*internalInferenceRequest
}

func NewInferenceServer(cfg *MindletConfig, logger *Logger, modelManager *ModelManager, connState *ConnectionState,
	llmServer LLMServer) *InferenceServer {
	return &InferenceServer{
		config:    cfg,
		logger:    logger,
		model:     modelManager,
		connState: connState,
		llmServer: llmServer,
		batchMap:  utils.NewConcurrentMap[string, map[int]string](),
	}
}

func (s *InferenceServer) Start(ctx context.Context) error {
	if err := s.llmServer.Start(ctx); err != nil {
		return err
	}

	go s.processRequests(ctx)
	return nil
}

// RequestInference queues an inference request and returns a channel for responses
func (s *InferenceServer) RequestInference(req *InferenceStreamRequest, id string) error {
	if _, err := s.model.GetModel(req.ModelID); err != nil {
		return fmt.Errorf("model not found: %s", req.ModelID)
	}

	s.requestsMu.Lock()
	defer s.requestsMu.Unlock()
	s.requestQueue = append(s.requestQueue, &internalInferenceRequest{
		RequestID: id,
		Request:   req,
	})
	return nil
}

func (s *InferenceServer) processRequests(ctx context.Context) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.requestsMu.Lock()
			if len(s.requestQueue) == 0 {
				s.requestsMu.Unlock()
				continue
			}
			if s.processingBatch.Load() {
				s.logger.Info("InferenceServer", "Already processing a batch")
				s.requestsMu.Unlock()
				continue
			}

			batchSize := min(len(s.requestQueue), s.config.BatchSize)
			batch := s.requestQueue[:batchSize]
			s.requestQueue = s.requestQueue[batchSize:]
			s.processingBatch.Store(true)
			s.requestsMu.Unlock()

			s.logger.Info("InferenceServer", "Processing batch of %d requests", batchSize)
			s.sendBatchRequest(ctx, batch)
		}
	}
}

func (s *InferenceServer) sendBatchRequest(ctx context.Context, batch []*internalInferenceRequest) {
	s.batchMu.Lock()
	defer s.batchMu.Unlock()

	s.inferenceBatch = internalInferenceBatch{
		BatchID:  GetBatchID(s.lastBatchID),
		Requests: batch,
	}
	s.lastBatchID++

	var reqs []*InferenceStreamRequest
	for _, req := range batch {
		reqs = append(reqs, req.Request)
	}
	requestMap := make(map[int]string)
	for i, req := range batch {
		requestMap[i] = req.RequestID
	}
	s.batchMap.Set(s.inferenceBatch.BatchID, requestMap)

	// Send batch request as before
	batchReq := &InferenceBatchStreamRequest{
		BatchID:  s.inferenceBatch.BatchID,
		Requests: reqs,
	}
	s.logger.Info("InferenceServer", "Sending batch request: %s, with %v requests", s.inferenceBatch.BatchID, len(batchReq.Requests))

	msg, err := NewRequestMessage(s.inferenceBatch.BatchID, ActionTypeInferenceBatchStream, batchReq)
	if err != nil {
		s.logger.Error("InferenceServer", "Failed to create batch request message: %v", err)
		s.processingBatch.Store(false)
		return
	}

	if err := s.connState.SendBlocking(ctx, InferenceConnection, msg); err != nil {
		s.logger.Error("InferenceServer", "Failed to send request to InferenceConnection: %v", err)
		s.processingBatch.Store(false)
		return
	}
}

func GetBatchID(id int) string {
	return fmt.Sprintf("batch-%d", id)
}

func (s *InferenceServer) HandleBatchResponse(response *InferenceBatchStreamResponse) error {
	s.batchMu.Lock()
	defer s.batchMu.Unlock()

	s.logger.Info("InferenceServer", "Handling batch response for BatchID: %s", response.BatchID)

	requestMap, exists := s.batchMap.Get(response.BatchID)
	if !exists {
		return fmt.Errorf("batch mapping not found: %s", response.BatchID)
	}

	for i, output := range response.Outputs {
		requestID := requestMap[i]
		s.logger.Info("InferenceServer", "Processing output for RequestID: %s, Type: %s", requestID, response.Type[i])

		respMsg, err := NewResponseMessage(requestID, ActionTypeInferenceStream, response.Status, &InferenceStreamResponse{
			Type:   response.Type[i],
			Text:   output,
			Status: response.Status,
		})
		if err != nil {
			s.logger.Error("InferenceServer", "Failed to create response message: %v", err)
			continue
		}

		if err := s.connState.SendBlocking(context.Background(), EngineConnection, respMsg); err != nil {
			s.logger.Error("InferenceServer", "Failed to send response to engine: %v", err)
		} else {
			s.logger.Info("InferenceServer", "Sent response for RequestID: %s, Type: %s", requestID, response.Type[i])
		}
	}

	if s.allFinal(response.Type) {
		s.batchMap.Remove(response.BatchID)
		s.processingBatch.Store(false)
		s.logger.Info("InferenceServer", "Batch %s is complete", response.BatchID)
	}

	return nil
}
func (s *InferenceServer) allFinal(types []ResponseType) bool {
	for _, t := range types {
		if t != ResponseTypeFinal {
			return false
		}
	}
	return true
}

// DefaultLLMServerServer implements the real Python backend
type DefaultLLMServer struct {
	config    *MindletConfig
	logger    *Logger
	pythonCmd *exec.Cmd
}

func NewDefaultLLMServer(cfg *MindletConfig, logger *Logger) *DefaultLLMServer {
	return &DefaultLLMServer{
		config: cfg,
		logger: logger,
	}
}

func (s *DefaultLLMServer) Start(ctx context.Context) error {
	cmd := exec.Command(s.config.PythonPath, "-m", "scripts.start_llm_server",
		"--model", string(s.config.ModelType),
		"--model_root_dir", s.config.RamFsRoot,
		"--max_sequence_length", fmt.Sprintf("%d", s.config.MaxSeqLength),
		"--system_prompt", s.config.SystemPrompt,
		"--port", fmt.Sprintf("%d", s.config.Port),
	)
	cmd.Dir = "/home/nlpfollower/Desktop/deltamind/chat_model"

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start LLM server: %v", err)
	}

	s.pythonCmd = cmd

	// Log stdout
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			s.logger.Info("LLMServer", "stdout: %s", scanner.Text())
		}
	}()

	// Log stderr
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			s.logger.Error("LLMServer", "stderr: %s", scanner.Text())
		}
	}()

	go func() {
		<-ctx.Done()
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	}()

	return nil
}

func (s *DefaultLLMServer) Stop() error {
	if s.pythonCmd != nil && s.pythonCmd.Process != nil {
		return s.pythonCmd.Process.Kill()
	}
	return nil
}

type MockLLMServer struct {
	logger    *Logger
	cfg       *MindletConfig
	connState *ConnectionState
}

func NewMockLLMServer(cfg *MindletConfig, logger *Logger, connState *ConnectionState) *MockLLMServer {
	return &MockLLMServer{
		logger:    logger,
		cfg:       cfg,
		connState: connState,
	}
}

func (s *MockLLMServer) Start(ctx context.Context) error {
	go func() {
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", s.cfg.Port))
		if err != nil {
			s.logger.Error("MockLLMServer", "Failed to connect to mindlet: %v", err)
			return
		}

		framedConn := NewFramedConn(conn)

		initData := InitialConnectionMessage{
			ConnectionType: InferenceConnection,
			ClientID:       "mock-llm-server",
		}

		initMsg, err := NewRequestMessage(GenerateUUID(), ActionTypeInitialConnection, initData)
		if err != nil {
			s.logger.Error("MockLLMServer", "Failed to create init message: %v", err)
			conn.Close()
			return
		}

		if err := framedConn.WriteMessage(initMsg); err != nil {
			s.logger.Error("MockLLMServer", "Failed to send init message: %v", err)
			conn.Close()
			return
		}

		// Start handling responses on this connection
		go s.handleMessages(framedConn)
	}()

	return nil
}

func (s *MockLLMServer) Stop() error {
	// Nothing to stop in mock
	return nil
}

func (s *MockLLMServer) handleMessages(conn *FramedConn) {
	for {
		s.logger.Info("MockLLMServer", "Waiting for message")
		msg, err := conn.ReadMessage()
		if err != nil {
			if err != io.EOF {
				s.logger.Error("MockLLMServer", "Failed to read message: %v", err)
			}
			return
		}
		s.logger.Info("MockLLMServer", "Received message: Type=%s, Action=%s", msg.Type, msg.Action)

		switch msg.Action {
		case ActionTypeInferenceBatchStream:
			var batchReq InferenceBatchStreamRequest
			if err := json.Unmarshal(msg.Data, &batchReq); err != nil {
				s.logger.Error("MockLLMServer", "Failed to unmarshal batch request: %v", err)
				continue
			}
			go s.simulateResponses(conn, batchReq)
		case ActionTypeLoadModel:
			var loadReq LoadModelRequest
			if err := json.Unmarshal(msg.Data, &loadReq); err != nil {
				s.logger.Error("MockLLMServer", "Failed to unmarshal load model request: %v", err)
				continue
			}
			s.logger.Info("MockLLMServer", "Received LoadModelRequest for model: %s", loadReq.ModelName)
			resp := LoadModelResponse{
				Status:  ResponseStatusSuccess,
				Message: fmt.Sprintf("Model %s loaded successfully", loadReq.ModelName),
			}
			respMsg, _ := NewResponseMessage(msg.ID, ActionTypeLoadModel, ResponseStatusSuccess, resp)
			if err := conn.WriteMessage(respMsg); err != nil {
				s.logger.Error("MockLLMServer", "Failed to send load model response: %v", err)
				continue
			}
			s.logger.Info("MockLLMServer", "Sent LoadModel response for model: %s", loadReq.ModelName)
		default:
			s.logger.Warn("MockLLMServer", "Unknown action type: %s", msg.Action)
		}
	}
}

func (s *MockLLMServer) simulateResponses(conn *FramedConn, req InferenceBatchStreamRequest) {
	numRequests := len(req.Requests)
	chunks := []string{"Hello", " ", "world!"}
	types := []ResponseType{ResponseTypeStream, ResponseTypeStream, ResponseTypeFinal}

	for i, chunk := range chunks {
		outputs := make([]string, numRequests)
		responseTypes := make([]ResponseType, numRequests)

		for j := 0; j < numRequests; j++ {
			outputs[j] = chunk
			responseTypes[j] = types[i]
		}

		resp := &InferenceBatchStreamResponse{
			BatchID: req.BatchID,
			Type:    responseTypes,
			Outputs: outputs,
			Status:  ResponseStatusSuccess,
		}

		respMsg, _ := NewResponseMessage(req.BatchID, ActionTypeInferenceBatchStream, ResponseStatusSuccess, resp)

		if err := conn.WriteMessage(respMsg); err != nil {
			s.logger.Error("MockLLMServer", "Failed to send response: %v", err)
			return
		}
		s.logger.Info("MockLLMServer", "Sent response chunk %d/%d for BatchID: %s", i+1, len(chunks), req.BatchID)

		time.Sleep(50 * time.Millisecond)
	}
}
