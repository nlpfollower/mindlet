package src

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os/exec"
	"path/filepath"
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
	requestQueue    []*internalInferenceRequest
	responseStreams map[string]chan *InferenceStreamResponse
	inferenceBatch  internalInferenceBatch
	lastBatchID     int
	processingBatch atomic.Bool
	queueMu         sync.Mutex
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
		config:          cfg,
		logger:          logger,
		model:           modelManager,
		connState:       connState,
		llmServer:       llmServer,
		responseStreams: make(map[string]chan *InferenceStreamResponse),
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
func (s *InferenceServer) RequestInference(req *InferenceStreamRequest, id string) (chan *InferenceStreamResponse, error) {
	// Check if model exists
	if _, err := s.model.GetModel(req.ModelID); err != nil {
		return nil, fmt.Errorf("model not found: %s", req.ModelID)
	}

	s.queueMu.Lock()
	defer s.queueMu.Unlock()
	s.requestQueue = append(s.requestQueue, &internalInferenceRequest{
		RequestID: id,
		Request:   req,
	})
	s.responseStreams[id] = make(chan *InferenceStreamResponse)
	return s.responseStreams[id], nil
}

func (s *InferenceServer) processRequests(ctx context.Context) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.queueMu.Lock()
			if len(s.requestQueue) == 0 || s.processingBatch.Load() {
				s.queueMu.Unlock()
				continue
			}

			batchSize := min(len(s.requestQueue), s.config.BatchSize)
			batch := s.requestQueue[:batchSize]
			s.requestQueue = s.requestQueue[batchSize:]
			s.queueMu.Unlock()

			s.processingBatch.Store(true)
			s.inferenceBatch = internalInferenceBatch{
				BatchID:  GetBatchID(s.lastBatchID),
				Requests: batch,
			}
			s.lastBatchID++

			var reqs []*InferenceStreamRequest
			for _, req := range batch {
				reqs = append(reqs, req.Request)
			}
			batchReq := &InferenceBatchStreamRequest{
				BatchID:  s.inferenceBatch.BatchID,
				Requests: reqs,
			}
			s.logger.Info("InferenceServer", "Sending batch request: %s, with %v request", s.inferenceBatch.BatchID,
				len(batchReq.Requests))

			if err := s.connState.SendBlocking(ctx, InferenceConnection, batchReq); err != nil {
				s.logger.Error("InferenceServer", "failed to send request to InferenceConnection: %v", err)
				s.processingBatch.Store(false)
				continue
			}
		}
	}
}

func GetBatchID(id int) string {
	return fmt.Sprintf("batch-%d", id)
}

func (s *InferenceServer) HandleBatchResponse(response *InferenceBatchStreamResponse) error {
	// Perform some basic validation, cause it's Python.
	if response.BatchID != s.inferenceBatch.BatchID {
		return fmt.Errorf("response batch ID mismatch: expected %s, got %s", s.inferenceBatch.BatchID, response.BatchID)
	}

	if len(response.Outputs) != len(s.inferenceBatch.Requests) ||
		len(response.Type) != len(s.inferenceBatch.Requests) {
		return fmt.Errorf("response length mismatch: expected %d, got %d",
			len(s.inferenceBatch.Requests), len(response.Outputs))
	}

	for ii, req := range s.inferenceBatch.Requests {
		resp := &InferenceStreamResponse{
			Type:   response.Type[ii],
			Text:   response.Outputs[ii],
			Status: response.Status,
		}

		// Check that the response stream map contains the request ID
		if _, ok := s.responseStreams[req.RequestID]; !ok {
			s.logger.Error("InferenceServer", "response stream not found for request ID: %s", req.RequestID)
			continue
		}

		switch response.Type[ii] {
		case ResponseTypeStream:
			s.responseStreams[req.RequestID] <- resp
		case ResponseTypeFinal:
			s.responseStreams[req.RequestID] <- resp
			s.processingBatch.Store(false)
			close(s.responseStreams[req.RequestID])
			delete(s.responseStreams, req.RequestID)
		case ResponseTypeEmpty:
			continue
		default:
			s.logger.Error("InferenceServer", "Unknown response type: %s", response.Type[ii])
		}
	}

	return nil
}

// DefaultLLMServerServer implements the real Python backend
type DefaultLLMServer struct {
	config    *MindletConfig
	logger    *Logger
	conn      *Connection
	pythonCmd *exec.Cmd
}

func NewDefaultLLMServer(cfg *MindletConfig, logger *Logger) *DefaultLLMServer {
	return &DefaultLLMServer{
		config: cfg,
		logger: logger,
	}
}

func (s *DefaultLLMServer) Start(ctx context.Context) error {
	pythonPath := filepath.Join(s.config.ProjectRoot, s.config.BootDir, s.config.PythonPath)
	workingDir := filepath.Join(s.config.ProjectRoot, s.config.BootDir, s.config.TrainingDir)

	cmd := exec.Command(pythonPath, "-m", "scripts.inference_server",
		"--model", "llama3",
		"--model_path", s.config.RamFsRoot,
		"--max_sequence_length", fmt.Sprintf("%d", s.config.MaxSeqLength),
		"--system_prompt", s.config.SystemPrompt,
		"--debug",
		"--output_dir", filepath.Join(s.config.ProjectRoot, s.config.OutputDir, s.config.LogDir),
	)

	cmd.Dir = workingDir

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start inference server: %v", err)
	}

	s.pythonCmd = cmd

	go func() {
		<-ctx.Done()
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	}()

	return nil
}

func (s *DefaultLLMServer) Stop() error {
	if s.pythonCmd.Process != nil {
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
	// Simulate the server starting and creating connections
	go func() {
		// Connect to the mindlet server
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", s.cfg.Port))
		if err != nil {
			s.logger.Error("MockLLMServer", "Failed to connect to mindlet: %v", err)
			return
		}

		// Send initial connection message for inference connection
		initMsg := InitialConnectionMessage{
			ConnectionType: InferenceConnection,
			ClientID:       "mock-llm-server",
		}

		if err := json.NewEncoder(conn).Encode(initMsg); err != nil {
			s.logger.Error("MockLLMServer", "Failed to send init message: %v", err)
			conn.Close()
			return
		}

		// Start handling responses on this connection
		go s.handleResponses(conn)
	}()

	return nil
}

func (s *MockLLMServer) Stop() error {
	// Nothing to stop in mock
	return nil
}

func (s *MockLLMServer) handleResponses(conn net.Conn) {
	defer conn.Close()

	decoder := json.NewDecoder(conn)
	for {
		var req WrappedRequest
		if err := decoder.Decode(&req); err != nil {
			if err != io.EOF {
				s.logger.Error("MockLLMServer", "Failed to decode request: %v", err)
			}
			return
		}
		s.logger.Info("MockLLMServer", "Received request: %s", req.Type)

		// Handle inference batch requests
		if req.Type == RequestTypeInferenceBatchStream {
			var batchReq InferenceBatchStreamRequest
			if err := json.Unmarshal(req.Data, &batchReq); err != nil {
				s.logger.Error("MockLLMServer", "Failed to unmarshal batch request: %v", err)
				continue
			}

			// Simulate streaming responses
			go s.simulateResponses(conn, batchReq)
		}
	}
}

func (s *MockLLMServer) simulateResponses(conn net.Conn, req InferenceBatchStreamRequest) {
	numRequests := len(req.Requests)
	chunks := []string{"Hello", " ", "world!"}
	types := []ResponseType{ResponseTypeStream, ResponseTypeStream, ResponseTypeFinal}

	for i, chunk := range chunks {
		outputs := make([]string, numRequests)
		responseTypes := make([]ResponseType, numRequests)

		// For each chunk, fill responses for all requests in the batch
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

		wrappedResp, _ := NewWrappedResponse("batch-resp", resp)
		wrappedResp.Type = RequestTypeInferenceBatchStream

		if err := json.NewEncoder(conn).Encode(wrappedResp); err != nil {
			s.logger.Error("MockLLMServer", "Failed to send response: %v", err)
			return
		}

		time.Sleep(50 * time.Millisecond)
	}
}
