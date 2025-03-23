package src

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

type Mindlet struct {
	config          *MindletConfig
	logger          *Logger
	listener        net.Listener
	connState       *ConnectionState
	model           *ModelManager
	inferenceServer *InferenceServer
	done            chan struct{}
	mu              sync.RWMutex
}

func NewMindlet(cfg *MindletConfig) (*Mindlet, error) {
	logger, err := NewLogger()
	if err != nil {
		return nil, fmt.Errorf("failed to create mindlet logger: %v", err)
	}

	modelManager, err := NewModelManager(cfg, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create model manager: %v", err)
	}

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Port))
	if err != nil {
		return nil, fmt.Errorf("failed to create listener: %v", err)
	}

	connState := NewConnectionState()

	// Create the InferenceServer with proper connections
	llmServer := NewDefaultLLMServer(cfg, logger)
	inferenceServer := NewInferenceServer(cfg, logger, modelManager, connState, llmServer)

	return &Mindlet{
		config:          cfg,
		logger:          logger,
		model:           modelManager,
		inferenceServer: inferenceServer,
		listener:        listener,
		connState:       connState,
		done:            make(chan struct{}),
	}, nil
}

func (m *Mindlet) Start(ctx context.Context) error {
	// Start accepting connections
	go m.acceptConnections(ctx)

	m.logger.Info("Mindlet", "Mindlet begun")

	// Wait for the engine connection
	if err := m.connState.WaitForConnection(ctx, EngineConnection, 5*time.Second); err != nil {
		return fmt.Errorf("failed waiting for engine connection: %v", err)
	}

	// Now start model manager and inference server
	if err := m.model.Start(ctx); err != nil {
		return fmt.Errorf("failed to start model manager: %v", err)
	}

	if err := m.inferenceServer.Start(ctx); err != nil {
		return fmt.Errorf("failed to start inference server: %v", err)
	}

	// Wait for inference connection
	if err := m.connState.WaitForConnection(ctx, InferenceConnection, 5*time.Second); err != nil {
		return fmt.Errorf("failed waiting for inference connection: %v", err)
	}

	m.logger.Info("Mindlet", "Mindlet started")

	<-ctx.Done()
	close(m.done)
	return nil
}

func (m *Mindlet) acceptConnections(ctx context.Context) {
	for {
		conn, err := m.listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				m.logger.Error("Mindlet", "Failed to accept connection: %v", err)
				continue
			}
		}
		go m.handleConnection(conn)
	}
}

func (m *Mindlet) handleConnection(conn net.Conn) {
	defer conn.Close()

	framedConn := NewFramedConn(conn)

	initMsg, err := framedConn.ReadMessage()
	if err != nil {
		m.logger.Error("Mindlet", "Failed to read initial message: %v", err)
		return
	}

	if initMsg.Action != ActionTypeInitialConnection {
		m.logger.Error("Mindlet", "Expected initial connection message, got: %s", initMsg.Action)
		return
	}

	var initData InitialConnectionMessage
	if err := json.Unmarshal(initMsg.Data, &initData); err != nil {
		m.logger.Error("Mindlet", "Failed to unmarshal initial connection data: %v", err)
		return
	}

	connection := NewConnection(framedConn, initData.ConnectionType)

	switch initData.ConnectionType {
	case EngineConnection, InferenceConnection:
		m.logger.Info("Mindlet", "Connection established: %s", initData.ConnectionType)
		if m.connState.CheckConnection(initData.ConnectionType) {
			m.logger.Error("Mindlet", "Connection already established: %s", initData.ConnectionType)
			m.sendError(connection, "Connection already established")
			return
		}
		if err := m.connState.SetConnection(connection); err != nil {
			m.logger.Error("Mindlet", "Failed to set connection: %v", err)
			m.sendError(connection, fmt.Sprintf("Failed to establish connection: %v", err))
			return
		}
	default:
		m.logger.Error("Mindlet", "Unknown connection type: %s", initData.ConnectionType)
		m.sendError(connection, "Unknown connection type")
		return
	}

	defer m.connState.ClearConnection(initData.ConnectionType)
	for {
		msg, err := connection.Receive()
		if err != nil {
			if err == io.EOF {
				m.logger.Info("Mindlet", "Connection closed: %s", initData.ConnectionType)
			} else {
				m.logger.Error("Mindlet", "Failed to receive message: %v", err)
			}
			return
		}

		if err := m.HandleMessage(msg); err != nil {
			m.logger.Error("Mindlet", "Failed to handle message: %v", err)
			// Consider whether to send an error response here
		}
	}
}

func (m *Mindlet) HandleMessage(msg *UnifiedMessage) error {
	switch msg.Type {
	case MessageTypeRequest:
		return m.handleRequest(msg)
	case MessageTypeResponse:
		return m.handleResponse(msg)
	default:
		return fmt.Errorf("unknown message type: %s", msg.Type)
	}
}

func (m *Mindlet) handleRequest(msg *UnifiedMessage) error {
	switch msg.Action {
	case ActionTypeInferenceStream:
		return m.handleEngineInferenceRequest(msg)
	case ActionTypeLoadModel:
		return m.handleEngineLoadModelRequest(msg)
	case ActionTypeVolumeAttach:
		return m.handleEngineVolumeAttachRequest(msg)
	case ActionTypeVolumeDetach:
		return m.handleEngineVolumeDetachRequest(msg)
	case ActionTypeLoadCheckpoint:
		return m.handleEngineLoadCheckpointRequest(msg)
	default:
		return fmt.Errorf("unknown request action: %s", msg.Action)
	}
}

func (m *Mindlet) handleResponse(msg *UnifiedMessage) error {
	switch msg.Action {
	case ActionTypeInferenceBatchStream:
		return m.handleInferenceBatchResponse(msg)
	case ActionTypeLoadModel:
		return m.handleLoadModelResponse(msg)
	default:
		return fmt.Errorf("unknown response action: %s", msg.Action)
	}
}

func (m *Mindlet) handleEngineInferenceRequest(msg *UnifiedMessage) error {
	var req InferenceStreamRequest
	if err := msg.UnmarshalData(&req); err != nil {
		return err
	}

	if err := m.inferenceServer.RequestInference(&req, msg.ID); err != nil {
		return err
	}

	return nil
}

func (m *Mindlet) handleEngineLoadModelRequest(msg *UnifiedMessage) error {
	var req LoadModelRequest
	if err := msg.UnmarshalData(&req); err != nil {
		return err
	}

	forwardMsg, err := NewRequestMessage(msg.ID, ActionTypeLoadModel, &req)
	if err != nil {
		return err
	}

	if err := m.connState.SendBlocking(context.Background(), InferenceConnection, forwardMsg); err != nil {
		m.logger.Error("Mindlet", "Failed to forward load model request: %v", err)
		return err
	}

	return nil
}

func (m *Mindlet) handleEngineVolumeAttachRequest(msg *UnifiedMessage) error {
	var req VolumeAttachRequest
	if err := msg.UnmarshalData(&req); err != nil {
		return err
	}

	if err := m.model.AttachVolume(req.Path, req.Models); err != nil {
		return err
	}

	resp, err := NewResponseMessage(msg.ID, ActionTypeVolumeAttach, ResponseStatusSuccess, &VolumeAttachResponse{
		Status:  ResponseStatusSuccess,
		Message: "Volume attached",
	})
	if err != nil {
		return err
	}

	return m.connState.SendBlocking(context.Background(), EngineConnection, resp)
}

func (m *Mindlet) handleEngineVolumeDetachRequest(msg *UnifiedMessage) error {
	var req VolumeDetachRequest
	if err := msg.UnmarshalData(&req); err != nil {
		return err
	}

	if err := m.model.DetachVolume(req.Path); err != nil {
		return err
	}

	// Handle volume detachment
	resp, err := NewResponseMessage(msg.ID, ActionTypeVolumeDetach, ResponseStatusSuccess, &VolumeDetachResponse{
		Status:  ResponseStatusSuccess,
		Message: fmt.Sprintf("Volume detached: %s", req.Path),
	})
	if err != nil {
		return err
	}

	return m.connState.SendBlocking(context.Background(), EngineConnection, resp)
}

func (m *Mindlet) handleEngineLoadCheckpointRequest(msg *UnifiedMessage) error {
	var req LoadCheckpointRequest
	if err := msg.UnmarshalData(&req); err != nil {
		return err
	}

	m.logger.Info("Mindlet", "Processing load checkpoint request: %v", req)
	modelID := m.model.GetDefaultModelID()
	if err := m.model.LoadCheckpoint(modelID, req.Checkpoint, false); err != nil {
		errResp, _ := NewResponseMessage(msg.ID, ActionTypeLoadCheckpoint, ResponseStatusError, &LoadCheckpointResponse{
			Type:    ResponseTypeFinal,
			Status:  ResponseStatusError,
			Message: fmt.Sprintf("Failed to load checkpoint: %v", err),
		})
		return m.connState.SendBlocking(context.Background(), InferenceConnection, errResp)
	}

	resp, _ := NewResponseMessage(msg.ID, ActionTypeLoadCheckpoint, ResponseStatusSuccess, &LoadCheckpointResponse{
		Type:    ResponseTypeFinal,
		Status:  ResponseStatusSuccess,
		Message: fmt.Sprintf("Checkpoint %d is loaded", req.Checkpoint),
	})
	m.logger.Info("Mindlet", "Checkpoint loaded: %d", req.Checkpoint)
	return m.connState.SendBlocking(context.Background(), InferenceConnection, resp)
}

func (m *Mindlet) handleInferenceBatchResponse(msg *UnifiedMessage) error {
	var resp InferenceBatchStreamResponse
	if err := msg.UnmarshalData(&resp); err != nil {
		return err
	}

	return m.inferenceServer.HandleBatchResponse(&resp)
}

func (m *Mindlet) handleLoadModelResponse(msg *UnifiedMessage) error {
	// Forward the response to the engine connection
	if err := m.connState.SendBlocking(context.Background(), EngineConnection, msg); err != nil {
		m.logger.Error("Mindlet", "Failed to forward load model response to engine: %v", err)
	}

	return nil
}

func (m *Mindlet) sendError(conn *Connection, message string) {
	errorMsg, _ := NewResponseMessage(GenerateUUID(), ActionTypeError, ResponseStatusError, map[string]string{"error": message})
	conn.Send(errorMsg)
}
