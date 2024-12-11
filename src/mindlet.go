package src

import (
	"context"
	"encoding/json"
	"fmt"
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

	var initMsg InitialConnectionMessage
	if err := json.NewDecoder(conn).Decode(&initMsg); err != nil {
		m.logger.Error("Mindlet", "Failed to decode initial message: %v", err)
		return
	}

	m.logger.Info("Mindlet", "Connection established: %s", initMsg.ConnectionType)
	connection := &Connection{
		conn:     conn,
		connType: initMsg.ConnectionType,
	}

	switch initMsg.ConnectionType {
	case EngineConnection, InferenceConnection:
		if m.connState.CheckConnection(initMsg.ConnectionType) {
			m.logger.Error("Mindlet", "Connection already established: %s", initMsg.ConnectionType)
			m.sendError(connection, "Connection already established")
			return
		}
		if err := m.connState.SetConnection(connection); err != nil {
			m.logger.Error("Mindlet", "Failed to set connection: %v", err)
			m.sendError(connection, fmt.Sprintf("Failed to establish connection: %v", err))
			return
		}
	default:
		m.logger.Error("Mindlet", "Unknown connection type: %s", initMsg.ConnectionType)
		m.sendError(connection, "Unknown connection type")
		return
	}

	switch initMsg.ConnectionType {
	case EngineConnection:
		m.handleEngineConnection(connection)
	case InferenceConnection:
		m.handleInferenceConnection(connection)
	}
}

// Connection handling methods
func (m *Mindlet) handleEngineConnection(conn *Connection) {
	defer m.connState.ClearConnection(EngineConnection)

	for {
		wrapped, err := conn.Receive()
		if err != nil {
			m.logger.Error("Mindlet", "Engine connection lost: %v", err)
			m.connState.ClearConnection(EngineConnection)
			return
		}

		req, err := UnwrapRequest(wrapped, conn.GetRemoteAddr())
		if err != nil {
			m.logger.Error("Mindlet", "Failed to unwrap request: %v", err)
			continue
		}

		switch req.Type {
		case RequestTypeInferenceStream:
			fmt.Println("Mindlet", "Inference stream request", req.RequestID)
			data := req.Data.(*InferenceStreamRequest)
			// Request inference from server and get the response channel.
			respChan, err := m.inferenceServer.RequestInference(data, req.RequestID)
			if err != nil {
				m.sendError(conn, fmt.Sprintf("Failed to request inference: %v", err))
				continue
			}

			// Stream responses back to engine
			go func() {
				for resp := range respChan {
					wrappedResp, _ := NewWrappedResponse(wrapped.RequestID, resp)
					if err := conn.Send(wrappedResp); err != nil {
						m.logger.Error("Mindlet", "Failed to send response to engine: %v", err)
						break
					}
				}
			}()

		case RequestTypeVolumeAttach:
			m.logger.Info("Mindlet", "Volume attach request")
			data := req.Data.(*VolumeAttachRequest)
			if err := m.model.AttachVolume(data.Path, data.Models); err != nil {
				m.sendError(conn, fmt.Sprintf("Failed to create model: %v", err))
				continue
			}
			m.logger.Info("Mindlet", "Volume attached: %s", data.Path)
			resp, _ := NewWrappedResponse(wrapped.RequestID, &VolumeAttachResponse{
				Status:  ResponseStatusSuccess,
				Message: "Volume attached",
			})
			if err := conn.Send(resp); err != nil {
				m.logger.Error("Mindlet", "Failed to send response: %v", err)
			}
			m.logger.Info("Mindlet", "Volume attached and reply sent: %s", data.Path)

		case RequestTypeVolumeDetach:
			data := req.Data.(*VolumeDetachRequest)
			// Handle volume detachment
			resp, _ := NewWrappedResponse(wrapped.RequestID, &VolumeDetachResponse{
				Status:  ResponseStatusSuccess,
				Message: fmt.Sprintf("Volume detached: %s", data.Path),
			})
			if err := conn.Send(resp); err != nil {
				m.logger.Error("Mindlet", "Failed to send response: %v", err)
			}

		default:
			m.logger.Warn("Mindlet", "Unknown request type: %s", req.Type)
		}
	}
}

func (m *Mindlet) handleInferenceConnection(conn *Connection) {
	defer m.connState.ClearConnection(InferenceConnection)

	for {
		wrapped, err := conn.Receive()
		if err != nil {
			m.logger.Error("Mindlet", "Failed to receive inference message: %v", err)
			return
		}

		switch wrapped.Type {
		case RequestTypeInferenceBatchStream:
			var resp InferenceBatchStreamResponse
			if err := json.Unmarshal(wrapped.Data, &resp); err != nil {
				m.logger.Error("Mindlet", "Failed to unmarshal inference response: %v", err)
				continue
			}

			if err := m.inferenceServer.HandleBatchResponse(&resp); err != nil {
				m.logger.Error("Mindlet", "Failed to handle batch response: %v", err)
			}

		case RequestTypeLoadCheckpoint:
			var req LoadCheckpointRequest
			if err := json.Unmarshal(wrapped.Data, &req); err != nil {
				m.logger.Error("Mindlet", "Failed to unmarshal checkpoint request: %v", err)
				continue
			}

			modelID := m.model.GetDefaultModelID()
			if err := m.model.LoadCheckpoint(modelID, req.Checkpoint, false); err != nil {
				m.sendError(conn, fmt.Sprintf("Failed to load checkpoint: %v", err))
				continue
			}

			resp, _ := NewWrappedResponse(wrapped.RequestID, &LoadCheckpointResponse{
				Type:    ResponseTypeFinal,
				Status:  ResponseStatusSuccess,
				Message: fmt.Sprintf("Checkpoint %d is loaded", req.Checkpoint),
			})
			if err := conn.Send(resp); err != nil {
				m.logger.Error("Mindlet", "Failed to send response: %v", err)
			}

		default:
			m.logger.Warn("Mindlet", "Unknown request type: %s", wrapped.Type)
		}
	}
}

func (m *Mindlet) sendError(conn *Connection, message string) {
	conn.Send(map[string]string{
		"status": "error",
		"error":  message,
	})
}
