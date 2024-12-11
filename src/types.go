package src

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"
)

type ConnectionType string

const (
	EngineConnection    ConnectionType = "ENGINE"
	InferenceConnection ConnectionType = "INFERENCE"
)

type InitialConnectionMessage struct {
	ConnectionType ConnectionType `json:"connection_type"`
	ClientID       string         `json:"client_id"`
}

type Connection struct {
	conn     net.Conn
	connType ConnectionType
	mu       sync.Mutex
}

func (c *Connection) Send(msg interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return json.NewEncoder(c.conn).Encode(msg)
}

func (c *Connection) Receive() (*WrappedRequest, error) {
	var wrapped *WrappedRequest
	if err := json.NewDecoder(c.conn).Decode(&wrapped); err != nil {
		return nil, err
	}
	return wrapped, nil
}

func (c *Connection) GetRemoteAddr() string {
	return c.conn.RemoteAddr().String()
}

type ConnectionState struct {
	engineConn    *Connection
	inferenceConn *Connection
	mu            sync.RWMutex
}

func NewConnectionState() *ConnectionState {
	return &ConnectionState{}
}

func (cs *ConnectionState) CheckConnection(connType ConnectionType) bool {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	switch connType {
	case EngineConnection:
		return cs.engineConn != nil
	case InferenceConnection:
		return cs.inferenceConn != nil
	default:
		return false
	}
}

func (cs *ConnectionState) SetConnection(conn *Connection) error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	switch conn.connType {
	case EngineConnection:
		if cs.engineConn != nil {
			cs.engineConn.conn.Close()
		}
		cs.engineConn = conn
	case InferenceConnection:
		if cs.inferenceConn != nil {
			cs.inferenceConn.conn.Close()
		}
		cs.inferenceConn = conn
	default:
		return fmt.Errorf("unknown connection type: %s", conn.connType)
	}
	return nil
}

func (cs *ConnectionState) ClearConnection(connType ConnectionType) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	switch connType {
	case EngineConnection:
		if cs.engineConn != nil {
			cs.engineConn.conn.Close()
			cs.engineConn = nil
		}
	case InferenceConnection:
		if cs.inferenceConn != nil {
			cs.inferenceConn.conn.Close()
			cs.inferenceConn = nil
		}
	}
}

func (cs *ConnectionState) SendBlocking(ctx context.Context, connType ConnectionType, req MindletRequest) error {
	for {
		requestID := GenerateUUID() // or pass in a known requestID if needed
		wrappedReq, err := NewWrappedRequest(requestID, req)
		if err != nil {
			return fmt.Errorf("failed to wrap request: %w", err)
		}

		// Wait for connection to be available
		if err := cs.WaitForConnection(ctx, connType, 0); err != nil {
			return fmt.Errorf("failed waiting for connection: %w", err)
		}

		// Get the connection under lock
		cs.mu.RLock()
		var conn *Connection
		switch connType {
		case EngineConnection:
			conn = cs.engineConn
		case InferenceConnection:
			conn = cs.inferenceConn
		}
		cs.mu.RUnlock()

		// Try to send the request
		if err := conn.Send(wrappedReq); err != nil {
			continue
		}

		return nil
	}
}

func (cs *ConnectionState) WaitForConnection(ctx context.Context, connType ConnectionType, timeout time.Duration) error {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	deadline := time.Now().Add(timeout)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if cs.CheckConnection(connType) {
				return nil
			}

			if timeout > 0 && time.Now().After(deadline) {
				return fmt.Errorf("%s connection not established within timeout", connType)
			}
		}
	}
}

type RequestType string
type ResponseType string
type ResponseStatus string

const (
	RequestTypeLoadCheckpoint       RequestType = "LOAD_CHECKPOINT"
	RequestTypeVolumeAttach         RequestType = "VOLUME_ATTACH"
	RequestTypeVolumeDetach         RequestType = "VOLUME_DETACH"
	RequestTypeInferenceStream      RequestType = "INFERENCE_STREAM"
	RequestTypeInferenceBatchStream RequestType = "INFERENCE_BATCH_STREAM"

	ResponseTypeFinal  ResponseType = "FINAL"
	ResponseTypeStream ResponseType = "STREAM"
	ResponseTypeEmpty  ResponseType = "EMPTY"

	ResponseStatusSuccess ResponseStatus = "SUCCESS"
	ResponseStatusError   ResponseStatus = "ERROR"
)

// Base interfaces remain the same
type MindletRequest interface {
	MindletRequestType() RequestType
}

type MindletResponse interface {
	MindletResponseType() RequestType
}

// Request/Response wrappers remain the same
type WrappedRequest struct {
	RequestID string          `json:"request_id"`
	Type      RequestType     `json:"type"`
	Data      json.RawMessage `json:"data"`
}

type WrappedResponse struct {
	RequestID string          `json:"request_id"`
	Type      RequestType     `json:"type"`
	Data      json.RawMessage `json:"data"`
}

// Internal request wrapper
type Request struct {
	RequestID    string                 `json:"request_id"`
	Type         RequestType            `json:"type"`
	ConnectionID string                 `json:"connection_id"`
	Data         MindletRequest         `json:"data"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
}

type Response struct {
	Response     *WrappedResponse
	ConnectionID string
	Timestamp    time.Time
}

// Request implementations
type LoadCheckpointRequest struct {
	Checkpoint int `json:"checkpoint"`
}

func (r *LoadCheckpointRequest) MindletRequestType() RequestType {
	return RequestTypeLoadCheckpoint
}

type LoadCheckpointResponse struct {
	Type    ResponseType   `json:"type"`
	Status  ResponseStatus `json:"status"`
	Message string         `json:"message"`
}

func (r *LoadCheckpointResponse) MindletResponseType() RequestType {
	return RequestTypeLoadCheckpoint
}

type VolumeAttachRequest struct {
	Path string `json:"path"`
	// Model ID -> Model Spec
	Models map[string]ModelSpec `json:"models"`
}

type ModelSpec struct {
	ID   string    `json:"id"`
	Type ModelType `json:"type"`
}

func (r *VolumeAttachRequest) MindletRequestType() RequestType {
	return RequestTypeVolumeAttach
}

type VolumeAttachResponse struct {
	Status  ResponseStatus `json:"status"`
	Message string         `json:"message"`
}

func (r *VolumeAttachResponse) MindletResponseType() RequestType {
	return RequestTypeVolumeAttach
}

type VolumeDetachRequest struct {
	Path string `json:"path"`
}

func (r *VolumeDetachRequest) MindletRequestType() RequestType {
	return RequestTypeVolumeDetach
}

type VolumeDetachResponse struct {
	Status  ResponseStatus `json:"status"`
	Message string         `json:"message"`
}

func (r *VolumeDetachResponse) MindletResponseType() RequestType {
	return RequestTypeVolumeDetach
}

type InferenceStreamRequest struct {
	ModelID string `json:"model_id"`
	Input   string `json:"input"`
}

func (r *InferenceStreamRequest) MindletRequestType() RequestType {
	return RequestTypeInferenceStream
}

type InferenceStreamResponse struct {
	Type   ResponseType   `json:"type"`
	Text   string         `json:"text,omitempty"`
	Status ResponseStatus `json:"status"`
}

func (r *InferenceStreamResponse) MindletResponseType() RequestType {
	return RequestTypeInferenceStream
}

type InferenceBatchStreamRequest struct {
	BatchID  string                    `json:"batch_id"`
	Requests []*InferenceStreamRequest `json:"requests"`
}

func (r *InferenceBatchStreamRequest) MindletRequestType() RequestType {
	return RequestTypeInferenceBatchStream
}

type InferenceBatchStreamResponse struct {
	BatchID string         `json:"batch_id"`
	Type    []ResponseType `json:"type"`
	Outputs []string       `json:"text"`
	Status  ResponseStatus `json:"status"`
}

func (r *InferenceBatchStreamResponse) MindletResponseType() RequestType {
	return RequestTypeInferenceBatchStream
}

// Helper functions for wrapping/unwrapping
func NewWrappedRequest[T MindletRequest](requestID string, req T) (*WrappedRequest, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	return &WrappedRequest{
		RequestID: requestID,
		Type:      req.MindletRequestType(),
		Data:      data,
	}, nil
}

func NewWrappedResponse[T MindletResponse](requestID string, resp T) (*WrappedResponse, error) {
	data, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}

	return &WrappedResponse{
		RequestID: requestID,
		Type:      resp.MindletResponseType(),
		Data:      data,
	}, nil
}

func UnwrapRequest(wrapped *WrappedRequest, connID string) (*Request, error) {
	switch wrapped.Type {
	case RequestTypeLoadCheckpoint:
		var req LoadCheckpointRequest
		if err := json.Unmarshal(wrapped.Data, &req); err != nil {
			return nil, err
		}
		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeLoadCheckpoint,
			ConnectionID: connID,
			Data:         &req,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeVolumeAttach:
		var req VolumeAttachRequest
		if err := json.Unmarshal(wrapped.Data, &req); err != nil {
			return nil, err
		}
		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeVolumeAttach,
			ConnectionID: connID,
			Data:         &req,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeVolumeDetach:
		var req VolumeDetachRequest
		if err := json.Unmarshal(wrapped.Data, &req); err != nil {
			return nil, err
		}
		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeVolumeDetach,
			ConnectionID: connID,
			Data:         &req,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeInferenceStream:
		var req InferenceStreamRequest
		if err := json.Unmarshal(wrapped.Data, &req); err != nil {
			return nil, err
		}
		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeInferenceStream,
			ConnectionID: connID,
			Data:         &req,
			CreatedAt:    time.Now(),
		}, nil
	}

	return nil, fmt.Errorf("unsupported request type: %s", wrapped.Type)
}

func UnwrapResponse[T MindletResponse](wrapped *WrappedResponse) (*T, error) {
	var response T
	if err := json.Unmarshal(wrapped.Data, &response); err != nil {
		return nil, err
	}
	return &response, nil
}
