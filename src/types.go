package src

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

type ConnectionType string

const (
	EngineConnection    ConnectionType = "ENGINE"
	InferenceConnection ConnectionType = "INFERENCE"
)

type FramedConn struct {
	conn net.Conn
	r    *bufio.Reader
	w    *bufio.Writer
}

func NewFramedConn(conn net.Conn) *FramedConn {
	return &FramedConn{
		conn: conn,
		r:    bufio.NewReader(conn),
		w:    bufio.NewWriter(conn),
	}
}

func (fc *FramedConn) WriteMessage(msg *UnifiedMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	// Write message length as a 4-byte big-endian integer
	if err := binary.Write(fc.w, binary.BigEndian, uint32(len(data))); err != nil {
		return err
	}

	// Write message data
	if _, err := fc.w.Write(data); err != nil {
		return err
	}

	return fc.w.Flush()
}

func (fc *FramedConn) ReadMessage() (*UnifiedMessage, error) {
	// Read message length
	var length uint32
	if err := binary.Read(fc.r, binary.BigEndian, &length); err != nil {
		return nil, err
	}

	// Read message data
	data := make([]byte, length)
	if _, err := io.ReadFull(fc.r, data); err != nil {
		return nil, err
	}

	var msg UnifiedMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, err
	}

	return &msg, nil
}

type Connection struct {
	framed   *FramedConn
	connType ConnectionType
}

func NewConnection(framed *FramedConn, connType ConnectionType) *Connection {
	return &Connection{
		framed:   framed,
		connType: connType,
	}
}

func (c *Connection) Send(msg *UnifiedMessage) error {
	return c.framed.WriteMessage(msg)
}

func (c *Connection) Receive() (*UnifiedMessage, error) {
	return c.framed.ReadMessage()
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
			cs.engineConn.framed.conn.Close()
		}
		cs.engineConn = conn
	case InferenceConnection:
		if cs.inferenceConn != nil {
			cs.inferenceConn.framed.conn.Close()
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
			cs.engineConn.framed.conn.Close()
			cs.engineConn = nil
		}
	case InferenceConnection:
		if cs.inferenceConn != nil {
			cs.inferenceConn.framed.conn.Close()
			cs.inferenceConn = nil
		}
	}
}

func (cs *ConnectionState) SendBlocking(ctx context.Context, connType ConnectionType, msg *UnifiedMessage) error {
	for {
		if err := cs.WaitForConnection(ctx, connType, 0); err != nil {
			return fmt.Errorf("failed waiting for connection: %w", err)
		}

		cs.mu.RLock()
		var conn *Connection
		switch connType {
		case EngineConnection:
			conn = cs.engineConn
		case InferenceConnection:
			conn = cs.inferenceConn
		}
		cs.mu.RUnlock()

		if err := conn.Send(msg); err != nil {
			continue
		}

		return nil
	}
}

func (cs *ConnectionState) ReceiveBlocking(ctx context.Context, connType ConnectionType) (*UnifiedMessage, error) {
	for {
		if err := cs.WaitForConnection(ctx, connType, 0); err != nil {
			return nil, fmt.Errorf("failed waiting for connection: %w", err)
		}

		cs.mu.RLock()
		var conn *Connection
		switch connType {
		case EngineConnection:
			conn = cs.engineConn
		case InferenceConnection:
			conn = cs.inferenceConn
		}
		cs.mu.RUnlock()

		msg, err := conn.Receive()
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue
			}
			return nil, err
		}

		return msg, nil
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

type ActionType string
type ResponseType string
type ResponseStatus string

const (
	ActionTypeInitialConnection    ActionType = "INITIAL_CONNECTION"
	ActionTypeLoadCheckpoint       ActionType = "LOAD_CHECKPOINT"
	ActionTypeVolumeAttach         ActionType = "VOLUME_ATTACH"
	ActionTypeVolumeDetach         ActionType = "VOLUME_DETACH"
	ActionTypeInferenceStream      ActionType = "INFERENCE_STREAM"
	ActionTypeInferenceBatchStream ActionType = "INFERENCE_BATCH_STREAM"
	ActionTypeLoadModel            ActionType = "LOAD_MODEL"
	ActionTypeError                ActionType = "ERROR"

	ResponseTypeFinal  ResponseType = "FINAL"
	ResponseTypeStream ResponseType = "STREAM"
	ResponseTypeEmpty  ResponseType = "EMPTY"

	ResponseStatusSuccess ResponseStatus = "SUCCESS"
	ResponseStatusError   ResponseStatus = "ERROR"
)

// Base interfaces remain the same
type MindletRequest interface {
	MindletRequestType() ActionType
}

type MindletResponse interface {
	MindletResponseType() ActionType
}

type InitialConnectionMessage struct {
	ConnectionType ConnectionType `json:"connection_type"`
	ClientID       string         `json:"client_id"`
}

// Request/Response wrappers remain the same
type WrappedRequest struct {
	RequestID string          `json:"request_id"`
	Type      ActionType      `json:"type"`
	Data      json.RawMessage `json:"data"`
}

type WrappedResponse struct {
	RequestID string          `json:"request_id"`
	Type      ActionType      `json:"type"`
	Data      json.RawMessage `json:"data"`
}

// Internal request wrapper
type Request struct {
	RequestID    string                 `json:"request_id"`
	Type         ActionType             `json:"type"`
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

func (r *LoadCheckpointRequest) MindletRequestType() ActionType {
	return ActionTypeLoadCheckpoint
}

type LoadCheckpointResponse struct {
	Type    ResponseType   `json:"type"`
	Status  ResponseStatus `json:"status"`
	Message string         `json:"message"`
}

func (r *LoadCheckpointResponse) MindletResponseType() ActionType {
	return ActionTypeLoadCheckpoint
}

type LoadModelRequest struct {
	ModelName string `json:"model_name"`
}

func (r *LoadModelRequest) MindletRequestType() ActionType {
	return ActionTypeLoadModel
}

type LoadModelResponse struct {
	Message string `json:"message"`
}

func (r *LoadModelResponse) MindletResponseType() ActionType {
	return ActionTypeLoadModel
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

func (r *VolumeAttachRequest) MindletRequestType() ActionType {
	return ActionTypeVolumeAttach
}

type VolumeAttachResponse struct {
	Status  ResponseStatus `json:"status"`
	Message string         `json:"message"`
}

func (r *VolumeAttachResponse) MindletResponseType() ActionType {
	return ActionTypeVolumeAttach
}

type VolumeDetachRequest struct {
	Path string `json:"path"`
}

func (r *VolumeDetachRequest) MindletRequestType() ActionType {
	return ActionTypeVolumeDetach
}

type VolumeDetachResponse struct {
	Status  ResponseStatus `json:"status"`
	Message string         `json:"message"`
}

func (r *VolumeDetachResponse) MindletResponseType() ActionType {
	return ActionTypeVolumeDetach
}

type InferenceStreamRequest struct {
	ModelID string `json:"model_id"`
	Input   string `json:"input"`
}

func (r *InferenceStreamRequest) MindletRequestType() ActionType {
	return ActionTypeInferenceStream
}

type InferenceStreamResponse struct {
	Type   ResponseType   `json:"type"`
	Text   string         `json:"text,omitempty"`
	Status ResponseStatus `json:"status"`
}

func (r *InferenceStreamResponse) MindletResponseType() ActionType {
	return ActionTypeInferenceStream
}

type InferenceBatchStreamRequest struct {
	BatchID  string                    `json:"batch_id"`
	Requests []*InferenceStreamRequest `json:"requests"`
}

func (r *InferenceBatchStreamRequest) MindletRequestType() ActionType {
	return ActionTypeInferenceBatchStream
}

type InferenceBatchStreamResponse struct {
	BatchID string         `json:"batch_id"`
	Type    []ResponseType `json:"type"`
	Outputs []string       `json:"text"`
	Status  ResponseStatus `json:"status"`
}

func (r *InferenceBatchStreamResponse) MindletResponseType() ActionType {
	return ActionTypeInferenceBatchStream
}
