package src

import (
	"encoding/json"
)

type MessageType string

const (
	MessageTypeRequest  MessageType = "REQUEST"
	MessageTypeResponse MessageType = "RESPONSE"
)

type UnifiedMessage struct {
	ID     string          `json:"id"`
	Type   MessageType     `json:"type"`
	Action ActionType      `json:"action"`
	Data   json.RawMessage `json:"data"`
	Status ResponseStatus  `json:"status,omitempty"`
}

func NewRequestMessage(id string, action ActionType, data interface{}) (*UnifiedMessage, error) {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}
	return &UnifiedMessage{
		ID:     id,
		Type:   MessageTypeRequest,
		Action: action,
		Data:   dataBytes,
	}, nil
}

func NewResponseMessage(id string, action ActionType, status ResponseStatus, data interface{}) (*UnifiedMessage, error) {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}
	return &UnifiedMessage{
		ID:     id,
		Type:   MessageTypeResponse,
		Action: action,
		Data:   dataBytes,
		Status: status,
	}, nil
}

func (m *UnifiedMessage) UnmarshalData(v interface{}) error {
	return json.Unmarshal(m.Data, v)
}
