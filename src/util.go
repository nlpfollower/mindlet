package src

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"
)

func isTCPPortOpen(port int) bool {
	addr := fmt.Sprintf("localhost:%d", port)
	conn, err := net.DialTimeout("tcp", addr, time.Second)
	if err != nil {
		return false
	}
	defer conn.Close()
	return true
}

func GenerateUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func getMaxCheckpoint(modelType ModelType) int {
	switch modelType {
	case Model8B:
		return 8
	case Model70B:
		return 32
	case Model405B:
		return 64
	default:
		return 8
	}
}

func isCheckpointFile(filename string) bool {
	return strings.HasPrefix(filename, "model-") && strings.Contains(filename, "-of-") && strings.HasSuffix(filename, ".safetensors")
}

func isServerRunning(port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("localhost:%d", port), time.Second)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func isServerAvailable(port int) bool {
	resp, err := http.Get(fmt.Sprintf("http://localhost:%d/status", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false
	}

	var status struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return false
	}

	return status.Status == "Available"
}
