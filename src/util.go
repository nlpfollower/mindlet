package src

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"time"
)

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
