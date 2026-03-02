// Package gateway implements the AURORA-X Edge Gateway.
//
// High-performance sensor event validation, timestamp correction,
// CRC fingerprinting, and routing using goroutine-per-connection.
package gateway

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// SensorEvent represents a raw sensor data event.
type SensorEvent struct {
	AssetID   string             `json:"asset_id"`
	Timestamp float64            `json:"timestamp"`
	Sensors   map[string]float64 `json:"sensors"`
	Metadata  map[string]string  `json:"metadata,omitempty"`
}

// ValidatedEvent is an event that has passed all gateway checks.
type ValidatedEvent struct {
	SensorEvent
	Fingerprint string  `json:"fingerprint"`
	GatewayTS   float64 `json:"gateway_ts"`
	Valid       bool    `json:"valid"`
}

// Gateway performs edge validation and preprocessing.
type Gateway struct {
	logger           *zap.Logger
	maxTimestampDrift float64 // seconds
	eventsProcessed  atomic.Uint64
	eventsDropped    atomic.Uint64
	mu               sync.RWMutex
	requiredFields   []string
}

// NewGateway creates a new edge gateway.
func NewGateway(logger *zap.Logger, maxDriftMs float64) *Gateway {
	return &Gateway{
		logger:           logger,
		maxTimestampDrift: maxDriftMs / 1000.0,
		requiredFields:   []string{"asset_id", "timestamp", "sensors"},
	}
}

// Process validates and enriches a raw sensor event.
func (g *Gateway) Process(raw []byte) (*ValidatedEvent, error) {
	var event SensorEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		g.eventsDropped.Add(1)
		return nil, fmt.Errorf("json decode failed: %w", err)
	}

	// Schema validation
	if event.AssetID == "" {
		g.eventsDropped.Add(1)
		return nil, fmt.Errorf("missing asset_id")
	}
	if event.Sensors == nil || len(event.Sensors) == 0 {
		g.eventsDropped.Add(1)
		return nil, fmt.Errorf("missing or empty sensors")
	}

	// Timestamp correction
	now := float64(time.Now().UnixMilli()) / 1000.0
	if event.Timestamp <= 0 {
		event.Timestamp = now
	} else if drift := math.Abs(now - event.Timestamp); drift > g.maxTimestampDrift {
		g.logger.Warn("large timestamp drift, correcting",
			zap.String("asset", event.AssetID),
			zap.Float64("drift_s", drift))
		event.Timestamp = now
	}

	// NaN/Inf sanitization
	for sensor, val := range event.Sensors {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			event.Sensors[sensor] = 0.0
		}
	}

	// CRC fingerprint
	payload, _ := json.Marshal(event)
	fingerprint := fmt.Sprintf("%x", sha256.Sum256(payload))[:16]

	g.eventsProcessed.Add(1)

	return &ValidatedEvent{
		SensorEvent: event,
		Fingerprint: fingerprint,
		GatewayTS:   now,
		Valid:        true,
	}, nil
}

// Stats returns gateway throughput statistics.
func (g *Gateway) Stats() map[string]uint64 {
	return map[string]uint64{
		"processed": g.eventsProcessed.Load(),
		"dropped":   g.eventsDropped.Load(),
	}
}
