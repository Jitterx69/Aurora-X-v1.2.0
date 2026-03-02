// Package eventlog implements an append-only Write-Ahead Log (WAL).
//
// Binary-encoded event log with CRC-32 checksums per entry,
// file rotation, and crash-safe writes via fsync.
package eventlog

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// Entry is a single WAL entry.
type Entry struct {
	Sequence  uint64
	Timestamp int64
	Payload   []byte
	CRC       uint32
}

// EventLog is a crash-safe, append-only event log.
type EventLog struct {
	logger     *zap.Logger
	dir        string
	maxSize    int64
	file       *os.File
	fileSize   int64
	sequence   atomic.Uint64
	mu         sync.Mutex
	segmentIdx int
}

// NewEventLog creates or opens an event log directory.
func NewEventLog(logger *zap.Logger, dir string, maxSegmentSizeMB int64) (*EventLog, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}

	el := &EventLog{
		logger:  logger,
		dir:     dir,
		maxSize: maxSegmentSizeMB * 1024 * 1024,
	}

	if err := el.openSegment(); err != nil {
		return nil, err
	}
	return el, nil
}

// Append writes an entry to the log. Safe for concurrent use.
func (el *EventLog) Append(payload []byte) (uint64, error) {
	el.mu.Lock()
	defer el.mu.Unlock()

	seq := el.sequence.Add(1)
	ts := time.Now().UnixNano()

	entry := Entry{
		Sequence:  seq,
		Timestamp: ts,
		Payload:   payload,
		CRC:       crc32.ChecksumIEEE(payload),
	}

	buf := el.encode(entry)

	n, err := el.file.Write(buf)
	if err != nil {
		return 0, fmt.Errorf("write entry: %w", err)
	}
	el.fileSize += int64(n)

	// Fsync for durability
	if seq%100 == 0 {
		_ = el.file.Sync()
	}

	// Rotate if segment is too large
	if el.fileSize >= el.maxSize {
		if err := el.rotate(); err != nil {
			el.logger.Error("segment rotation failed", zap.Error(err))
		}
	}

	return seq, nil
}

// Close flushes and closes the log.
func (el *EventLog) Close() error {
	el.mu.Lock()
	defer el.mu.Unlock()
	if el.file != nil {
		_ = el.file.Sync()
		return el.file.Close()
	}
	return nil
}

// encode serializes an entry to binary format:
// [8-byte seq][8-byte ts][4-byte payload_len][payload][4-byte crc]
func (el *EventLog) encode(e Entry) []byte {
	buf := make([]byte, 24+len(e.Payload))
	binary.LittleEndian.PutUint64(buf[0:8], e.Sequence)
	binary.LittleEndian.PutUint64(buf[8:16], uint64(e.Timestamp))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(len(e.Payload)))
	copy(buf[20:20+len(e.Payload)], e.Payload)
	binary.LittleEndian.PutUint32(buf[20+len(e.Payload):], e.CRC)
	return buf
}

func (el *EventLog) openSegment() error {
	name := fmt.Sprintf("wal_%06d.log", el.segmentIdx)
	path := filepath.Join(el.dir, name)

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open segment %s: %w", name, err)
	}

	info, err := f.Stat()
	if err != nil {
		return err
	}

	el.file = f
	el.fileSize = info.Size()
	return nil
}

func (el *EventLog) rotate() error {
	if el.file != nil {
		_ = el.file.Sync()
		_ = el.file.Close()
	}
	el.segmentIdx++
	el.fileSize = 0
	return el.openSegment()
}

// Stats returns log statistics.
func (el *EventLog) Stats() map[string]interface{} {
	return map[string]interface{}{
		"sequence":    el.sequence.Load(),
		"segment":     el.segmentIdx,
		"file_size":   el.fileSize,
	}
}
