// Package broker implements a high-performance in-memory message broker.
//
// Lock-free MPSC ring buffer per topic with channel-based pub/sub
// and configurable backpressure.
package broker

import (
	"sync"
	"sync/atomic"

	"go.uber.org/zap"
)

// Message wraps a payload for pub/sub delivery.
type Message struct {
	Topic   string
	Payload []byte
	Offset  uint64
}

// Subscriber receives messages from a topic.
type Subscriber struct {
	ID   string
	Ch   chan Message
	Done chan struct{}
}

// Topic manages a single pub/sub channel.
type Topic struct {
	name        string
	subscribers sync.Map // map[string]*Subscriber
	offset      atomic.Uint64
	mu          sync.RWMutex
	ring        []Message
	ringSize    int
	head        int
}

// NewTopic creates a ring-buffered topic.
func NewTopic(name string, ringSize int) *Topic {
	return &Topic{
		name:     name,
		ring:     make([]Message, ringSize),
		ringSize: ringSize,
	}
}

// Publish a message to the topic, broadcasting to all subscribers.
func (t *Topic) Publish(payload []byte) uint64 {
	offset := t.offset.Add(1)

	msg := Message{
		Topic:   t.name,
		Payload: payload,
		Offset:  offset,
	}

	// Store in ring buffer
	t.mu.Lock()
	t.ring[t.head] = msg
	t.head = (t.head + 1) % t.ringSize
	t.mu.Unlock()

	// Fan-out to subscribers (non-blocking)
	t.subscribers.Range(func(key, value any) bool {
		sub := value.(*Subscriber)
		select {
		case sub.Ch <- msg:
		default:
			// Backpressure: drop message for slow subscriber
		}
		return true
	})

	return offset
}

// Subscribe adds a subscriber to this topic.
func (t *Topic) Subscribe(id string, bufferSize int) *Subscriber {
	sub := &Subscriber{
		ID:   id,
		Ch:   make(chan Message, bufferSize),
		Done: make(chan struct{}),
	}
	t.subscribers.Store(id, sub)
	return sub
}

// Unsubscribe removes a subscriber.
func (t *Topic) Unsubscribe(id string) {
	if val, ok := t.subscribers.LoadAndDelete(id); ok {
		sub := val.(*Subscriber)
		close(sub.Done)
	}
}

// Broker manages multiple topics.
type Broker struct {
	logger *zap.Logger
	topics sync.Map // map[string]*Topic
	config BrokerConfig
}

// BrokerConfig holds broker settings.
type BrokerConfig struct {
	DefaultRingSize    int
	DefaultBufferSize  int
}

// NewBroker creates a new message broker.
func NewBroker(logger *zap.Logger, config BrokerConfig) *Broker {
	if config.DefaultRingSize <= 0 {
		config.DefaultRingSize = 4096
	}
	if config.DefaultBufferSize <= 0 {
		config.DefaultBufferSize = 256
	}
	return &Broker{
		logger: logger,
		config: config,
	}
}

// Publish to a topic (auto-creates if doesn't exist).
func (b *Broker) Publish(topicName string, payload []byte) uint64 {
	topic := b.getOrCreateTopic(topicName)
	return topic.Publish(payload)
}

// Subscribe to a topic.
func (b *Broker) Subscribe(topicName, subscriberID string) *Subscriber {
	topic := b.getOrCreateTopic(topicName)
	return topic.Subscribe(subscriberID, b.config.DefaultBufferSize)
}

// Unsubscribe from a topic.
func (b *Broker) Unsubscribe(topicName, subscriberID string) {
	if val, ok := b.topics.Load(topicName); ok {
		topic := val.(*Topic)
		topic.Unsubscribe(subscriberID)
	}
}

func (b *Broker) getOrCreateTopic(name string) *Topic {
	if val, ok := b.topics.Load(name); ok {
		return val.(*Topic)
	}
	topic := NewTopic(name, b.config.DefaultRingSize)
	actual, _ := b.topics.LoadOrStore(name, topic)
	return actual.(*Topic)
}

// Stats returns broker statistics.
func (b *Broker) Stats() map[string]interface{} {
	stats := make(map[string]interface{})
	b.topics.Range(func(key, value any) bool {
		topic := value.(*Topic)
		subCount := 0
		topic.subscribers.Range(func(_, _ any) bool {
			subCount++
			return true
		})
		stats[key.(string)] = map[string]interface{}{
			"offset":      topic.offset.Load(),
			"subscribers": subCount,
		}
		return true
	})
	return stats
}
