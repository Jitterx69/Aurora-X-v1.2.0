// AURORA-X Go Services — Main Entry Point
//
// Starts the edge gateway, message broker, event log, and API server.
// Communicates with the Python orchestrator via gRPC/Unix domain socket.
package main

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"go.uber.org/zap"

	"aurora-x/services/internal/api"
	"aurora-x/services/internal/broker"
	"aurora-x/services/internal/eventlog"
	"aurora-x/services/internal/gateway"

	"context"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"time"

	"google.golang.org/grpc"
)

func initTracer(ctx context.Context, logger *zap.Logger) (func(context.Context) error, error) {
	endpoint := os.Getenv("JAEGER_OTLP_ENDPOINT")
	if endpoint == "" {
		endpoint = "jaeger:4317"
	}

	logger.Info("Initializing OTLP Tracer", zap.String("endpoint", endpoint))

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String("aurora-x-go-services"),
			semconv.ServiceVersionKey.String("0.1.0"),
			semconv.DeploymentEnvironmentKey.String(os.Getenv("AURORA_MODE")),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithDialOption(grpc.WithBlock()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace exporter: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

	return tp.Shutdown, nil
}

func main() {
	// Structured logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if os.Getenv("AURORA_TRACING_ENABLED") != "false" {
		shutdown, err := initTracer(ctx, logger)
		if err != nil {
			logger.Error("failed to initialize tracer", zap.Error(err))
		} else {
			defer func() {
				if err := shutdown(context.Background()); err != nil {
					logger.Error("failed to shutdown tracer", zap.Error(err))
				}
			}()
		}
	}

	logger.Info("AURORA-X Go Services starting",
		zap.String("version", "0.1.0"))

	// ── Initialize components ──
	gw := gateway.NewGateway(logger, 5000)

	brk := broker.NewBroker(logger, broker.BrokerConfig{
		DefaultRingSize:   8192,
		DefaultBufferSize: 512,
	})

	wal, err := eventlog.NewEventLog(logger, "data/wal", 64)
	if err != nil {
		logger.Fatal("failed to create event log", zap.Error(err))
	}
	defer wal.Close()

	apiSrv := api.NewServer(logger, 8088)

	// ── Start API server ──
	go func() {
		if err := apiSrv.Start(); err != nil {
			logger.Error("API server error", zap.Error(err))
		}
	}()

	// ── Start Unix socket listener for Python IPC ──
	socketPath := "/tmp/aurora_go.sock"
	os.Remove(socketPath) // clean up stale socket

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		logger.Fatal("failed to listen on unix socket", zap.Error(err))
	}
	defer listener.Close()
	defer os.Remove(socketPath)

	logger.Info("Unix socket ready", zap.String("path", socketPath))

	// ── Pipeline: socket → gateway → broker → WAL → API broadcast ──
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				logger.Error("accept error", zap.Error(err))
				continue
			}

			go handleConnection(conn, gw, brk, wal, apiSrv, logger)
		}
	}()

	// ── Graceful shutdown ──
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh

	logger.Info("shutting down", zap.String("signal", sig.String()))

	// Print final stats
	gwStats := gw.Stats()
	walStats := wal.Stats()
	brkStats := brk.Stats()
	logger.Info("final gateway stats",
		zap.Uint64("processed", gwStats["processed"]),
		zap.Uint64("dropped", gwStats["dropped"]))
	logger.Info("final WAL stats", zap.Any("stats", walStats))
	logger.Info("final broker stats", zap.Any("stats", brkStats))
}

func handleConnection(
	conn net.Conn,
	gw *gateway.Gateway,
	brk *broker.Broker,
	wal *eventlog.EventLog,
	apiSrv *api.Server,
	logger *zap.Logger,
) {
	defer conn.Close()

	buf := make([]byte, 65536)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			return // connection closed
		}

		raw := buf[:n]

		// 1) Gateway validation
		event, err := gw.Process(raw)
		if err != nil {
			logger.Debug("event dropped", zap.Error(err))
			continue
		}

		// 2) Broker publish
		payload, _ := json.Marshal(event)
		brk.Publish(fmt.Sprintf("sensor.%s", event.AssetID), payload)

		// 3) WAL append
		wal.Append(payload)

		// 4) Broadcast to WebSocket clients
		apiSrv.Broadcast(payload)
	}
}
