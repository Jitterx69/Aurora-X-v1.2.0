# ────────────────────────────────────────────────────────────
# AURORA-X  ·  Multi-Stage Build  ·  Rust → Go → Python
# ────────────────────────────────────────────────────────────

# ── Stage 1: Rust Extension ──
FROM rust:1.78-slim AS rust-builder

RUN apt-get update && apt-get install -y python3-dev python3-pip && \
    pip3 install --break-system-packages maturin

WORKDIR /build/aurora_core
COPY aurora_core/ .
RUN maturin build --release --out /wheels

# ── Stage 2: Go Services ──
FROM golang:1.22-alpine AS go-builder

WORKDIR /build/services
COPY services/ .
RUN go mod download && \
    CGO_ENABLED=0 go build -ldflags="-s -w" -o /go-services ./cmd/aurora-services

# ── Stage 3: Python Runtime ──
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install Rust wheel
COPY --from=rust-builder /wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Copy Go binary
COPY --from=go-builder /go-services /usr/local/bin/aurora-services

# Copy Python source
COPY aurora_x/ ./aurora_x/
COPY config/ ./config/
COPY dashboard/ ./dashboard/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Data directories
RUN mkdir -p /app/data/wal /app/data/models /app/data/checkpoints

# Expose ports: API (8080), Go services (8088), Dashboard (3000)
EXPOSE 8080 8088 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Start Go services in background, then Python orchestrator
CMD aurora-services & python3 -m aurora_x.main
