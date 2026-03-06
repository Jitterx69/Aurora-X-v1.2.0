# ────────────────────────────────────────────────────────────
# AURORA-X  ·  Multi-Stage Build  ·  Rust → Go → Python
# SECURITY: Non-root runtime, minimal attack surface
# ────────────────────────────────────────────────────────────

# ── Stage 1: Rust Extension ──
FROM rust:1.78-slim AS rust-builder

RUN apt-get update && apt-get install -y python3-dev python3-pip && \
    pip3 install --break-system-packages maturin && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build/aurora_core
COPY aurora_core/ .
RUN maturin build --release --out /wheels

# ── Stage 2: Go Services ──
FROM golang:1.22-alpine AS go-builder

WORKDIR /build/services
COPY services/ .
RUN go mod download && go mod verify && \
    CGO_ENABLED=0 go build -ldflags="-s -w" -o /go-services ./cmd/aurora-services

# ── Stage 3: Python Runtime ──
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AURORA-X"
LABEL org.opencontainers.image.description="Autonomous Uncertainty-Resolved Optimization Platform"
LABEL org.opencontainers.image.vendor="Jitterx69"
LABEL org.opencontainers.image.source="https://github.com/Jitterx69/Aurora-X-v1.2.0"

WORKDIR /app

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Rust wheel
COPY --from=rust-builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Copy Go binary
COPY --from=go-builder /go-services /usr/local/bin/aurora-services

# Copy Python source
COPY aurora_x/ ./aurora_x/
COPY config/ ./config/
COPY dashboard/ ./dashboard/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Create non-root user ──
RUN groupadd -r aurora && useradd -r -g aurora -d /app -s /sbin/nologin aurora

# Data directories (owned by aurora user)
RUN mkdir -p /app/data/wal /app/data/models /app/data/checkpoints && \
    chown -R aurora:aurora /app

# Drop to non-root user
USER aurora

# Expose ports: API (8080), Go services (8088), Dashboard (3000)
EXPOSE 8080 8088 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Start Go services in background, then Python orchestrator
CMD ["sh", "-c", "aurora-services & python3 -m aurora_x.main"]
