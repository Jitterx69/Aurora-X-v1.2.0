# AURORA-X Multi-Stage Build (Rust + Go + Python)

# rayon >= 1.11 requires Rust 1.80+
FROM rust:1.83-slim AS rust-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv pkg-config && \
    pip3 install --break-system-packages --no-cache-dir maturin && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build/aurora_core
COPY aurora_core/ .

ENV RUSTFLAGS=""
RUN maturin build --release --out /wheels

# -- Stage 2: Go Services --
FROM golang:1.22-alpine AS go-builder

WORKDIR /build/services
COPY services/ .
RUN go mod download && go mod verify && \
    CGO_ENABLED=0 go build -ldflags="-s -w" -o /go-services ./cmd/aurora-services

# -- Stage 3: Python Runtime --
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AURORA-X"
LABEL org.opencontainers.image.description="Autonomous Uncertainty-Resolved Optimization Platform"
LABEL org.opencontainers.image.vendor="Jitterx69"
LABEL org.opencontainers.image.source="https://github.com/Jitterx69/Aurora-X-v1.2.0"

WORKDIR /app

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

# Create non-root user
RUN groupadd -r aurora && useradd -r -g aurora -d /app -s /sbin/nologin aurora

# Data directories
RUN mkdir -p /app/data/wal /app/data/models /app/data/checkpoints && \
    chown -R aurora:aurora /app

USER aurora

EXPOSE 8080 8088 3000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

CMD ["sh", "-c", "aurora-services & python3 -m aurora_x.main"]
