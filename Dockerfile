# AURORA-X Multi-Stage Build (Rust + Go + Python)
# Hardened for production and security verified.

# -- Stage 1: Build Environment (Rust + Go) --
FROM rust:1.83-slim-bookworm AS builder

# Build-time dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv pkg-config curl git build-essential \
    && pip3 install --break-system-packages --no-cache-dir maturin \
    && rm -rf /var/lib/apt/lists/*

# Install Go 1.25.8 (latest secure)
COPY --from=golang:1.25.8-bookworm /usr/local/go/ /usr/local/go/
ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /build

# Build Rust extensions (aurora_core)
COPY aurora_core/ ./aurora_core/
RUN cd aurora_core && maturin build --release --out /wheels

# Build Go services
COPY services/ ./services/
RUN cd services && go mod tidy && go mod download && go mod verify && \
    CGO_ENABLED=0 go build -ldflags="-s -w" -o /app/aurora-services ./cmd/aurora-services

# Copy Python requirements for the final layer
COPY requirements.txt .
COPY aurora_x/ ./aurora_x/
COPY config/ ./config/
COPY dashboard/ ./dashboard/

# -- Stage 2: Production Runtime --
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="AURORA-X"
LABEL org.opencontainers.image.description="Autonomous Uncertainty-Resolved Optimization Platform"
LABEL org.opencontainers.image.vendor="Jitterx69"
LABEL org.opencontainers.image.source="https://github.com/Jitterx69/Aurora-X-v1.2.0"

WORKDIR /app

# System upgrades and runtime dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Rust wheels from builder
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Copy Go binary from builder
COPY --from=builder /app/aurora-services /usr/local/bin/aurora-services

# Copy Python code and config from builder (cleaner)
COPY --from=builder /build/aurora_x/ ./aurora_x/
COPY --from=builder /build/config/ ./config/
COPY --from=builder /build/dashboard/ ./dashboard/
COPY --from=builder /build/requirements.txt .

# Sync dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Security: Non-root user
RUN groupadd -r aurora && useradd -r -g aurora -d /app -s /sbin/nologin aurora

# Data persistence structure with permissions
RUN mkdir -p /app/data/wal /app/data/models /app/data/checkpoints && \
    chown -R aurora:aurora /app

USER aurora

EXPOSE 8080 8088 3000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Start services
CMD ["sh", "-c", "aurora-services & python3 -m aurora_x.main"]
