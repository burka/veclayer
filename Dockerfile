# Stage 1: Build
FROM rust:1.93-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev protobuf-compiler libprotobuf-dev libstdc++-12-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
# Build with the `server` feature bundle (cli + store-sqlite + llm + http + auth).
# Default features omit `http`/`auth`; without `server` the HTTP `serve` command errors at runtime.
RUN cargo build --release --locked --features server

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/veclayer /usr/local/bin/veclayer

ENV VECLAYER_DATA_DIR=/data

# Security: fail-closed auth. The HTTP server refuses to bind a public address
# without authentication. Set VECLAYER_PASSPHRASE (or configure an identity/token
# via `veclayer identity` + `veclayer auth`) before starting.
# VECLAYER_ALLOW_INSECURE_BIND=1 overrides this for trusted networks (not recommended).
ENV VECLAYER_AUTH_REQUIRED=true

EXPOSE 8080

CMD ["veclayer", "serve", "--host", "0.0.0.0", "--port", "8080"]
