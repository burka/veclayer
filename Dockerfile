# Stage 1: Build
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev protobuf-compiler libprotobuf-dev libstdc++-12-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release --locked

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/veclayer /usr/local/bin/veclayer

ENV VECLAYER_DATA_DIR=/data
EXPOSE 8080

CMD ["veclayer", "serve", "--host", "0.0.0.0", "--port", "8080"]
