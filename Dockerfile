# Build Stage
FROM rust:1.81 as builder
WORKDIR /usr/src/awesomaze
COPY . .
RUN cargo build --release

# Runtime Stage
FROM nvidia/cuda:12.6.2-base-ubuntu24.04
RUN apt-get update && apt-get install -y libssl-dev libstdc++6 libgcc1 && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
  && apt-get install -y \
  libxext6 \
  libvulkan1 \
  libvulkan-dev \
  vulkan-tools
COPY --from=builder /usr/src/awesomaze/target/release/awesomaze /usr/local/bin/awesomaze

ENTRYPOINT ["awesomaze"]

