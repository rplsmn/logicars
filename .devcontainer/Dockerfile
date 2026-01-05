# Lightweight Rust-based devcontainer image
FROM rust:1.70-slim

# Install system deps commonly needed for Rust projects
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user matching typical VS Code remote users
ARG USERNAME=positron
ARG UID=1000
RUN useradd -m -u ${UID} ${USERNAME} || true
WORKDIR /home/${USERNAME}
USER ${USERNAME}

ENV CARGO_HOME=/home/${USERNAME}/.cargo
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV PATH=/home/${USERNAME}/.cargo/bin:${PATH}

# Ensure dirs exist (these are expected to be backed by named volumes)
RUN mkdir -p /home/${USERNAME}/.cargo/target /home/${USERNAME}/.cargo/registry /home/${USERNAME}/.rustup

CMD ["sleep", "infinity"]
