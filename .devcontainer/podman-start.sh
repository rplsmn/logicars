#!/bin/sh
# Helper script: build image, run container with named volumes, and (optionally) generate a systemd unit.

IMAGE_NAME=logicars-dev
CONTAINER_NAME=logicars-devcontainer

podman build -t ${IMAGE_NAME} -f .devcontainer/Dockerfile .

podman run -d --name ${CONTAINER_NAME} \
  -v logicars_cargo_target:/home/positron/.cargo/target \
  -v logicars_cargo_registry:/home/positron/.cargo/registry \
  -v logicars_rustup:/home/positron/.rustup \
  -v "$(pwd)":/workspaces/logicars:Z \
  -w /workspaces/logicars \
  ${IMAGE_NAME} sleep infinity

# Uncomment to auto-generate systemd unit and enable it
# podman generate systemd --new --name ${CONTAINER_NAME} -f
# sudo systemctl enable container-${CONTAINER_NAME}.service
# sudo systemctl start container-${CONTAINER_NAME}.service
