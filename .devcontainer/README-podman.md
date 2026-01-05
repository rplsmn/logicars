Podman notes for this devcontainer

This devcontainer is designed to be compatible with Podman (via Docker CLI compatibility).
It uses three named volumes for persistence:
  - logicars_cargo_target -> /home/positron/.cargo/target
  - logicars_cargo_registry -> /home/positron/.cargo/registry
  - logicars_rustup -> /home/positron/.rustup

Quick start (example):

  # build the image
  podman build -t logicars-dev -f .devcontainer/Dockerfile .

  # run the container (detached) and mount the workspace
  podman run -d --name logicars-devcontainer \
    -v logicars_cargo_target:/home/vscode/.cargo/target \
    -v logicars_cargo_registry:/home/vscode/.cargo/registry \
    -v logicars_rustup:/home/vscode/.rustup \
    -v "$(pwd)":/workspaces/logicars:Z \
    -w /workspaces/logicars \
    logicars-dev sleep infinity

  # generate a systemd unit so the container restarts on boot
  podman generate systemd --new --name logicars-devcontainer -f
  sudo systemctl enable container-logicars-devcontainer.service
  sudo systemctl start container-logicars-devcontainer.service

When opening in VS Code, point Remote-Containers (or the devcontainer extension) at this folder and choose "From Dockerfile"; if using the Podman socket, make sure your Docker-compatible CLI is configured to talk to Podman.
