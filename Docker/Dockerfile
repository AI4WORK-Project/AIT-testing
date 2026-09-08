# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.22

FROM ${UV_IMAGE} AS uv-bin

FROM ${PYTHON_IMAGE} AS base
COPY --from=uv-bin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/wisdom/.venv/bin:$PATH"
WORKDIR /opt/wisdom

# Resolve dependencies before copying source so source-only edits reuse this layer.
COPY pyproject.toml uv.lock README.md setup.py ./
RUN uv sync --frozen --no-install-project --no-dev

COPY wisdom ./wisdom
COPY models_info ./models_info
COPY wisdom_classification_train.py wisdom_yolo_train.py wisdom_pose_train.py ./
COPY run_wisdom.py convert_torchvision_dataset.py ./
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "--frozen", "python", "run_wisdom.py", "--help"]

FROM base AS test

# The locked Torch build is CUDA-capable, but validation is intentionally CPU-only.
ENV CUDA_VISIBLE_DEVICES=""
COPY tests ./tests
RUN uv sync --frozen --group test --extra bo

CMD ["uv", "run", "--offline", "--no-sync", "pytest", "-q", "tests/test_wisdom_e2e.py"]
