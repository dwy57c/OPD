ARG COEVO_BASE_IMAGE=meinian-megatron-grpo:swift4.1.3-vllm0.19.1-trl0.29.1-mcore1.3.2@sha256:8c0641fd72cf18cc32ba439ed7a1fadf3fe6d5bf54cc42929c677d7cd63f4163
FROM ${COEVO_BASE_IMAGE}

ARG COEVO_EXPECTED_SWIFT_VERSION=4.1.3
ARG COEVO_TAU2_COMMIT=17e07b1da2bbc0cadfddeea36412686e0604127b
WORKDIR /opt/coevo

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/tau2-bench /opt/tau2-bench
COPY pyproject.toml README.md constraints-container.txt /opt/coevo/
COPY correctability_coevolution /opt/coevo/correctability_coevolution

RUN python -m pip install --no-cache-dir \
      --ignore-requires-python \
      --constraint /opt/coevo/constraints-container.txt \
      '/opt/tau2-bench[voice]' '/opt/coevo[dev]' \
    && test "$(python -c 'import swift; print(swift.__version__)')" = "$COEVO_EXPECTED_SWIFT_VERSION"

USER research

ENV TAU2_DATA_DIR=/opt/tau2-bench/data
ENV COEVO_TAU2_REVISION=${COEVO_TAU2_COMMIT}

WORKDIR /workspace/OPD/correctability_coevolution
