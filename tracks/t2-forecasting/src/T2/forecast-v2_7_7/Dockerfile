FROM python:3.13-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.16@sha256:adc68cd785ca65ea25c0611043b0a00b4ea3a22e1b54102fc084406d888082ee /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt ./
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-deps -r requirements.txt

FROM python:3.13-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e
LABEL qfbench2.interface_version="2.0"
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY forecast.py ./
COPY ARTIFACT_PROVENANCE.md ./
RUN chmod 755 /app/forecast.py && ln -sf /app/forecast.py /opt/venv/bin/forecast
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1 \
    XDG_CACHE_HOME=/tmp PYTHONHASHSEED=0
USER 65534:65534
CMD ["forecast", "--help"]
