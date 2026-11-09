# Track 4 submission image (t4agent v2). Build for the platform:
#   docker buildx build --platform linux/amd64 -t <registry>/<repo>:<tag> --push .
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

# Required on every submission image (gate g0_integrity); must match the descriptor.
LABEL qfbench2.interface_version="2.0"

# The platform runs a non-root user with a read-only root filesystem and a small noexec /tmp:
# nothing may be written outside /output, so no bytecode caches and HOME points at /tmp.
# v21.7: a fixed hash seed keeps set iteration order, and so the prompts, the same on every run (Python randomises it).
# v21.11: lower-only caps on the unit clock (config.derive_clock, H5). The ingestion stage runs every unit in series
# within one 12-hour clock, so a 600 s card stops at 360 s (LLM work at 290 s), not 470 / 400 s; a shorter card's own
# clock still wins.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HOME=/tmp \
    PIP_NO_CACHE_DIR=1 \
    T4_CAP_TIME_BUDGET=360 \
    T4_CAP_LLM_DEADLINE=290

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m compileall -q /usr/local/lib/python3.13
COPY t4agent ./t4agent
COPY ARTIFACT_PROVENANCE.md ./
RUN python -m compileall -q /app/t4agent

# The harness appends: analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
# so `analyze` arrives as the first argument and the CLI accepts it as a leading positional.
ENTRYPOINT ["python", "-m", "t4agent"]
CMD ["analyze", "--help"]
