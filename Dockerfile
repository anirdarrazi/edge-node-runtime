ARG PYTHON_BASE_IMAGE=python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN python -m pip install --no-cache-dir .

CMD ["node-agent"]
