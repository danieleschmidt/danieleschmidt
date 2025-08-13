# Multi-stage Dockerfile for Terragon SDLC Framework
# Production-optimized with security best practices

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install --no-deps .

# Production stage
FROM python:3.11-slim as production

# Set metadata labels
LABEL maintainer="Terragon Labs <noreply@terragon.ai>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="terragon-sdlc" \
      org.label-schema.description="Terragon-Optimized SDLC Implementation" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/danieleschmidt/terragon-sdlc" \
      org.label-schema.schema-version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/app/.local/bin:$PATH" \
    TERRAGON_ENVIRONMENT=production \
    TERRAGON_LOG_LEVEL=INFO

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Create application directories
RUN mkdir -p /app /app/data /app/logs /app/config \
    && chown -R app:app /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Switch to application user
USER app
WORKDIR /app

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app *.md *.yml *.yaml *.toml *.txt ./

# Create configuration file
RUN echo "# Terragon SDLC Configuration" > config/terragon.yml && \
    echo "environment: production" >> config/terragon.yml && \
    echo "logging:" >> config/terragon.yml && \
    echo "  level: INFO" >> config/terragon.yml && \
    echo "  format: json" >> config/terragon.yml && \
    echo "cache:" >> config/terragon.yml && \
    echo "  type: local" >> config/terragon.yml && \
    echo "  size: 1000" >> config/terragon.yml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import src.monitoring_framework; print('Health check passed')" || exit 1

# Expose ports
EXPOSE 8000 8080 9090

# Volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Use tini as init system
ENTRYPOINT ["tini", "--"]

# Default command
CMD ["python3", "-m", "src.monitoring_framework", "--serve", "--port", "8000"]

# Development stage (optional)
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    htop \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipython \
    jupyter

USER app

# Override for development
CMD ["python3", "-c", "print('Development container ready. Use: docker exec -it <container> /bin/bash')"]