#!/bin/bash
# Terragon SDLC Framework Deployment Script
# Production deployment with comprehensive validation and rollback

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-staging}"
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-docker.io/terragon}"
IMAGE_NAME="${REGISTRY}/sdlc-framework"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        if [ "${ROLLBACK_ON_FAILURE:-true}" = "true" ]; then
            log_info "Initiating rollback..."
            rollback_deployment
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Validation functions
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check environment file
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        log_warning ".env file not found, using defaults"
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    fi
    
    # Validate Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

validate_configuration() {
    log_info "Validating configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Validate Python syntax
    if ! python3 validate_sdlc.py; then
        log_error "Code validation failed"
        exit 1
    fi
    
    # Validate Docker Compose
    if ! docker-compose config &> /dev/null; then
        log_error "Docker Compose configuration is invalid"
        exit 1
    fi
    
    log_success "Configuration validation passed"
}

pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check disk space (need at least 5GB)
    local available_space
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5242880 ]; then
        log_error "Insufficient disk space. Need at least 5GB available"
        exit 1
    fi
    
    # Check memory (need at least 2GB)
    local available_memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    if [ "$available_memory" -lt 2048 ]; then
        log_warning "Low available memory: ${available_memory}MB. Recommended: 2GB+"
    fi
    
    # Check port availability
    local ports=("80" "443" "5432" "6379" "8000" "8080" "9090" "3000")
    for port in "${ports[@]}"; do
        if ss -ln | grep -q ":$port "; then
            log_warning "Port $port is already in use"
        fi
    done
    
    log_success "Pre-deployment checks passed"
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Set build arguments
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    # Build production image
    if ! docker build \
        --target production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        --tag "${IMAGE_NAME}:${VERSION}" \
        --tag "${IMAGE_NAME}:latest" \
        .; then
        log_error "Failed to build Docker image"
        exit 1
    fi
    
    log_success "Docker images built successfully"
}

run_tests() {
    log_info "Running test suite..."
    
    # Run tests in isolated container
    if ! docker run --rm \
        -v "${PROJECT_ROOT}:/app" \
        -w /app \
        "${IMAGE_NAME}:${VERSION}" \
        python3 validate_sdlc.py; then
        log_error "Tests failed"
        exit 1
    fi
    
    log_success "All tests passed"
}

backup_current_deployment() {
    if [ "${DEPLOYMENT_ENV}" = "production" ]; then
        log_info "Creating backup of current deployment..."
        
        # Backup database
        if docker-compose ps postgres | grep -q "Up"; then
            docker-compose exec -T postgres pg_dump -U terragon terragon_sdlc > \
                "/tmp/terragon_backup_$(date +%Y%m%d_%H%M%S).sql"
        fi
        
        # Backup volumes
        docker run --rm \
            -v terragon_data:/source:ro \
            -v "/tmp:/backup" \
            alpine tar czf "/backup/terragon_data_$(date +%Y%m%d_%H%M%S).tar.gz" -C /source .
        
        log_success "Backup completed"
    fi
}

deploy_application() {
    log_info "Deploying application..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export VERSION
    export DEPLOYMENT_ENV
    
    # Pull latest images (if using registry)
    if [ "${USE_REGISTRY:-false}" = "true" ]; then
        docker-compose pull
    fi
    
    # Deploy with rolling update strategy
    if [ "${DEPLOYMENT_ENV}" = "production" ]; then
        # Production deployment with zero-downtime
        docker-compose up -d --no-deps --scale terragon-sdlc=2 terragon-sdlc
        sleep 30  # Wait for new instance to be healthy
        docker-compose up -d --no-deps --scale terragon-sdlc=1 terragon-sdlc
    else
        # Development/staging deployment
        docker-compose up -d
    fi
    
    log_success "Application deployed"
}

health_checks() {
    log_info "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8000/health &> /dev/null; then
            log_success "Application is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for application to be ready..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test API endpoints
    local endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8080/metrics"
        "http://localhost:9090/api/v1/label/job/values"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if ! curl -f -s "$endpoint" &> /dev/null; then
            log_error "Smoke test failed for endpoint: $endpoint"
            return 1
        fi
    done
    
    log_success "Smoke tests passed"
}

post_deployment_tasks() {
    log_info "Running post-deployment tasks..."
    
    # Database migrations (if any)
    if [ -f "${PROJECT_ROOT}/database/migrations.sql" ]; then
        docker-compose exec -T postgres psql -U terragon -d terragon_sdlc \
            -f /docker-entrypoint-initdb.d/migrations.sql
    fi
    
    # Clear caches
    docker-compose exec redis redis-cli FLUSHDB
    
    # Update monitoring configuration
    if docker-compose ps prometheus | grep -q "Up"; then
        docker-compose exec prometheus promtool config reload || true
    fi
    
    log_success "Post-deployment tasks completed"
}

rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Stop current deployment
    docker-compose down
    
    # Restore from backup if available
    local latest_backup
    latest_backup=$(ls -t /tmp/terragon_backup_*.sql 2>/dev/null | head -n1)
    
    if [ -n "$latest_backup" ]; then
        log_info "Restoring database from backup: $latest_backup"
        docker-compose up -d postgres
        sleep 10
        docker-compose exec -T postgres psql -U terragon -d terragon_sdlc < "$latest_backup"
    fi
    
    # Restart with previous version
    export VERSION="previous"
    docker-compose up -d
    
    log_warning "Rollback completed"
}

show_status() {
    log_info "Deployment Status:"
    echo "==================="
    docker-compose ps
    echo ""
    
    log_info "Application Logs (last 20 lines):"
    echo "=================================="
    docker-compose logs --tail=20 terragon-sdlc
}

main() {
    log_info "Starting Terragon SDLC Framework deployment"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Version: $VERSION"
    echo ""
    
    # Deployment pipeline
    validate_environment
    validate_configuration
    pre_deployment_checks
    build_images
    run_tests
    
    if [ "${DEPLOYMENT_ENV}" = "production" ]; then
        backup_current_deployment
    fi
    
    deploy_application
    
    # Wait for deployment to stabilize
    sleep 30
    
    health_checks
    smoke_tests
    post_deployment_tasks
    
    show_status
    
    log_success "🎉 Deployment completed successfully!"
    log_info "Application is available at: http://localhost:8000"
    log_info "Monitoring dashboard: http://localhost:3000"
    log_info "Metrics endpoint: http://localhost:9090"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --rollback)
            rollback_deployment
            exit 0
            ;;
        --status)
            show_status
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env ENV        Set deployment environment (staging|production)"
            echo "  --version VER    Set version to deploy"
            echo "  --rollback       Rollback to previous deployment"
            echo "  --status         Show current deployment status"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main deployment
main