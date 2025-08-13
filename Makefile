# Terragon SDLC Framework Makefile
# Production-grade automation for development, testing, and deployment

.PHONY: help install install-dev clean test test-unit test-integration test-performance
.PHONY: lint format security quality-gates coverage docs build publish
.PHONY: docker-build docker-test docker-push benchmark profile optimize
.PHONY: deploy-staging deploy-prod monitoring setup-ci teardown

# Configuration
PYTHON := python3
PIP := pip3
PROJECT_NAME := terragon-sdlc
VERSION := $(shell grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
DOCKER_IMAGE := terragon/sdlc-framework
DOCKER_TAG := $(VERSION)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Terragon SDLC Framework - Make Commands$(NC)"
	@echo "========================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install package for production
	@echo "$(BLUE)Installing Terragon SDLC Framework...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -e .[dev,docs,ml,quantum,cloud]
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

clean: ## Clean up build artifacts and cache
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf docs/_build/ coverage.xml junit.xml
	@echo "$(GREEN)Cleanup complete!$(NC)"

# Testing
test: test-unit test-integration ## Run all tests
	@echo "$(GREEN)All tests completed!$(NC)"

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v --cov=src --cov-append

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	pytest tests/performance/ -v --benchmark-only

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	pytest tests/e2e/ -v

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	pytest tests/ -n auto --cov=src

# Code Quality
lint: ## Run all linting tools
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 src tests
	pylint src
	mypy src
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src tests
	isort src tests
	@echo "$(GREEN)Code formatted!$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check src tests
	isort --check-only src tests

# Security
security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	bandit -r src
	safety check
	$(PYTHON) -m src.security_scanner --scan-types all
	@echo "$(GREEN)Security scan complete!$(NC)"

# Quality Gates
quality-gates: ## Run all quality gates
	@echo "$(BLUE)Running quality gates...$(NC)"
	$(PYTHON) -m src.quality_gates --gates all --project-root .
	@echo "$(GREEN)Quality gates passed!$(NC)"

# Coverage
coverage: ## Generate test coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=xml --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

coverage-badge: ## Generate coverage badge
	coverage-badge -o coverage.svg

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Documentation built in docs/_build/html/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation on http://localhost:8000$(NC)"
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && make clean

# Build and Package
build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete! Packages in dist/$(NC)"

publish: build ## Publish to PyPI
	@echo "$(BLUE)Publishing to PyPI...$(NC)"
	twine upload dist/*
	@echo "$(GREEN)Published to PyPI!$(NC)"

publish-test: build ## Publish to TestPyPI
	@echo "$(BLUE)Publishing to TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*
	@echo "$(GREEN)Published to TestPyPI!$(NC)"

# Docker
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):latest
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-test: ## Test Docker image
	@echo "$(BLUE)Testing Docker image...$(NC)"
	docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) pytest tests/ -v
	@echo "$(GREEN)Docker image tests passed!$(NC)"

docker-push: docker-build ## Push Docker image
	@echo "$(BLUE)Pushing Docker image...$(NC)"
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):latest
	@echo "$(GREEN)Docker image pushed!$(NC)"

docker-run: ## Run Docker container
	docker run -it --rm -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

# Performance and Optimization
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m src.performance_optimizer --benchmark
	@echo "$(GREEN)Benchmarks complete!$(NC)"

profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.prof -m src.backlog_manager --metrics
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative'); p.print_stats(20)"
	@echo "$(GREEN)Profiling complete!$(NC)"

optimize: ## Run optimization analysis
	@echo "$(BLUE)Running optimization analysis...$(NC)"
	$(PYTHON) -m src.performance_optimizer --analyze --report optimization_report.json
	@echo "$(GREEN)Optimization analysis complete! Report: optimization_report.json$(NC)"

# Monitoring and Analytics
monitoring: ## Setup monitoring and metrics collection
	@echo "$(BLUE)Setting up monitoring...$(NC)"
	$(PYTHON) -m src.monitoring_framework --setup
	@echo "$(GREEN)Monitoring setup complete!$(NC)"

metrics: ## Generate metrics report
	@echo "$(BLUE)Generating metrics report...$(NC)"
	$(PYTHON) -m src.backlog_manager --metrics
	$(PYTHON) -m src.wsjf_engine --analyze-backlog backlog.yml --export-report metrics_report.json
	@echo "$(GREEN)Metrics report generated!$(NC)"

# CI/CD
setup-ci: ## Setup CI/CD configuration
	@echo "$(BLUE)Setting up CI/CD...$(NC)"
	mkdir -p .github/workflows
	cp templates/ci.yml .github/workflows/
	@echo "$(GREEN)CI/CD setup complete!$(NC)"

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	# Add your staging deployment commands here
	@echo "$(GREEN)Deployed to staging!$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	# Add your production deployment commands here
	@echo "$(GREEN)Deployed to production!$(NC)"

# Development Utilities
demo: ## Run demo with sample data
	@echo "$(BLUE)Running demo...$(NC)"
	$(PYTHON) -m src.bioneuro_olfactory_fusion --test-fusion
	$(PYTHON) -m src.backlog_manager --demo
	@echo "$(GREEN)Demo complete!$(NC)"

validate: ## Validate project configuration
	@echo "$(BLUE)Validating project configuration...$(NC)"
	$(PYTHON) -c "import toml; toml.load('pyproject.toml')"
	$(PYTHON) -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"
	@echo "$(GREEN)Configuration valid!$(NC)"

install-hooks: ## Install git hooks
	@echo "$(BLUE)Installing git hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Git hooks installed!$(NC)"

check: format-check lint security quality-gates test ## Run all checks (CI pipeline)
	@echo "$(GREEN)All checks passed!$(NC)"

# Database and Migration (for future use)
migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	# Add migration commands here
	@echo "$(GREEN)Migrations complete!$(NC)"

seed: ## Seed database with sample data
	@echo "$(BLUE)Seeding database...$(NC)"
	# Add seeding commands here
	@echo "$(GREEN)Database seeded!$(NC)"

# Teardown
teardown: ## Clean up everything
	@echo "$(YELLOW)Warning: This will remove all build artifacts, caches, and virtual environments!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo; \
		$(MAKE) clean; \
		rm -rf venv/ .venv/; \
		docker system prune -f; \
		echo "$(GREEN)Teardown complete!$(NC)"; \
	else \
		echo; \
		echo "$(BLUE)Teardown cancelled.$(NC)"; \
	fi

# Version management
version: ## Show current version
	@echo "$(BLUE)Current version: $(VERSION)$(NC)"

version-bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bump2version patch
	@echo "$(GREEN)Version bumped!$(NC)"

version-bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bump2version minor
	@echo "$(GREEN)Version bumped!$(NC)"

version-bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	bump2version major
	@echo "$(GREEN)Version bumped!$(NC)"

# Quick development workflow
dev: install-dev format lint test ## Quick development setup and validation
	@echo "$(GREEN)Development workflow complete!$(NC)"

# Full CI pipeline
ci: install-dev check build ## Complete CI pipeline
	@echo "$(GREEN)CI pipeline successful!$(NC)"

# Production release pipeline
release: ci version publish docker-build docker-push ## Complete release pipeline
	@echo "$(GREEN)Release pipeline successful!$(NC)"