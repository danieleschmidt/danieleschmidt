#!/usr/bin/env python3
"""
Autonomous Project Structure Generator
Creates standardized project templates with SDLC best practices
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProjectTemplate:
    """Project template configuration"""
    name: str
    type: str  # python, nodejs, rust, go, docker, etc.
    description: str
    directories: List[str]
    files: Dict[str, str]  # filename -> content
    dependencies: List[str]
    dev_dependencies: List[str]
    scripts: Dict[str, str]
    features: List[str]  # ci, testing, security, monitoring, etc.

class ProjectGenerator:
    """Generates standardized project structures with SDLC integration"""
    
    TEMPLATES = {
        "python-api": ProjectTemplate(
            name="Python API",
            type="python",
            description="FastAPI-based REST API with comprehensive SDLC",
            directories=[
                "src/api", "src/core", "src/models", "src/services", "src/utils",
                "tests/unit", "tests/integration", "tests/e2e", "tests/performance",
                "docs/api", "docs/deployment", "docs/architecture",
                "scripts", "config", "docker", ".github/workflows",
                "monitoring", "security"
            ],
            files={},  # Will be populated by _get_python_api_files()
            dependencies=[
                "fastapi>=0.104.0", "uvicorn[standard]>=0.24.0", "pydantic>=2.4.0",
                "sqlalchemy>=2.0.0", "alembic>=1.12.0", "redis>=5.0.0",
                "celery>=5.3.0", "httpx>=0.25.0", "python-jose[cryptography]>=3.3.0",
                "passlib[bcrypt]>=1.7.4", "python-multipart>=0.0.6"
            ],
            dev_dependencies=[
                "pytest>=7.4.0", "pytest-asyncio>=0.21.0", "pytest-cov>=4.1.0",
                "pytest-mock>=3.11.0", "black>=23.7.0", "isort>=5.12.0",
                "flake8>=6.0.0", "mypy>=1.5.0", "bandit>=1.7.5", "safety>=2.3.0",
                "pre-commit>=3.4.0", "locust>=2.16.0"
            ],
            scripts={
                "dev": "uvicorn src.main:app --reload --host 0.0.0.0 --port 8000",
                "test": "pytest tests/ -v --cov=src --cov-report=html",
                "lint": "flake8 src/ tests/ && black --check src/ tests/ && isort --check src/ tests/",
                "format": "black src/ tests/ && isort src/ tests/",
                "typecheck": "mypy src/",
                "security": "bandit -r src/ && safety check",
                "build": "docker build -t app .",
                "start": "uvicorn src.main:app --host 0.0.0.0 --port 8000"
            },
            features=["ci", "testing", "security", "monitoring", "docker", "api-docs"]
        ),
        
        "nodejs-app": ProjectTemplate(
            name="Node.js Application",
            type="nodejs",
            description="Express.js application with TypeScript and comprehensive tooling",
            directories=[
                "src/controllers", "src/middleware", "src/models", "src/routes", 
                "src/services", "src/utils", "src/types",
                "tests/unit", "tests/integration", "tests/e2e",
                "docs", "scripts", "config", "docker", ".github/workflows",
                "monitoring", "security"
            ],
            files={},  # Will be populated by _get_nodejs_files()
            dependencies=[
                "express@^4.18.0", "cors@^2.8.5", "helmet@^7.0.0",
                "morgan@^1.10.0", "compression@^1.7.4", "dotenv@^16.3.0",
                "jsonwebtoken@^9.0.2", "bcryptjs@^2.4.3", "validator@^13.11.0",
                "winston@^3.10.0", "swagger-ui-express@^5.0.0"
            ],
            dev_dependencies=[
                "@types/node@^20.5.0", "@types/express@^4.17.17", 
                "@types/cors@^2.8.13", "@types/morgan@^1.9.4",
                "typescript@^5.1.6", "ts-node@^10.9.1", "nodemon@^3.0.1",
                "jest@^29.6.2", "@types/jest@^29.5.3", "ts-jest@^29.1.1",
                "supertest@^6.3.3", "@types/supertest@^2.0.12",
                "eslint@^8.47.0", "@typescript-eslint/eslint-plugin@^6.4.0",
                "prettier@^3.0.1", "husky@^8.0.3", "lint-staged@^13.2.3"
            ],
            scripts={
                "dev": "nodemon src/index.ts",
                "build": "tsc",
                "start": "node dist/index.js",
                "test": "jest --coverage",
                "test:watch": "jest --watch",
                "lint": "eslint src/ tests/ --ext .ts",
                "format": "prettier --write src/ tests/",
                "typecheck": "tsc --noEmit"
            },
            features=["ci", "testing", "security", "monitoring", "docker", "api-docs"]
        ),
        
        "rust-cli": ProjectTemplate(
            name="Rust CLI Tool",
            type="rust",
            description="Command-line tool with Rust best practices",
            directories=[
                "src/commands", "src/config", "src/utils", "src/models",
                "tests/integration", "tests/unit",
                "docs", "scripts", ".github/workflows",
                "benches", "examples"
            ],
            files={},  # Will be populated by _get_rust_files()
            dependencies=[
                "clap", "serde", "serde_json", "tokio", "anyhow", "thiserror",
                "tracing", "tracing-subscriber", "config", "directories"
            ],
            dev_dependencies=[
                "criterion", "tempfile", "assert_cmd", "predicates"
            ],
            scripts={
                "build": "cargo build",
                "build-release": "cargo build --release",
                "test": "cargo test",
                "bench": "cargo bench",
                "lint": "cargo clippy -- -D warnings",
                "format": "cargo fmt",
                "doc": "cargo doc --open",
                "install": "cargo install --path ."
            },
            features=["ci", "testing", "benchmarking", "documentation"]
        )
    }
    
    def __init__(self, output_dir: str = "."):
        """Initialize project generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_project(self, 
                        template_name: str, 
                        project_name: str,
                        features: Optional[List[str]] = None) -> Path:
        """Generate a new project from template"""
        
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.TEMPLATES[template_name]
        project_path = self.output_dir / project_name
        
        if project_path.exists():
            raise FileExistsError(f"Project directory already exists: {project_path}")
        
        logger.info(f"Generating {template.name} project: {project_name}")
        
        # Create project directory
        project_path.mkdir(parents=True)
        
        # Create directory structure
        self._create_directories(project_path, template.directories)
        
        # Generate template files
        template_files = self._get_template_files(template)
        self._create_files(project_path, template_files, project_name)
        
        # Generate configuration files
        self._create_config_files(project_path, template, project_name)
        
        # Create SDLC files
        if not features:
            features = template.features
        self._create_sdlc_files(project_path, template, features)
        
        # Initialize version control
        self._init_git_repo(project_path)
        
        # Generate documentation
        self._create_documentation(project_path, template, project_name)
        
        logger.info(f"✅ Project generated successfully at {project_path}")
        return project_path
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List available project templates"""
        return [
            {
                "name": name,
                "type": template.type,
                "description": template.description,
                "features": template.features
            }
            for name, template in self.TEMPLATES.items()
        ]
    
    def _create_directories(self, project_path: Path, directories: List[str]) -> None:
        """Create directory structure"""
        for directory in directories:
            dir_path = project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if directory.startswith("src/") or directory.startswith("tests/"):
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
    
    def _create_files(self, project_path: Path, files: Dict[str, str], project_name: str) -> None:
        """Create files from templates"""
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Replace placeholders
            content = content.replace("{{PROJECT_NAME}}", project_name)
            content = content.replace("{{TIMESTAMP}}", datetime.now().isoformat())
            
            with open(full_path, 'w') as f:
                f.write(content)
    
    def _create_config_files(self, project_path: Path, template: ProjectTemplate, project_name: str) -> None:
        """Create configuration files"""
        
        if template.type == "python":
            self._create_python_config(project_path, template, project_name)
        elif template.type == "nodejs":
            self._create_nodejs_config(project_path, template, project_name)
        elif template.type == "rust":
            self._create_rust_config(project_path, template, project_name)
    
    def _create_python_config(self, project_path: Path, template: ProjectTemplate, project_name: str) -> None:
        """Create Python configuration files"""
        
        # requirements.txt
        with open(project_path / "requirements.txt", 'w') as f:
            f.write('\n'.join(template.dependencies))
        
        # requirements-dev.txt
        with open(project_path / "requirements-dev.txt", 'w') as f:
            f.write('\n'.join(template.dev_dependencies))
        
        # setup.py
        setup_content = f'''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{project_name}",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of the project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/{project_name}",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={{
        "console_scripts": [
            "{project_name}=src.main:main",
        ],
    }},
)
'''
        with open(project_path / "setup.py", 'w') as f:
            f.write(setup_content)
        
        # pyproject.toml
        pyproject_content = f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "A modern Python project"
readme = "README.md"
requires-python = ">=3.9"
license = {{file = "LICENSE"}}
authors = [
    {{name = "Your Name", email = "your.email@example.com"}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[project.urls]
Homepage = "https://github.com/yourusername/{project_name}"
Repository = "https://github.com/yourusername/{project_name}"
Issues = "https://github.com/yourusername/{project_name}/issues"

[project.scripts]
{project_name} = "src.main:main"

[tool.black]
line-length = 88
target-version = ['py39']
include = '\\.pyi?$'
extend-exclude = '''
/(
  \\.git
| \\.hg
| \\.mypy_cache
| \\.tox
| \\.venv
| _build
| buck-out
| build
| dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=term-missing --cov-report=html"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/venv/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
'''
        with open(project_path / "pyproject.toml", 'w') as f:
            f.write(pyproject_content)
    
    def _create_nodejs_config(self, project_path: Path, template: ProjectTemplate, project_name: str) -> None:
        """Create Node.js configuration files"""
        
        # package.json
        package_json = {
            "name": project_name.lower().replace('_', '-'),
            "version": "0.1.0",
            "description": "A modern Node.js application",
            "main": "dist/index.js",
            "scripts": template.scripts,
            "keywords": [],
            "author": "Your Name <your.email@example.com>",
            "license": "MIT",
            "dependencies": {dep.split('@')[0]: dep.split('@')[1] if '@' in dep else "latest" 
                           for dep in template.dependencies},
            "devDependencies": {dep.split('@')[0]: dep.split('@')[1] if '@' in dep else "latest" 
                              for dep in template.dev_dependencies},
            "engines": {
                "node": ">=18.0.0",
                "npm": ">=8.0.0"
            }
        }
        
        with open(project_path / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # tsconfig.json
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "lib": ["ES2020"],
                "outDir": "./dist",
                "rootDir": "./src",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True,
                "declaration": True,
                "declarationMap": True,
                "sourceMap": True
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist", "tests"]
        }
        
        with open(project_path / "tsconfig.json", 'w') as f:
            json.dump(tsconfig, f, indent=2)
    
    def _create_rust_config(self, project_path: Path, template: ProjectTemplate, project_name: str) -> None:
        """Create Rust configuration files"""
        
        # Cargo.toml
        cargo_toml = f'''[package]
name = "{project_name.lower().replace('_', '-')}"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "A modern Rust application"
license = "MIT"
repository = "https://github.com/yourusername/{project_name}"

[[bin]]
name = "{project_name.lower().replace('_', '-')}"
path = "src/main.rs"

[dependencies]
{chr(10).join(f'{dep} = "*"' for dep in template.dependencies)}

[dev-dependencies] 
{chr(10).join(f'{dep} = "*"' for dep in template.dev_dependencies)}

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
'''
        
        with open(project_path / "Cargo.toml", 'w') as f:
            f.write(cargo_toml)
    
    def _create_sdlc_files(self, project_path: Path, template: ProjectTemplate, features: List[str]) -> None:
        """Create SDLC-related files"""
        
        # .gitignore
        gitignore_content = self._get_gitignore_content(template.type)
        with open(project_path / ".gitignore", 'w') as f:
            f.write(gitignore_content)
        
        # Dockerfile
        if "docker" in features:
            dockerfile_content = self._get_dockerfile_content(template.type)
            with open(project_path / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # docker-compose.yml
            compose_content = self._get_compose_content(template.type)
            with open(project_path / "docker-compose.yml", 'w') as f:
                f.write(compose_content)
        
        # GitHub Actions workflow
        if "ci" in features:
            workflow_content = self._get_workflow_content(template.type)
            workflow_path = project_path / ".github/workflows/ci.yml"
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            with open(workflow_path, 'w') as f:
                f.write(workflow_content)
        
        # Security files
        if "security" in features:
            self._create_security_files(project_path)
        
        # Monitoring files
        if "monitoring" in features:
            self._create_monitoring_files(project_path)
    
    def _create_security_files(self, project_path: Path) -> None:
        """Create security-related files"""
        
        # SECURITY.md
        security_content = '''# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by emailing [security@example.com].

Do not report security vulnerabilities through public GitHub issues.

## Security Best Practices

- Keep dependencies updated
- Use secure coding practices
- Follow OWASP guidelines
- Implement proper authentication and authorization
- Validate all inputs
- Use HTTPS everywhere
- Store secrets securely
'''
        with open(project_path / "SECURITY.md", 'w') as f:
            f.write(security_content)
    
    def _create_monitoring_files(self, project_path: Path) -> None:
        """Create monitoring configuration files"""
        
        # health_check.py or health check endpoint
        monitoring_dir = project_path / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        health_check = '''#!/usr/bin/env python3
"""Health check endpoint for monitoring systems"""

import sys
import time
import requests
from typing import Dict, Any

def check_health() -> Dict[str, Any]:
    """Perform comprehensive health checks"""
    checks = {
        "timestamp": time.time(),
        "status": "healthy",
        "checks": {}
    }
    
    # Database connectivity check
    try:
        # Add your database check here
        checks["checks"]["database"] = {"status": "healthy", "response_time": 0.001}
    except Exception as e:
        checks["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        checks["status"] = "unhealthy"
    
    # External service checks
    try:
        # Add external service checks here
        checks["checks"]["external_services"] = {"status": "healthy"}
    except Exception as e:
        checks["checks"]["external_services"] = {"status": "unhealthy", "error": str(e)}
        checks["status"] = "unhealthy"
    
    return checks

if __name__ == "__main__":
    health = check_health()
    print(health)
    sys.exit(0 if health["status"] == "healthy" else 1)
'''
        
        with open(monitoring_dir / "health_check.py", 'w') as f:
            f.write(health_check)
    
    def _create_documentation(self, project_path: Path, template: ProjectTemplate, project_name: str) -> None:
        """Create comprehensive documentation"""
        
        # README.md
        readme_content = f'''# {project_name}

{template.description}

## Features

{chr(10).join(f'- {feature.replace("-", " ").title()}' for feature in template.features)}

## Installation

### Prerequisites

- {self._get_runtime_requirements(template.type)}

### Setup

```bash
git clone https://github.com/yourusername/{project_name}
cd {project_name}
{self._get_install_commands(template.type)}
```

## Usage

```bash
{self._get_usage_examples(template.type)}
```

## Development

### Setup Development Environment

```bash
{self._get_dev_setup_commands(template.type)}
```

### Running Tests

```bash
{template.scripts.get('test', 'npm test')}
```

### Code Quality

```bash
{template.scripts.get('lint', 'npm run lint')}
{template.scripts.get('format', 'npm run format')}
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## API Documentation

{self._get_api_docs_info(template)}

## Deployment

### Docker

```bash
docker build -t {project_name} .
docker run -p 8000:8000 {project_name}
```

### Production

See [docs/deployment.md](docs/deployment.md) for production deployment guidelines.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Security

See [SECURITY.md](SECURITY.md) for security policies and reporting procedures.
'''
        
        with open(project_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        # CONTRIBUTING.md
        contributing_content = '''# Contributing Guide

Thank you for your interest in contributing to this project!

## Development Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `npm test`
6. Run code quality checks: `npm run lint`
7. Commit your changes: `git commit -am 'Add some feature'`
8. Push to the branch: `git push origin feature/your-feature`
9. Submit a pull request

## Code Style

- Follow the existing code style
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused

## Testing

- Write unit tests for new functionality
- Maintain test coverage above 80%
- Test edge cases and error conditions
- Use descriptive test names

## Documentation

- Update README.md for new features
- Add API documentation for new endpoints
- Include code examples where helpful

## Pull Request Guidelines

- Provide a clear description of changes
- Link to relevant issues
- Include screenshots for UI changes
- Ensure all checks pass

## Questions?

Feel free to open an issue for questions or discussions.
'''
        
        with open(project_path / "CONTRIBUTING.md", 'w') as f:
            f.write(contributing_content)
        
        # LICENSE
        license_content = '''MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
        
        with open(project_path / "LICENSE", 'w') as f:
            f.write(license_content)
    
    def _init_git_repo(self, project_path: Path) -> None:
        """Initialize Git repository with initial commit"""
        try:
            subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: Project structure generated"], 
                         cwd=project_path, check=True, capture_output=True)
            logger.info("Git repository initialized")
        except subprocess.CalledProcessError:
            logger.warning("Failed to initialize Git repository")
    
    def _get_template_files(self, template: ProjectTemplate) -> Dict[str, str]:
        """Get template-specific files"""
        if template.type == "python":
            return self._get_python_api_files()
        elif template.type == "nodejs":
            return self._get_nodejs_files()
        elif template.type == "rust":
            return self._get_rust_files()
        return {}
    
    def _get_python_api_files(self) -> Dict[str, str]:
        """Get Python API template files"""
        return {
            "src/main.py": '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import api_router
from src.core.config import settings

app = FastAPI(
    title="{{PROJECT_NAME}} API",
    description="A modern FastAPI application",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to {{PROJECT_NAME}} API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "{{TIMESTAMP}}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
            "src/api/routes.py": '''from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/items")
async def get_items():
    return {"items": []}

@api_router.post("/items")
async def create_item(item: dict):
    return {"message": "Item created", "item": item}
''',
            "src/core/config.py": '''from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "{{PROJECT_NAME}}"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    ALLOWED_HOSTS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"

settings = Settings()
''',
            "tests/test_main.py": '''import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
'''
        }
    
    def _get_nodejs_files(self) -> Dict[str, str]:
        """Get Node.js template files"""
        return {
            "src/index.ts": '''import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { config } from './config';
import { router } from './routes';

const app = express();

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/v1', router);

app.get('/', (req, res) => {
  res.json({ message: 'Welcome to {{PROJECT_NAME}} API' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

const port = config.port || 3000;

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
''',
            "src/config/index.ts": '''export const config = {
  port: process.env.PORT || 3000,
  nodeEnv: process.env.NODE_ENV || 'development',
  dbUrl: process.env.DATABASE_URL || 'postgresql://localhost:5432/mydb'
};
''',
            "tests/index.test.ts": '''import request from 'supertest';
import express from 'express';

const app = express();
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

describe('API Tests', () => {
  test('Health check endpoint', async () => {
    const response = await request(app).get('/health');
    expect(response.status).toBe(200);
    expect(response.body.status).toBe('healthy');
  });
});
'''
        }
    
    def _get_rust_files(self) -> Dict[str, str]:
        """Get Rust template files"""
        return {
            "src/main.rs": '''use clap::Parser;
use anyhow::Result;

mod commands;
mod config;
mod utils;

#[derive(Parser)]
#[command(name = "{{PROJECT_NAME}}")]
#[command(about = "A modern Rust CLI tool")]
struct Cli {
    #[command(subcommand)]
    command: commands::Commands,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        commands::Commands::Run { verbose } => {
            commands::run(verbose).await
        }
    }
}
''',
            "src/commands/mod.rs": '''use clap::Subcommand;
use anyhow::Result;

#[derive(Subcommand)]
pub enum Commands {
    Run {
        #[arg(short, long)]
        verbose: bool,
    },
}

pub async fn run(verbose: bool) -> Result<()> {
    if verbose {
        println!("Running in verbose mode");
    }
    println!("Hello from {{PROJECT_NAME}}!");
    Ok(())
}
''',
            "tests/integration_test.rs": '''use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_help() {
    let mut cmd = Command::cargo_bin("{{PROJECT_NAME}}").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("A modern Rust CLI tool"));
}
'''
        }
    
    def _get_gitignore_content(self, project_type: str) -> str:
        """Get .gitignore content for project type"""
        common = '''# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# Environment variables
.env
.env.local
.env.production
'''
        
        if project_type == "python":
            return common + '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json
'''
        elif project_type == "nodejs":
            return common + '''
# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn-integrity
.node_repl_history
*.tgz
.yarn/cache/
.yarn/unplugged/
.yarn/build-state.yml
.pnp.*

# Build outputs
dist/
build/

# Testing
coverage/
.nyc_output/

# TypeScript
*.tsbuildinfo
'''
        elif project_type == "rust":
            return common + '''
# Rust
/target/
Cargo.lock
**/*.rs.bk
*.pdb

# Rust docs
/doc/
'''
        
        return common
    
    def _get_dockerfile_content(self, project_type: str) -> str:
        """Get Dockerfile content for project type"""
        if project_type == "python":
            return '''# Multi-stage build for Python
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY . .

# Set ownership and permissions
RUN chown -R appuser:appuser /app
USER appuser

# Add local packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "-m", "src.main"]
'''
        elif project_type == "nodejs":
            return '''# Multi-stage build for Node.js
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine

# Create non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

USER nodejs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

CMD ["node", "dist/index.js"]
'''
        elif project_type == "rust":
            return '''# Multi-stage build for Rust
FROM rust:1.70 as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src/

# Copy actual source code
COPY src ./src

# Build the actual binary
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false appuser

# Copy binary
COPY --from=builder /app/target/release/{{PROJECT_NAME}} /usr/local/bin/{{PROJECT_NAME}}

# Set ownership
RUN chown appuser:appuser /usr/local/bin/{{PROJECT_NAME}}

USER appuser

ENTRYPOINT ["/usr/local/bin/{{PROJECT_NAME}}"]
'''
        
        return ""
    
    def _get_compose_content(self, project_type: str) -> str:
        """Get docker-compose.yml content"""
        return f'''version: '3.8'

services:
  app:
    build: .
    ports:
      - "{'8000:8000' if project_type == 'python' else '3000:3000'}"
    environment:
      - NODE_ENV=development
    volumes:
      - .:/app
      - /app/node_modules
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
'''
    
    def _get_workflow_content(self, project_type: str) -> str:
        """Get GitHub Actions workflow content"""
        # Return a simplified version of the main CI workflow
        return '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup environment
      run: echo "Setting up for ''' + project_type + '''"
    - name: Run tests
      run: echo "Running tests"
'''
    
    def _get_runtime_requirements(self, project_type: str) -> str:
        """Get runtime requirements string"""
        requirements = {
            "python": "Python 3.9+",
            "nodejs": "Node.js 18+",
            "rust": "Rust 1.70+"
        }
        return requirements.get(project_type, "Runtime environment")
    
    def _get_install_commands(self, project_type: str) -> str:
        """Get installation commands"""
        commands = {
            "python": "pip install -r requirements.txt",
            "nodejs": "npm install",
            "rust": "cargo build --release"
        }
        return commands.get(project_type, "# Install dependencies")
    
    def _get_dev_setup_commands(self, project_type: str) -> str:
        """Get development setup commands"""
        commands = {
            "python": "pip install -r requirements-dev.txt\\npre-commit install",
            "nodejs": "npm install\\nnpm run build",
            "rust": "cargo build\\nrustup component add clippy rustfmt"
        }
        return commands.get(project_type, "# Setup development environment")
    
    def _get_usage_examples(self, project_type: str) -> str:
        """Get usage examples"""
        examples = {
            "python": "python -m src.main\\n# or\\nuvicorn src.main:app --reload",
            "nodejs": "npm start\\n# or for development\\nnpm run dev",
            "rust": "cargo run\\n# or\\n./target/release/{{PROJECT_NAME}} --help"
        }
        return examples.get(project_type, "# Run the application")
    
    def _get_api_docs_info(self, template: ProjectTemplate) -> str:
        """Get API documentation information"""
        if "api-docs" in template.features:
            if template.type == "python":
                return "API documentation is automatically generated and available at `/docs` (Swagger UI) and `/redoc` (ReDoc)."
            elif template.type == "nodejs":
                return "API documentation is available at `/api-docs` when running the application."
        return "See the API endpoints in the source code and tests."

def main():
    """CLI interface for project generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Project Structure Generator")
    parser.add_argument("template", help="Template name")
    parser.add_argument("project_name", help="Project name")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--features", nargs="*", help="Features to include")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")
    
    args = parser.parse_args()
    
    generator = ProjectGenerator(args.output_dir)
    
    if args.list_templates:
        templates = generator.list_templates()
        print("Available templates:")
        for template in templates:
            print(f"  {template['name']}: {template['description']}")
        return
    
    try:
        project_path = generator.generate_project(
            args.template,
            args.project_name,
            args.features
        )
        print(f"✅ Project '{args.project_name}' generated successfully at {project_path}")
    except Exception as e:
        print(f"❌ Error generating project: {e}")

if __name__ == "__main__":
    main()