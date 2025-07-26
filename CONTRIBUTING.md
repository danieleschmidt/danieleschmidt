# Contributing Guidelines

## Autonomous Backlog Management Process

This repository uses an autonomous backlog management system based on WSJF (Weighted Shortest Job First) scoring to prioritize and execute work items.

## How It Works

### 1. Backlog Discovery
- Items are tracked in `backlog.yml`
- TODO/FIXME comments are automatically converted to backlog items
- Issues and PR comments are monitored for actionable tasks

### 2. WSJF Scoring
Items are scored using the formula:
```
WSJF = (Value + Time Criticality + Risk Reduction) / Effort
```

All dimensions use Fibonacci scale: 1, 2, 3, 5, 8, 13

### 3. Execution Principles
- **TDD First**: Write failing test → make it pass → refactor
- **Security by Design**: Input validation, auth, safe logging
- **Small Changes**: Trunk-based development with frequent merges
- **Documentation**: Keep README, CHANGELOG, and API docs current

### 4. Status Flow
```
NEW → REFINED → READY → DOING → PR → DONE/BLOCKED
```

## Manual Contributions

### Adding New Backlog Items
Edit `backlog.yml` following the existing format:
```yaml
- id: "PREFIX-###"
  title: "Clear, actionable title"
  type: "feature|bug|infrastructure|documentation"
  description: "Detailed description of the work"
  acceptance_criteria:
    - "Specific, testable criteria"
  effort: 1-13  # Fibonacci scale
  value: 1-13
  time_criticality: 1-13
  risk_reduction: 1-13
  status: "NEW"
  risk_tier: "low|medium|high"
  created_at: "ISO timestamp"
```

### Pull Request Process
1. Create feature branch from `main`
2. Implement with tests first (TDD)
3. Ensure CI passes (lint, tests, build)
4. Link to backlog item in PR description
5. Include rollback plan for high-risk changes

### Testing Standards
- Unit tests for all business logic
- Integration tests for API endpoints
- Minimal but focused E2E tests
- No vanity coverage - test meaningful scenarios

## Code Quality
- Follow existing code style and patterns
- Use existing libraries and utilities
- Validate inputs and handle errors gracefully
- Never commit secrets or sensitive data
- Document security decisions

## Questions?
Check `docs/status/` for current metrics and progress reports.