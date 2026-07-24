# Changelog

## [1.1.0] - 2026-07-24

### Added

- Docker and Render deployment configuration with a persistent data disk.
- Automated API checks on every GitHub push and pull request.
- Environment-based runtime storage for crawler and model artifacts.
- Reproducible pinned Python dependencies.

### Security

- Protected production data refreshes with a private update token.
- Added per-client rate limiting to public manual predictions.
- Replaced internal exception details with safe public error responses.
