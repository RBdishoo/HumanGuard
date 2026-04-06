# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (main) | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in HumanGuard, please **do not open a public GitHub issue**.

Instead, report it by emailing the maintainer directly. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or proof-of-concept (if available)
- Affected component(s) and version/commit

You will receive an acknowledgement within 48 hours and a resolution timeline within 7 days.

## Scope

In scope:
- `/api/*` endpoints (authentication bypass, injection, SSRF, rate-limit bypass)
- Webhook signature verification
- API key storage and validation
- Data exposure via public endpoints

Out of scope:
- Denial-of-service via high request volume (no SLA guarantee on free tier)
- Issues requiring physical access to the deployment environment
- Vulnerabilities in third-party dependencies that have no upstream fix

## Security Posture

Key controls in place:
- API keys stored as `id + SHA-256(secret)` — plaintext never persisted
- CORS restricted to an explicit allowlist; no wildcard origins in production
- IP rate limiting on `/api/register` (3 registrations per IP per hour)
- Atomic quota enforcement via `UPDATE … WHERE count < limit RETURNING` (no TOCTOU race)
- Webhook payloads signed with HMAC-SHA256 (`X-HumanGuard-Signature`)
- Webhook URLs validated against private/loopback/link-local addresses (SSRF guard)
- RDS PostgreSQL in private subnet; `PubliclyAccessible=False`
- All secrets managed via AWS Secrets Manager; no plaintext credentials in source or environment variables
- CloudFront enforces HTTPS; HTTP requests are redirected
