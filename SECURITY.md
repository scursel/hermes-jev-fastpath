# Security policy

## Supported version

Security fixes are applied to the latest commit on the default branch. No long-term support
branches are currently maintained.

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/scursel/hermes-jev-fastpath/security/advisories/new).
Do not include credentials, real session transcripts, or unredacted telemetry in a public
issue.

Useful reports include:

- affected commit and Hermes version;
- minimal reproduction using synthetic data;
- expected and observed behavior;
- impact on credential isolation, request routing, provider-call counts, response adapters,
  or telemetry privacy;
- a suggested fix, if available.

## Security boundary

The plugin is trusted in-process Hermes code. Its primary controls are:

- fixed handler allowlist;
- typed Jev response validation;
- no dynamic imports, shell, subprocess, `eval`, or `exec` in runtime code;
- profile-scoped credential resolution;
- one bounded HTTPS request with redirects rejected;
- hard timeout, response-size cap, and circuit breaker;
- fail-open fallback to the normal Hermes provider path;
- bounded, redacted telemetry with hashed session and turn identifiers.

These controls reduce risk but do not make plugins a sandbox. Audit and pin the exact commit
before production installation.
