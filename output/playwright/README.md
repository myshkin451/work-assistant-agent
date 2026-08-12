# T-003 browser evidence

These screenshots were captured from the loopback-only Docker Compose stack on
2026-08-12 with Playwright CLI.

- `t003-normal-flow.png`: desktop completed Run with one Tool step and a source.
- `t003-mobile-history.png`: 390×844 historical conversation after PostgreSQL-backed refresh.

The browser acceptance also exercised an active Run cancellation and observed
the `cancelled` UI state. Console checks reported zero errors and zero warnings
for the accepted normal, refresh, cancellation, and mobile sessions. Generated
Playwright session files remain local and ignored.
