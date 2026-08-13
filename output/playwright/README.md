# Browser evidence

## T-003

These screenshots were captured from the loopback-only Docker Compose stack on
2026-08-12 with Playwright CLI.

- `t003-normal-flow.png`: desktop completed Run with one Tool step and a source.
- `t003-mobile-history.png`: 390×844 historical conversation after PostgreSQL-backed refresh.

The browser acceptance also exercised an active Run cancellation and observed
the `cancelled` UI state. Console checks reported zero errors and zero warnings
for the accepted normal, refresh, cancellation, and mobile sessions. Generated
Playwright session files remain local and ignored.

## T-005

These screenshots were captured on 2026-08-13 from the isolated
`work-assistant-t005-e3` Compose project with PostgreSQL and the explicit
`development_header` IdentityProvider:

- `t005-unauthenticated.png`: no Principal returns the bounded authentication
  alert, clears content, and disables new conversation and composer controls.
- `t005-principal-a-desktop.png` and `t005-principal-a-mobile.png`: Principal A
  completed and refreshed its own persisted Run on desktop and 390×844 without
  horizontal overflow.
- `t005-principal-b-desktop.png`: a separate browser context completed and
  refreshed Principal B's own history; direct REST/SSE probes of A returned
  `403` without A content or event frames.
- `t005-sse-a-before-backend-restart.png`: A's private question and active Run
  were visibly rendered before a forced backend disconnect.
- `t005-sse-b-reconnect-cleared.png`: the same page had switched to B before the
  next SSE connection; reconnect returned `403`, cleared A's title/content and
  draft, removed the stop action, and disabled all mutations.

The accepted A normal lane had zero unexpected console errors. Expected browser
network errors from deliberate `401`, `403`, and hard-disconnect negative lanes
were classified separately. URL/resource entries, DOM, localStorage,
sessionStorage, REST/SSE product payloads, and screenshots contained no test
subject. No trace, HAR, or raw request-header artifact was retained.

After the final cancellation-cleanup race fix, the backend image was rebuilt and
the three isolated browser contexts were run again. The unauthenticated context
remained blocked; A and B each completed, refreshed, and saw exactly one distinct
owned history item; A's 390x844 view had no horizontal overflow. Browser storage,
DOM, and URL checks again contained no subject. This post-fix rerun retained no
additional screenshot, trace, HAR, or raw-header artifact because the code change
was confined to backend task cancellation and did not change the rendered UI.
