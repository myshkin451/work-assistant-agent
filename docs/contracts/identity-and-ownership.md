# Principal and ownership contract v0.3

This contract defines the public core's identity seam and subject-scoped product
ownership. It is deliberately neutral: authentication credentials, login UI,
company SSO, roles-based policy, and identity-provider-specific claims belong to
downstream adapters, not this repository.

## Principal

`Principal` is the authenticated application identity passed to every product
API operation. Its `subject` is an opaque, deployment-wide stable identifier.
Subject comparison is case-sensitive and exact; no role, organization, display
name, or session field grants an ownership bypass.

A valid subject:

- is non-empty after exact input validation and at most 255 characters;
- has no surrounding whitespace or Unicode control/format characters;
- does not use the reserved `urn:work-assistant:internal:` namespace.

Optional display, organization, role, and session metadata may be supplied by a
downstream provider, but T-005 does not interpret it for authorization. A
Principal never contains a password, bearer token, cookie, API key, or raw
provider assertion.

## IdentityProvider

An `IdentityProvider` implements the neutral asynchronous boundary:

```python
async def authenticate(request: fastapi.Request) -> Principal | None: ...
```

`None`, an invalid Principal, or a provider error is treated as
`401 authentication_required`. The Host never falls back to another provider or
to an anonymous identity after an authentication failure.

The built-in modes are:

| mode | allowed environment | behavior |
| --- | --- | --- |
| `external` | development, test, production | requires an `IdentityProvider` injected into `create_app`; absence aborts startup |
| `anonymous` | development or test only | explicit local compatibility mode using one fixed neutral Principal |
| `development_header` | development or test only | reads exactly one `X-Work-Assistant-Dev-Subject` header; missing, duplicate, invalid, or reserved values are unauthenticated |

The default mode is `external`. Production rejects either built-in development
provider during configuration validation, and `external` without an injected
provider fails before the database or Runtime startup work in the application
lifespan. Production also requires a non-loopback exact origin allowlist; the
development loopback defaults are rejected. The Compose quick start explicitly
selects `anonymous`; it is a single-Principal development environment, not a
production identity system.

## Ownership invariants

- Every Thread stores one immutable, non-null `owner_subject` from the current
  Principal at creation.
- Every Run stores one immutable, non-null `actor_subject` from the current
  Principal. A Run may be created only by its Thread owner, so new rows have an
  actor equal to the owner.
- Messages and Events do not duplicate identity fields. Their Thread and Run
  relationship is protected by a composite database foreign key, and ownership
  is inherited from those parent records.
- Public Thread, Run, Message, Event, REST, and SSE representations do not expose
  owner or actor subjects.
- The default ownership authorizer accepts only exact subject equality. Reserved
  internal subjects are denied to every Principal. There is no admin, role,
  wildcard, sharing, delegation, transfer, or claim bypass in this contract.

All user-addressable repository methods require an explicit Principal. Internal
executor methods may advance a previously authorized Run without carrying an
HTTP credential, but they fail closed if the Run and Thread ownership facts are
inconsistent. Identity metadata is not passed into model messages, prompts,
Tools, checkpoints, product Events, or error details.

## Lookup and error semantics

`/health` remains public. Every `/api` request authenticates before product
validation or lookup. Once authenticated, the Host distinguishes a known
foreign resource from an unknown identifier:

- no valid Principal: `401` with `authentication_required`;
- existing Thread owned by another Principal: `403` with `thread_forbidden`;
- existing Run whose Thread owner or actor differs: `403` with `run_forbidden`;
- unknown identifier: `404` with `thread_not_found` or `run_not_found`;
- browser mutation with an untrusted or duplicate `Origin`: `403` with
  `origin_forbidden`.

This deliberate `403`/`404` distinction is part of the v0.3 contract. Forbidden
responses reveal no title, message, Run status, owner, actor, event, idempotency
result, or active-Run conflict. Lists are owner-filtered in the database. Every
SSE request and reconnect authenticates again; an authorization failure is a
normal JSON response emitted before the stream starts.

For browser mutations, an absent `Origin` is accepted for non-browser clients;
when `Origin` is present, every `POST /api/**` requires exactly one value from
the configured allowlist before any mutation executes. This is the neutral
Host's ambient-cookie CSRF baseline. Credentialed CORS controls response access;
it is not itself a CSRF defense. A downstream cookie provider must additionally
set an appropriate Secure/SameSite cookie policy and may add stronger request
binding without weakening the Host's exact-origin check.

## v0.2 migration

Pre-v0.3 rows have no attributable owner. Migration `0002_principal_ownership`
preserves them and deterministically assigns Thread owner and Run actor to:

```text
urn:work-assistant:internal:legacy-unowned:v0.2
```

No IdentityProvider may emit this reserved subject, and the authorizer denies it
even if a malformed Principal is constructed elsewhere. Consequently migrated
rows appear in no normal list and direct access returns `403` for every valid
Principal. They are never claimed by the first caller, the anonymous development
Principal, or a wildcard. A future reassignment, if needed, must be an explicit,
audited offline operation outside this Task.

After backfill, owner and actor columns are `NOT NULL` with no runtime default.
Downgrading to the pre-ownership schema is allowed only when all product tables
are empty; a non-empty downgrade is rejected because running an older unauthenticated
application would make preserved data shared again. The safe application rollback
target after this migration is another ownership-aware version, not v0.2.

The migration is online-connection-only in the Alembic sense: `--sql` offline
generation is intentionally unsupported because the upgrade must inspect legacy
relationships and the downgrade must inspect table contents. Operationally it
must still run in an exclusive application downtime window with every v0.2
writer stopped. “Online connection” does not mean concurrent old/new application
writes are safe during backfill or downgrade checks.

## Browser transport boundary

REST and fetch-based SSE use credential mode `include`, allowing a downstream
deployment to use an HttpOnly session cookie. CORS origins are exact and
credentialed; wildcard origins are not supported. The development subject is
never placed in a URL, frontend build variable, DOM control, localStorage,
sessionStorage, product payload, or SSE data. Browser E3 injects the development
header from isolated test contexts only.
