# Third-party licenses

Work Assistant Agent is licensed under MIT. Third-party packages are not
relicensed by this repository; each remains subject to its own license.

This source-release review is based on `backend/uv.lock` and
`frontend/pnpm-lock.yaml` as committed for v0.1. It covers 75 locked Python
packages and 252 unique locked JavaScript package/version pairs. No AGPL,
GPL-only, SSPL, BUSL, Commons Clause, proprietary, or unlicensed dependency was
identified.

## License summary

| Ecosystem | Locked packages | Main license families |
| --- | ---: | --- |
| Python | 75 | MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, LGPL-3.0-only, MPL-2.0, PSF-2.0, and compatible dual licenses |
| JavaScript | 252 | MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, MPL-2.0, MIT-0, BlueOak-1.0.0, CC0-1.0, Python-2.0, and 0BSD |

The frontend production dependency closure is React, React DOM, and Scheduler;
all three are MIT-licensed. The wider JavaScript inventory includes build and
test dependencies.

## Packages requiring particular attention

| Ecosystem | Package | Version | Scope | SPDX / license |
| --- | --- | ---: | --- | --- |
| Python | psycopg | 3.3.4 | runtime | LGPL-3.0-only |
| Python | psycopg-binary | 3.3.4 | runtime | LGPL-3.0-only |
| Python | psycopg-pool | 3.3.1 | runtime | LGPL-3.0-only |
| Python | certifi | 2026.7.22 | runtime | MPL-2.0 |
| Python | orjson | 3.11.9 | runtime | MPL-2.0 AND (Apache-2.0 OR MIT) |
| Python | tqdm | 4.70.0 | transitive runtime | MPL-2.0 AND MIT |
| Python | pathspec | 1.1.1 | development | MPL-2.0 |
| JavaScript | lightningcss and platform packages | 1.33.0 | build / development | MPL-2.0 |

Apache-2.0 dependencies retain their required copyright, license, and NOTICE
terms. MPL-2.0 obligations apply to covered files when those files are
distributed in a modified form. LGPL-3.0 obligations must be reviewed for any
distribution that includes the psycopg implementation or bundled native
libraries; recipients must retain the applicable notices and license rights.

## Evidence and distribution boundary

- Exact package versions and hashes: `backend/uv.lock` and
  `frontend/pnpm-lock.yaml`.
- Python license metadata: installed wheel `*.dist-info/METADATA` and the
  corresponding exact-version PyPI records.
- JavaScript license metadata: exact-version npm package metadata.
- License terms: [SPDX License List](https://spdx.org/licenses/),
  [GNU LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html),
  [Mozilla MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/), and
  [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

This inventory supports publication of this Git source repository. It is not a
license manifest for a Docker image, operating-system packages, or a binary
release. Before distributing an image or binary, regenerate an SBOM and the
complete third-party notice/license bundle from the exact release artifact,
including base images and bundled native libraries.
