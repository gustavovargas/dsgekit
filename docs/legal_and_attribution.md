# Legal and Attribution Notes

This document provides a public legal notice and third-party attribution references.

It is not legal advice.

## Non-affiliation Notice

- `dsgekit` is an independent project.
- References to Dynare are descriptive (compatibility/migration context only).
- The project is not affiliated with, endorsed by, or sponsored by Dynare maintainers.

## Third-Party References (Dynare)

Checked on: 2026-02-07

- Dynare software license: GNU GPL v3 or later  
  https://www.dynare.org/license/
- Dynare manual license: GNU Free Documentation License 1.3  
  https://www.dynare.org/manual/index.html#preface
- Dynare website content: CC BY-NC-ND 4.0 (site footer)  
  https://www.dynare.org/

## Scope

- This page does not include internal release procedures.
- Internal release gate checklist: `docs/release_legal_gate.md`.
- Public references to Dynare in this repository are intended only for interoperability and migration context.

## Third-Party Dependency Notices

- Inventory file: `THIRD_PARTY_NOTICES.md` (repository root).
- Generation script: `tools/generate_third_party_notices.py`.
- Rebuild command:

```bash
python tools/generate_third_party_notices.py --extras sym,plot,speed,jax
```

Notes:

- The inventory is generated from `pyproject.toml` plus installed package metadata.
- Rows marked `missing` are declared in the selected dependency closure but not installed in the current environment.

## Naming Policy

- Internal wording policy for third-party terms: `docs/brand_naming_policy.md`.
