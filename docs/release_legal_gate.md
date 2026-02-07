# Release Legal Gate

Internal pre-release legal and attribution gate before creating a public tag/release.

This is a process checklist, not legal advice.

## Goal

Ensure each public release has auditable evidence for:

- non-affiliation wording
- third-party attribution references
- dependency-license inventory generation
- naming-policy compliance on public-facing text

## When To Run

- Run once per public release candidate, after feature freeze and before tag creation.
- Evidence must be attached to the release PR.

## Required Inputs

- `docs/legal_and_attribution.md`
- `docs/brand_naming_policy.md`
- `THIRD_PARTY_NOTICES.md`
- `tools/generate_third_party_notices.py`

## Checklist

- [ ] Rebuild dependency notices and commit any changes:
  - `python tools/generate_third_party_notices.py --extras sym,plot,speed,jax`
- [ ] Confirm legal page is current:
  - verify non-affiliation notice is present
  - verify third-party license/reference links are valid
- [ ] Confirm naming-policy compliance in public surfaces:
  - run: `rg -n --hidden --glob '!.venv/**' --glob '!.pytest_cache/**' --glob '!.git/**' 'Dynare|dynare' README.md docs src tests .github/workflows CHANGELOG.md DEVELOPMENT.md`
  - expected result for public docs/copy: references are limited to explicitly allowed context (legal/attribution and intentional migration notes)
- [ ] Confirm release notes/README changes do not introduce endorsement-style wording.
- [ ] Attach evidence block to the release PR (template below).

## PR Evidence Block (Copy/Paste)

```markdown
## DYN-G04 Legal Gate Evidence

- Date (UTC):
- Release candidate:
- Gate owner:

### Commands Run
- `python tools/generate_third_party_notices.py --extras sym,plot,speed,jax`
- `rg -n --hidden --glob '!.venv/**' --glob '!.pytest_cache/**' --glob '!.git/**' 'Dynare|dynare' README.md docs src tests .github/workflows CHANGELOG.md DEVELOPMENT.md`

### Results
- THIRD_PARTY_NOTICES status:
- Mention-audit status:
- Files reviewed:
  - `docs/legal_and_attribution.md`
  - `docs/brand_naming_policy.md`
  - `THIRD_PARTY_NOTICES.md`

### Reviewer Sign-off
- [ ] Maintainer A
- [ ] Maintainer B
```

## Exit Criteria

The legal gate passes when:

- checklist items are completed
- evidence block is present in the release PR
- at least one maintainer confirms completion in PR review
