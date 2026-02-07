# Brand and Naming Policy

Internal guide for references to third-party projects in CLI, docs, and repository copy.

## Goal

Use third-party names only when strictly necessary for interoperability or migration, with neutral wording and no implication of affiliation, endorsement, sponsorship, or certification.

## Scope

Applies to:

- CLI help, command descriptions, and generated dashboards
- documentation pages and tutorials
- repository-facing copy (`README.md`, release notes, workflow logs)

## Writing Standard

- Default to neutral language:
  - "`.mod` workflow"
  - "baseline compatibility checks"
  - "baseline regression suite"
- Mention a third-party name only when one of these is true:
  - legal attribution is required
  - migration context would be ambiguous without the explicit name
- Place non-affiliation context in `docs/legal_and_attribution.md` instead of repeating disclaimers across many pages.

## Allowed Patterns

- "legacy workflow migration (see legal notice)"
- "baseline regression suite"

## Disallowed Patterns

- "official integration"
- "approved/certified by <third-party>"
- "partnered with <third-party> maintainers"
- promotional phrasing centered on another project's brand

## Exception Zones

These are the only places where explicit third-party names are expected:

- `docs/legal_and_attribution.md`
- dedicated migration page(s) where the audience is users coming from that tool
- fixture names that encode baseline provenance

## Pre-merge Checklist

- Is the third-party name necessary in this exact sentence?
- Could neutral wording communicate the same idea?
- If an explicit name is used, is it in an allowed exception zone?
- Are legal/attribution references still correct (`docs/legal_and_attribution.md`, `THIRD_PARTY_NOTICES.md`)?
