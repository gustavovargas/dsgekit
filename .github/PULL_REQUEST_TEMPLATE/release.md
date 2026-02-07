## Release PR

### Scope

- Release candidate:
- Tag target:
- Related issue:

### DYN-G04 Legal Gate Checklist

- [ ] Rebuilt `THIRD_PARTY_NOTICES.md` with:
  - `python tools/generate_third_party_notices.py --extras sym,plot,speed,jax`
- [ ] Reviewed `docs/legal_and_attribution.md`
- [ ] Reviewed `docs/brand_naming_policy.md`
- [ ] Ran mention audit:
  - `rg -n --hidden --glob '!.venv/**' --glob '!.pytest_cache/**' --glob '!.git/**' 'Dynare|dynare' README.md docs src tests .github/workflows CHANGELOG.md DEVELOPMENT.md`
- [ ] No endorsement/affiliation wording introduced in release notes or README

### Evidence

- Command output summary:
- Files changed for legal/attribution:
- Reviewer notes:

### Sign-off

- [ ] Maintainer A
- [ ] Maintainer B
