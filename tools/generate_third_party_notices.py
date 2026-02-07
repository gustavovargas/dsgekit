"""Generate THIRD_PARTY_NOTICES.md from pyproject + installed metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as importlib_metadata
import platform
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
try:  # pragma: no cover - fallback for minimal environments
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover
    Requirement = None
    default_environment = None


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_")


def parse_requirement_name(requirement: str) -> str | None:
    if Requirement is not None and default_environment is not None:
        try:
            req = Requirement(requirement)
        except Exception:
            req = None
        if req is not None:
            if req.marker is not None:
                env = default_environment()
                env["extra"] = ""
                if not req.marker.evaluate(env):
                    return None
            return normalize_name(req.name)

    match = NAME_RE.match(requirement)
    if not match:
        return None
    return normalize_name(match.group(1))


def parse_requirement_extras(requirement: str) -> list[str]:
    req = requirement.split(";", 1)[0].strip()
    left = req.split(" ", 1)[0]
    if "[" not in left or "]" not in left:
        return []
    extras_text = left.split("[", 1)[1].split("]", 1)[0]
    return [e.strip() for e in extras_text.split(",") if e.strip()]


def discover_selected_requirements(
    pyproject: dict[str, Any],
    selected_extras: list[str],
) -> tuple[str, list[str]]:
    project = pyproject.get("project", {})
    project_name = str(project.get("name", "")).strip()
    deps = list(project.get("dependencies", []) or [])
    optional = dict(project.get("optional-dependencies", {}) or {})
    project_norm = normalize_name(project_name)
    queue = [e.strip() for e in selected_extras if e.strip()]
    visited_extras: set[str] = set()
    selected_requirements: list[str] = list(deps)

    while queue:
        extra = queue.pop(0)
        if extra in visited_extras:
            continue
        visited_extras.add(extra)
        for req in optional.get(extra, []):
            selected_requirements.append(req)
            req_name = parse_requirement_name(req)
            if req_name != project_norm:
                continue
            for nested in parse_requirement_extras(req):
                if nested not in visited_extras:
                    queue.append(nested)

    return project_name, selected_requirements


def collect_dependency_closure(
    root_names: set[str],
) -> tuple[set[str], set[str], dict[str, importlib_metadata.Distribution]]:
    installed: dict[str, importlib_metadata.Distribution] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name") or ""
        key = normalize_name(name)
        if key and key not in installed:
            installed[key] = dist

    visited: set[str] = set()
    missing: set[str] = set()
    queue = sorted(root_names)

    while queue:
        name = queue.pop(0)
        if name in visited or name in missing:
            continue
        dist = installed.get(name)
        if dist is None:
            missing.add(name)
            continue
        visited.add(name)
        for req in dist.requires or []:
            child = parse_requirement_name(req)
            if child and child not in visited and child not in missing:
                queue.append(child)

    return visited, missing, installed


def extract_license(dist: importlib_metadata.Distribution) -> str:
    meta = dist.metadata
    license_raw = (meta.get("License") or "").strip()
    if license_raw and license_raw.upper() not in {"UNKNOWN", "NONE"}:
        compact = " ".join(license_raw.split())
        if len(compact) > 120:
            compact = compact[:117] + "..."
        return compact

    classifiers = meta.get_all("Classifier", [])
    license_classifiers = [c for c in classifiers if c.startswith("License :: ")]
    if license_classifiers:
        return " | ".join(license_classifiers)
    return "UNKNOWN"


def extract_homepage(dist: importlib_metadata.Distribution) -> str:
    meta = dist.metadata
    homepage = (meta.get("Home-page") or "").strip()
    if homepage:
        return " ".join(homepage.split())
    for entry in meta.get_all("Project-URL", []):
        parts = [p.strip() for p in entry.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            return " ".join(parts[1].split())
    return "-"


def build_markdown(
    *,
    pyproject_path: Path,
    extras: list[str],
    installed_names: set[str],
    missing_names: set[str],
    installed_map: dict[str, importlib_metadata.Distribution],
) -> str:
    ts = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    extras_text = ",".join(extras) if extras else "(none)"
    lines = [
        "# Third-Party Notices",
        "",
        "This file is generated from dependency metadata.",
        "",
        f"- Generated at: {ts}",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Source: `{pyproject_path}`",
        f"- Selected extras: `{extras_text}`",
        "",
        "## Inventory",
        "",
        "| Package | Version | License | Home page | Status |",
        "|---|---:|---|---|---|",
    ]

    for name in sorted(installed_names):
        dist = installed_map[name]
        package_name = dist.metadata.get("Name", name)
        version = dist.version
        license_text = extract_license(dist).replace("|", "\\|")
        homepage = extract_homepage(dist).replace("|", "\\|")
        lines.append(
            f"| {package_name} | {version} | {license_text} | {homepage} | installed |"
        )

    for name in sorted(missing_names):
        lines.append(f"| {name} | - | - | - | missing |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- License/homepage values come from package metadata and may be incomplete.",
            "- `missing` means declared in selected dependency closure but not installed in the current environment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate THIRD_PARTY_NOTICES.md from pyproject + installed packages.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("THIRD_PARTY_NOTICES.md"),
        help="Output markdown path",
    )
    parser.add_argument(
        "--extras",
        type=str,
        default="sym,plot,speed,jax",
        help="Comma-separated extras to include (default: sym,plot,speed,jax)",
    )
    args = parser.parse_args()

    if not args.pyproject.exists():
        raise SystemExit(f"pyproject file not found: {args.pyproject}")

    pyproject = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    project_name, requirements = discover_selected_requirements(pyproject, extras)
    root_names = {
        name
        for req in requirements
        if (name := parse_requirement_name(req)) is not None
        and name != normalize_name(project_name)
    }

    installed, missing, installed_map = collect_dependency_closure(root_names)
    markdown = build_markdown(
        pyproject_path=args.pyproject,
        extras=extras,
        installed_names=installed,
        missing_names=missing,
        installed_map=installed_map,
    )
    args.output.write_text(markdown, encoding="utf-8")
    print(
        f"Generated {args.output} with {len(installed)} installed and "
        f"{len(missing)} missing packages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
