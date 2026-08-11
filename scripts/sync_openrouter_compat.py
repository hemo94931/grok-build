#!/usr/bin/env python3
"""Sync OpenRouter compatibility facts from pi-ai into shell and sampler artifacts.

The model member set and ordering in the shell catalog are preserved. Only existing
OpenRouter entries whose IDs are present in the pinned pi-ai source are updated.
Run with --check in CI/tests to verify that committed artifacts are current.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PI_AI_VERSION = "0.84.1"
SOURCE_GENERATED_AT = "2026-08-07T05:53:06.539Z"
SOURCE_PACKAGE_PATH = "@earendil-works/pi-ai/dist/providers/data/openrouter.json"
SOURCE_API_KEY = "openai-completions"

SUPPORTED_COMPAT_KEYS = {
    "supportsStore",
    "supportsDeveloperRole",
    "supportsReasoningEffort",
    "maxTokensField",
    "requiresReasoningContentOnAssistantMessages",
    "thinkingFormat",
    "zaiToolStream",
}
REGISTERED_IGNORED_COMPAT_KEYS = {
    "cacheControlFormat": (
        "Registered for schema/drift visibility; prompt cache-control conversion "
        "is outside ticket 02 and is not consumed by the sampler body adapter."
    )
}
ALL_COMPAT_KEYS = SUPPORTED_COMPAT_KEYS | set(REGISTERED_IGNORED_COMPAT_KEYS)
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
WIRE_EFFORT = {"off": "none"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_default_source() -> Path:
    override = os.environ.get("PI_AI_OPENROUTER_DATA")
    if override:
        return Path(override).expanduser().resolve()

    relative = Path(SOURCE_PACKAGE_PATH)
    candidates = [
        Path.home() / ".nvm/versions/node",
        Path("/usr/local/lib/node_modules"),
        Path("/usr/lib/node_modules"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        direct = root / relative
        if direct.is_file():
            return direct.resolve()
        nested = root / "@earendil-works/pi-coding-agent/node_modules" / relative
        if nested.is_file():
            return nested.resolve()
        nested_pattern = (
            "*/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/"
            "@earendil-works/pi-ai/dist/providers/data/openrouter.json"
        )
        matches = sorted(root.glob(nested_pattern), reverse=True)
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        "could not locate pinned pi-ai OpenRouter data; pass --source or set "
        "PI_AI_OPENROUTER_DATA"
    )


def validate_source_version(source_path: Path) -> None:
    package_json = source_path.parents[3] / "package.json"
    if not package_json.is_file():
        return
    package = load_json(package_json)
    actual = package.get("version") if isinstance(package, dict) else None
    if actual != PI_AI_VERSION:
        raise ValueError(
            f"expected @earendil-works/pi-ai {PI_AI_VERSION}, found {actual!r}"
        )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def render_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def validate_source_model(model_id: str, model: dict[str, Any]) -> None:
    compat = model.get("compat", {})
    if not isinstance(compat, dict):
        raise ValueError(f"{model_id}: compat must be an object")
    unknown = sorted(set(compat) - ALL_COMPAT_KEYS)
    if unknown:
        raise ValueError(
            f"{model_id}: unknown compat keys {unknown}; consume them or register "
            "them explicitly in sync_openrouter_compat.py"
        )
    thinking_map = model.get("thinkingLevelMap", {})
    if not isinstance(thinking_map, dict):
        raise ValueError(f"{model_id}: thinkingLevelMap must be an object")
    for level, mapped in thinking_map.items():
        if level not in THINKING_LEVELS:
            raise ValueError(f"{model_id}: unknown thinking level {level!r}")
        if mapped is not None and not isinstance(mapped, str):
            raise ValueError(
                f"{model_id}: thinkingLevelMap.{level} must be string or null"
            )


def supported_efforts(model: dict[str, Any]) -> list[str]:
    if not model.get("reasoning", False):
        return ["none"]
    thinking_map = model.get("thinkingLevelMap", {})
    efforts: list[str] = []
    for level in THINKING_LEVELS:
        mapped = thinking_map.get(level, "__missing__")
        if mapped is None:
            continue
        if level in {"xhigh", "max"} and mapped == "__missing__":
            continue
        efforts.append(WIRE_EFFORT.get(level, level))
    return efforts


def build_outputs(source_path: Path) -> tuple[str, str, str]:
    validate_source_version(source_path)
    root = repo_root()
    catalog_path = root / "crates/codegen/xai-grok-shell/src/auth/providers/models.json"
    catalog = load_json(catalog_path)
    if not isinstance(catalog, list):
        raise ValueError("shell provider catalog must be a JSON array")
    original_members = [(entry.get("provider"), entry.get("id")) for entry in catalog]

    raw_source = load_json(source_path)
    source_models = raw_source.get(SOURCE_API_KEY)
    if not isinstance(source_models, dict):
        raise ValueError(f"source must contain object key {SOURCE_API_KEY!r}")
    for model_id, model in source_models.items():
        if not isinstance(model, dict):
            raise ValueError(f"{model_id}: source model must be an object")
        validate_source_model(model_id, model)

    openrouter_entries = [entry for entry in catalog if entry.get("provider") == "openrouter"]
    catalog_ids = {entry["id"] for entry in openrouter_entries}
    source_ids = set(source_models)
    matching_ids = catalog_ids & source_ids

    facts: list[dict[str, Any]] = []
    for entry in catalog:
        if entry.get("provider") != "openrouter" or entry.get("id") not in matching_ids:
            continue
        source = source_models[entry["id"]]
        entry["reasoning"] = bool(source.get("reasoning", False))
        entry["reasoningEfforts"] = supported_efforts(source)
        compat = source.get("compat", {})
        entry["compat"] = dict(compat)
        if "thinkingLevelMap" in source:
            entry["thinkingLevelMap"] = dict(source["thinkingLevelMap"])
        else:
            entry.pop("thinkingLevelMap", None)

        fact: dict[str, Any] = {
            "provider": "openrouter",
            "id": entry["id"],
            "reasoning": entry["reasoning"],
        }
        if "thinkingLevelMap" in entry:
            fact["thinkingLevelMap"] = entry["thinkingLevelMap"]
        if compat:
            fact["compat"] = compat
        facts.append(fact)

    if [(entry.get("provider"), entry.get("id")) for entry in catalog] != original_members:
        raise AssertionError("sync changed the provider catalog member set or ordering")

    manifest = {
        "schemaVersion": 1,
        "source": {
            "package": "@earendil-works/pi-ai",
            "version": PI_AI_VERSION,
            "dataFile": SOURCE_PACKAGE_PATH,
            "generatedAt": SOURCE_GENERATED_AT,
            "api": SOURCE_API_KEY,
        },
        "counts": {
            "catalogMembers": len(catalog),
            "catalogOpenRouterMembers": len(openrouter_entries),
            "sourceOpenRouterMembers": len(source_models),
            "matchingOpenRouterMembers": len(matching_ids),
            "generatedWireFacts": len(facts),
            "catalogOnlyOpenRouterMembers": len(catalog_ids - source_ids),
            "sourceOnlyOpenRouterMembers": len(source_ids - catalog_ids),
        },
        "drift": {
            "catalogOnlyOpenRouterIds": sorted(catalog_ids - source_ids),
            "sourceOnlyOpenRouterIds": sorted(source_ids - catalog_ids),
        },
        "compatKeys": {
            "consumed": sorted(SUPPORTED_COMPAT_KEYS),
            "registeredIgnored": REGISTERED_IGNORED_COMPAT_KEYS,
        },
    }
    return render_json(catalog), render_json(facts), render_json(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    source_path = args.source.expanduser().resolve() if args.source else discover_default_source()
    outputs = {
        root / "crates/codegen/xai-grok-shell/src/auth/providers/models.json": None,
        root
        / "crates/codegen/xai-grok-sampler/src/provider_wire/generated_wire_facts.json": None,
        root / "scripts/openrouter_compat_manifest.json": None,
    }
    catalog, facts, manifest = build_outputs(source_path)
    rendered = [catalog, facts, manifest]
    stale: list[Path] = []
    for (path, _), content in zip(outputs.items(), rendered, strict=True):
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    if stale:
        for path in stale:
            print(f"stale generated artifact: {path.relative_to(root)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
