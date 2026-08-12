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
DEEPSEEK_SOURCE_PACKAGE_PATH = "@earendil-works/pi-ai/dist/providers/data/deepseek.json"
ZAI_SOURCE_PACKAGE_PATH = "@earendil-works/pi-ai/dist/providers/data/zai.json"
ZAI_CODING_CN_SOURCE_PACKAGE_PATH = (
    "@earendil-works/pi-ai/dist/providers/data/zai-coding-cn.json"
)
SOURCE_API_KEY = "openai-completions"
DEEPSEEK_MODEL_IDS = ("deepseek-v4-flash", "deepseek-v4-pro")
ZAI_MODEL_SPECS = (
    ("glm-4.7", "GLM-4.7", 204_800),
    ("glm-5-turbo", "GLM-5-Turbo", 200_000),
    ("glm-5.2", "GLM-5.2", 1_000_000),
    ("glm-5.2-highspeed", "GLM-5.2 Highspeed", 1_000_000),
)
ZAI_REGIONS = (
    ("zai", "zai.json", "https://api.z.ai/api/coding/paas/v4"),
    (
        "zai-coding-cn",
        "zai-coding-cn.json",
        "https://open.bigmodel.cn/api/coding/paas/v4",
    ),
)

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


def build_deepseek_models(source_path: Path) -> list[dict[str, Any]]:
    deepseek_path = source_path.with_name("deepseek.json")
    if not deepseek_path.is_file():
        raise FileNotFoundError(f"missing pinned DeepSeek data next to OpenRouter source: {deepseek_path}")
    raw = load_json(deepseek_path)
    source_models = raw.get(SOURCE_API_KEY) if isinstance(raw, dict) else None
    if not isinstance(source_models, dict):
        raise ValueError(f"DeepSeek source must contain object key {SOURCE_API_KEY!r}")
    if tuple(source_models) != DEEPSEEK_MODEL_IDS:
        raise ValueError(
            f"expected exact DeepSeek model set/order {DEEPSEEK_MODEL_IDS}, "
            f"found {tuple(source_models)}"
        )

    models: list[dict[str, Any]] = []
    expected_common = {
        "provider": "deepseek",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "reasoning": True,
        "contextWindow": 1_000_000,
        "maxTokens": 384_000,
    }
    expected_map = {
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "max": "max",
    }
    for model_id in DEEPSEEK_MODEL_IDS:
        source = source_models[model_id]
        if not isinstance(source, dict):
            raise ValueError(f"{model_id}: DeepSeek source model must be an object")
        validate_source_model(model_id, source)
        for key, expected in expected_common.items():
            if source.get(key) != expected:
                raise ValueError(f"{model_id}: expected {key}={expected!r}, found {source.get(key)!r}")
        if source.get("id") != model_id:
            raise ValueError(f"{model_id}: source id mismatch")
        if source.get("thinkingLevelMap") != expected_map:
            raise ValueError(f"{model_id}: thinkingLevelMap drifted")
        compat = source.get("compat", {})
        expected_compat = {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
        }
        if compat != expected_compat:
            raise ValueError(f"{model_id}: compat drifted: {compat!r}")
        models.append(
            {
                "provider": source["provider"],
                "id": source["id"],
                "name": source["name"],
                "api": source["api"],
                "baseUrl": source["baseUrl"],
                "reasoning": source["reasoning"],
                "contextWindow": source["contextWindow"],
                "maxTokens": source["maxTokens"],
                "reasoningEfforts": ["high", "max"],
                "thinkingLevelMap": dict(source["thinkingLevelMap"]),
                "compat": {
                    **compat,
                    "supportsReasoningEffort": True,
                    "maxTokensField": "max_completion_tokens",
                },
            }
        )
    return models


def build_zai_models(source_path: Path) -> list[dict[str, Any]]:
    expected_ids = tuple(model_id for model_id, _, _ in ZAI_MODEL_SPECS)
    expected_common_compat = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "zai",
        "zaiToolStream": True,
    }
    expected_glm52_map = {
        "minimal": None,
        "low": "high",
        "medium": "high",
        "high": "high",
        "max": "max",
    }
    models: list[dict[str, Any]] = []
    # Z.AI's four model definitions are a single shared source contract. Each
    # regional pi-ai file is validated against it before the shared definition
    # is expanded into provider-specific catalog rows and sampler facts.
    for provider, filename, base_url in ZAI_REGIONS:
        region_path = source_path.with_name(filename)
        if not region_path.is_file():
            raise FileNotFoundError(f"missing pinned Z.AI data next to OpenRouter source: {region_path}")
        raw = load_json(region_path)
        source_models = raw.get(SOURCE_API_KEY) if isinstance(raw, dict) else None
        if not isinstance(source_models, dict):
            raise ValueError(f"{filename} must contain object key {SOURCE_API_KEY!r}")
        if tuple(source_models) != expected_ids:
            raise ValueError(
                f"expected exact shared Z.AI model set/order {expected_ids}, "
                f"found {tuple(source_models)} in {filename}"
            )
        for model_id, name, context_window in ZAI_MODEL_SPECS:
            source = source_models[model_id]
            if not isinstance(source, dict):
                raise ValueError(f"{filename}:{model_id}: source model must be an object")
            validate_source_model(model_id, source)
            expected = {
                "provider": provider,
                "id": model_id,
                "name": name,
                "api": "openai-completions",
                "baseUrl": base_url,
                "reasoning": True,
                "contextWindow": context_window,
                "maxTokens": 131_072,
            }
            for key, value in expected.items():
                if source.get(key) != value:
                    raise ValueError(
                        f"{filename}:{model_id}: expected {key}={value!r}, "
                        f"found {source.get(key)!r}"
                    )
            supports_effort = model_id == "glm-5.2"
            expected_compat = {
                **expected_common_compat,
                "supportsReasoningEffort": supports_effort,
            }
            if source.get("compat") != expected_compat:
                raise ValueError(f"{filename}:{model_id}: compat drifted: {source.get('compat')!r}")
            expected_map = expected_glm52_map if supports_effort else None
            if source.get("thinkingLevelMap") != expected_map:
                raise ValueError(f"{filename}:{model_id}: thinkingLevelMap drifted")
            model = {
                **expected,
                "reasoningEfforts": (
                    ["low", "medium", "high", "max"]
                    if supports_effort
                    else ["minimal", "low", "medium", "high"]
                ),
                "compat": expected_compat,
            }
            if expected_map is not None:
                model["thinkingLevelMap"] = dict(expected_map)
            models.append(model)
    return models


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
    deepseek_models = build_deepseek_models(source_path)
    zai_models = build_zai_models(source_path)
    original_members = [(entry.get("provider"), entry.get("id")) for entry in catalog]
    generated_models = deepseek_models + zai_models
    generated_providers = {"deepseek", "zai", "zai-coding-cn"}
    existing_generated = [
        entry for entry in catalog if entry.get("provider") in generated_providers
    ]
    expected_generated_members = [
        (entry["provider"], entry["id"]) for entry in generated_models
    ]
    if existing_generated:
        existing_members = [
            (entry.get("provider"), entry.get("id")) for entry in existing_generated
        ]
        if existing_members not in (
            [("deepseek", model_id) for model_id in DEEPSEEK_MODEL_IDS],
            expected_generated_members,
        ):
            raise AssertionError("generated provider catalog member set/order drifted")
        if catalog[-len(existing_generated):] != existing_generated:
            raise AssertionError("generated provider models must remain append-only at catalog tail")
        catalog[-len(existing_generated):] = generated_models
    else:
        catalog.extend(generated_models)
    expected_members = [
        member for member in original_members if member[0] not in generated_providers
    ] + expected_generated_members

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

    facts.extend(
        {
            "provider": entry["provider"],
            "id": entry["id"],
            "reasoning": entry["reasoning"],
            **(
                {"thinkingLevelMap": entry["thinkingLevelMap"]}
                if "thinkingLevelMap" in entry
                else {}
            ),
            "compat": entry["compat"],
        }
        for entry in generated_models
    )

    if [(entry.get("provider"), entry.get("id")) for entry in catalog] != expected_members:
        raise AssertionError("sync changed existing catalog members or non-append ordering")

    manifest = {
        "schemaVersion": 1,
        "source": {
            "package": "@earendil-works/pi-ai",
            "version": PI_AI_VERSION,
            "dataFile": SOURCE_PACKAGE_PATH,
            "deepSeekDataFile": DEEPSEEK_SOURCE_PACKAGE_PATH,
            "zaiDataFile": ZAI_SOURCE_PACKAGE_PATH,
            "zaiCodingCnDataFile": ZAI_CODING_CN_SOURCE_PACKAGE_PATH,
            "generatedAt": SOURCE_GENERATED_AT,
            "api": SOURCE_API_KEY,
        },
        "counts": {
            "catalogMembers": len(catalog),
            "catalogOpenRouterMembers": len(openrouter_entries),
            "sourceOpenRouterMembers": len(source_models),
            "matchingOpenRouterMembers": len(matching_ids),
            "generatedWireFacts": len(facts),
            "catalogDeepSeekMembers": len(deepseek_models),
            "generatedDeepSeekWireFacts": len(deepseek_models),
            "catalogZaiMembers": len(zai_models),
            "generatedZaiWireFacts": len(zai_models),
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
