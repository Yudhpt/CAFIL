"""YAML 配置加载、严格覆盖和路径引用解析。"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


_PATH_REFERENCE = re.compile(r"\$\{paths\.([A-Za-z0-9_.-]+)\}")
_ENV_REFERENCE = re.compile(r"[$]\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for CAFIL configuration loading. Please install `pyyaml`."
        ) from exc
    return yaml


def _read_yaml(path: Path) -> Dict[str, Any]:
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings; values in ``update`` always take precedence."""
    merged = deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_with_defaults(cfg_path: Path) -> Dict[str, Any]:
    """Merge YAML ``defaults`` left-to-right, then apply the current YAML file."""
    return _load_with_defaults(cfg_path.resolve(), stack=())


def _load_with_defaults(cfg_path: Path, *, stack: Sequence[Path]) -> Dict[str, Any]:
    """Recursive defaults loader with explicit cycle and missing-file errors."""
    if cfg_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, cfg_path))
        raise ValueError(f"Config defaults cycle detected: {cycle}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Default config file not found: {cfg_path}")

    payload = _read_yaml(cfg_path)
    defaults = payload.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ValueError(f"`defaults` must be a list: {cfg_path}")

    merged: Dict[str, Any] = {}
    for item in defaults:
        if not isinstance(item, str):
            raise ValueError(f"defaults entries must be strings: {cfg_path}")
        child = (cfg_path.parent / item).resolve()
        merged = deep_merge(merged, _load_with_defaults(child, stack=(*stack, cfg_path)))
    return deep_merge(merged, payload)


def _parse_override(raw: str) -> Tuple[str, Any]:
    """Parse ``key=value`` with YAML typing for values."""
    if "=" not in raw:
        raise ValueError(f"Override must be key=value: {raw}")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key is empty: {raw}")
    return key, _require_yaml().safe_load(raw_value)


def apply_overrides_strict(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply dot-path overrides only to existing leaf keys."""
    out = deepcopy(cfg)
    for raw in overrides:
        dotted_key, value = _parse_override(raw)
        keys: List[str] = dotted_key.split(".")
        cursor: Dict[str, Any] = out
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                raise KeyError(f"Unknown override path: {dotted_key}")
            cursor = cursor[key]
        leaf = keys[-1]
        if leaf not in cursor:
            raise KeyError(f"Unknown override key: {dotted_key}")
        cursor[leaf] = value
    return out


def _lookup_path_value(paths: Dict[str, Any], dotted_key: str) -> Any:
    """Look up a scalar value under the top-level ``paths`` mapping."""
    value: Any = paths
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Unknown paths reference: ${{paths.{dotted_key}}}")
        value = value[key]
    return value


def resolve_path_references(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand strict ``${env:NAME}`` and ``${paths.key}`` references.

    Environment references are resolved first and must name a non-empty
    variable. This keeps public configurations portable without silently
    falling back to a machine-specific data or checkpoint directory.
    """
    resolved_cfg = deepcopy(cfg)
    paths = resolved_cfg.get("paths", {})
    if paths is None:
        paths = {}
    if not isinstance(paths, dict):
        raise ValueError("`paths` must be a mapping when present")

    resolving: set[str] = set()
    resolved_paths: Dict[str, Any] = {}

    def resolve_path(dotted_key: str) -> Any:
        if dotted_key in resolved_paths:
            return resolved_paths[dotted_key]
        if dotted_key in resolving:
            raise ValueError(f"Config paths reference cycle detected at: paths.{dotted_key}")
        resolving.add(dotted_key)
        try:
            value = resolve_value(_lookup_path_value(paths, dotted_key))
            resolved_paths[dotted_key] = value
            return value
        finally:
            resolving.remove(dotted_key)

    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            def replace_env(match: re.Match[str]) -> str:
                variable = match.group(1)
                resolved = os.environ.get(variable, "").strip()
                if not resolved:
                    raise ValueError(
                        f"Required environment variable is not set: {variable}. "
                        "See README.md for CAFIL_DATA_ROOT, CAFIL_OUTPUT_ROOT, and CAFIL_DINO_HOME."
                    )
                return resolved

            def replace(match: re.Match[str]) -> str:
                reference = match.group(1)
                target = resolve_path(reference)
                if isinstance(target, (dict, list)):
                    raise ValueError(
                        f"Path reference must resolve to a scalar: ${{paths.{reference}}}"
                    )
                return str(target)
            return _PATH_REFERENCE.sub(replace, _ENV_REFERENCE.sub(replace_env, value))
        if isinstance(value, dict):
            return {key: resolve_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value

    for key in paths:
        resolve_path(str(key))
    resolved_cfg["paths"] = resolve_value(paths)
    return resolve_value(resolved_cfg)


def load_config(cfg_path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load a CAFIL YAML file through one strict and reproducible resolution path.

    Resolution order is defaults merge, strict CLI override, then environment
    and ``${paths.*}`` interpolation. Consequently, an override of ``paths``
    propagates to all consumers.
    """
    path = Path(cfg_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    cfg = apply_overrides_strict(load_with_defaults(path), overrides)
    return resolve_path_references(cfg)


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write resolved YAML without sorting keys, preserving readable diffs."""
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
