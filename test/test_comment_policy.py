from __future__ import annotations

import ast
import io
from pathlib import Path
import re
import tokenize


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"修复|旧|兼容|较新的|已提升到|新运行|原 C1|"
    r"上(?:一)?版本|前一版本|新版|新版本|"
    r"\blegacy\b|\bdeprecated\b|\bnewer\b|\bcurrently\b|\bpreviously\b|\bno longer\b|previous version|old version|new version",
    re.IGNORECASE,
)
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "results"}


def _annotation_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    violations: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT and FORBIDDEN.search(token.string):
            violations.append(f"{path.relative_to(ROOT)}:{token.start[0]} comment")

    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring and FORBIDDEN.search(docstring):
            line = node.body[0].lineno if node.body else getattr(node, "lineno", 1)
            violations.append(f"{path.relative_to(ROOT)}:{line} docstring")
    return violations


def test_source_annotations_describe_current_behavior() -> None:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        violations.extend(_annotation_violations(path))
    assert violations == []
