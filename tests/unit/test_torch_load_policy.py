from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY = Path(__file__).parents[2]


def test_supported_user_artifact_loads_declare_device_and_pickle_policy() -> None:
    files = [
        *sorted((REPOSITORY / "wisdom").rglob("*.py")),
        *sorted((REPOSITORY / "coverage_methods").rglob("*.py")),
        *sorted(REPOSITORY.glob("wisdom_*_train.py")),
    ]
    violations: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "load"
            ):
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            missing = {"map_location", "weights_only"} - keywords
            if missing:
                violations.append(
                    f"{path.relative_to(REPOSITORY)}:{node.lineno} missing {sorted(missing)}"
                )
    assert violations == []
