"""Public, deterministic API linter for v0.1.5 candidate code.

The linter reports only syntax, name, signature, import, and schema mistakes.
It never runs the evaluator or reveals hidden objective values.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import builtins
from typing import Iterable


@dataclass(frozen=True)
class LintIssue:
    code: str
    message: str
    line: int = 0

    def payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "line": self.line}


ALLOWED_MODULES = {"execution_sdk", "math", "numpy", "random"}
PUBLIC_SDK_NAMES = {"Budget", "project_box"}
SDK_SIGNATURES = {"project_box": ("values", "bounds"), "Budget": ("limit",),
                  "Budget.consume": ("amount",), "initialize_seed": ("seed",),
                  "finite_vector": ("values", "dimension")}
SAFE_BUILTINS = set(dir(builtins))


def _line(node: ast.AST) -> int:
    return int(getattr(node, "lineno", 0))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id in {"execution_sdk", "sdk"}:
            return node.func.attr
        if node.func.attr == "consume":
            return "Budget.consume"
    return None


def _bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


def _undefined_names(function: ast.FunctionDef, globals_bound: set[str]) -> Iterable[LintIssue]:
    local = _bound_names(function)
    local.update(arg.arg for arg in (*function.args.posonlyargs, *function.args.args,
                                     *function.args.kwonlyargs))
    local.update(globals_bound)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local and node.id not in SAFE_BUILTINS:
            yield LintIssue("undefined_name", f"name '{node.id}' is not defined", _line(node))


def lint_code(source: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [LintIssue("syntax_error", error.msg, int(error.lineno or 0))]

    solve_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    if len(solve_nodes) != 1:
        issues.append(LintIssue("solve_definition", "code must define exactly one top-level solve", 0))
    else:
        solve = solve_nodes[0]
        args = [arg.arg for arg in (*solve.args.posonlyargs, *solve.args.args, *solve.args.kwonlyargs)]
        if args != ["problem", "seed", "budget"] or solve.args.vararg or solve.args.kwarg or solve.args.defaults or solve.args.kw_defaults:
            issues.append(LintIssue("solve_signature", "solve must have exactly solve(problem, seed, budget)", _line(solve)))
        issues.extend(_undefined_names(solve, _bound_names(tree) - {"solve"}))

    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    issues.append(LintIssue("forbidden_import", f"module '{alias.name}' is not in the frozen SDK allowlist", _line(node)))
                imported[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module).split(".")[0]
            if root not in ALLOWED_MODULES:
                issues.append(LintIssue("forbidden_import", f"module '{node.module}' is not in the frozen SDK allowlist", _line(node)))
            for alias in node.names:
                if alias.name == "*":
                    issues.append(LintIssue("wildcard_import", "wildcard imports are not allowed", _line(node)))
                imported[alias.asname or alias.name] = f"{root}.{alias.name}"

    if any(isinstance(node, ast.Name) and node.id == "np" and isinstance(node.ctx, ast.Load)
           for node in ast.walk(tree)) and imported.get("np") != "numpy":
        issues.append(LintIssue("numpy_alias_missing", "np is used but 'import numpy as np' is missing", 0))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in SDK_SIGNATURES:
                parameters = SDK_SIGNATURES[name]
                required = 0 if name == "Budget.consume" else len(parameters)
                supplied = len(node.args) + len(node.keywords)
                keywords = {item.arg for item in node.keywords}
                invalid_keywords = None in keywords or not keywords.issubset(parameters)
                if not required <= supplied <= len(parameters) or invalid_keywords:
                    issues.append(LintIssue("sdk_arity", f"{name} expects signature ({', '.join(parameters)})", _line(node)))
            if name in {"initialize_seed", "finite_vector"}:
                issues.append(LintIssue("non_public_sdk_api", f"{name} is wrapper-owned; use only Budget/project_box", _line(node)))
        if isinstance(node, ast.Attribute) and node.attr in {"initialize_seed", "finite_vector"}:
            issues.append(LintIssue("non_public_sdk_api", f"{node.attr} is wrapper-owned; use only Budget/project_box", _line(node)))

    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    schema_dict_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(value, ast.Dict):
                keys = {key.value for key in value.keys if isinstance(key, ast.Constant)}
                if {"point", "oracle_calls"}.issubset(keys):
                    schema_dict_names.update(target.id for target in targets if isinstance(target, ast.Name))
    valid_return = False
    for node in returns:
        if isinstance(node.value, ast.Dict):
            keys = {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
            if {"point", "oracle_calls"}.issubset(keys):
                valid_return = True
        elif isinstance(node.value, ast.Name) and node.value.id in schema_dict_names:
            valid_return = True
    if not valid_return:
        issues.append(LintIssue("return_schema", "a return dict must contain point and oracle_calls", 0))

    # Stable order and de-duplication make public repair feedback reproducible.
    unique: dict[tuple[str, str, int], LintIssue] = {(item.code, item.message, item.line): item for item in issues}
    return sorted(unique.values(), key=lambda item: (item.line, item.code, item.message))


def lint_or_raise(source: str) -> None:
    issues = lint_code(source)
    if issues:
        joined = "; ".join(f"{item.code}@{item.line}: {item.message}" for item in issues[:8])
        raise ValueError(joined)
