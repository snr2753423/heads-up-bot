"""Static facts about a public agent.py. Inferences are labeled as such."""

from __future__ import annotations

import ast


def _returns(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "Action":
                names.append(node.value.attr)
    return names


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _dict_numbers(node: ast.AST) -> dict:
    """Pull simple numeric constants out of a dict, including short lists."""
    found = {}
    if not isinstance(node, ast.Dict):
        return found
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            found[key.value] = value.value
        elif isinstance(value, ast.List):
            nums = []
            for item in value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
                    nums.append(item.value)
                else:
                    nums = []
                    break
            if nums:
                found[key.value] = nums
    return found


def analyze_agent(source: str, extra_sources: dict[str, str] | None = None) -> dict:
    """Return mechanically checked facts plus separately marked inferences."""
    extra_sources = extra_sources or {}
    facts = []
    inferences = []
    archetype = "unknown"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "archetype": "unparsed",
            "facts": [f"agent.py did not parse: {exc}"],
            "inferences": [],
            "params": {},
        }
    choose = _function(tree, "choose_action")
    returns = _returns(choose) if choose else []
    if choose is None:
        facts.append("agent.py has no choose_action function")
    else:
        facts.append("choose_action returns: " + (", ".join(returns) or "no direct Action return"))
    text = source
    if (
        returns
        and set(returns) <= {"CHECK_CALL", "FOLD"}
        and "to_call == 0" in text
        and "RAISE_POT" not in "".join(returns)
    ):
        archetype = "starter_checkfold"
        facts.append("The only actions returned by choose_action are check/call and fold, and a free check is taken.")
    if "return Action.CHECK_CALL" in text and "RAISE_" not in text and "random" not in text.lower():
        if archetype == "unknown":
            archetype = "calling_station"
            facts.append("Source returns CHECK_CALL and never raises.")
    params = {}
    for filename, body in extra_sources.items():
        try:
            extra_tree = ast.parse(body)
        except SyntaxError:
            inferences.append(f"{filename} did not parse")
            continue
        for node in ast.walk(extra_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DEFAULT_PARAMS":
                        params = _dict_numbers(node.value)
                        facts.append(f"{filename} defines DEFAULT_PARAMS with {len(params)} numeric entries")
    if params:
        archetype = "equity_threshold"
        raise_eq = params.get("raise_eq")
        bluff = params.get("bluff_freq")
        steal = params.get("steal_freq")
        if isinstance(raise_eq, list):
            facts.append(f"Value-raise equity thresholds by street are {raise_eq}.")
        if bluff == 0 and steal == 0:
            facts.append("bluff_freq and steal_freq are both 0.")
        elif isinstance(bluff, (int, float)) and bluff <= 0.15:
            inferences.append(
                "Bluffs are rare in the published parameters. A small increase in bets when checked to is the counter, and it is an inference until a match confirms the behavior."
            )
        if isinstance(params.get("call_margin"), list):
            facts.append(f"Call margins by street are {params['call_margin']}.")
    if archetype == "unknown" and "omaha_score" in text:
        inferences.append("The file calls omaha_score, but the betting rule was not identified mechanically.")
    return {
        "archetype": archetype,
        "facts": facts,
        "inferences": inferences,
        "params": params,
    }
