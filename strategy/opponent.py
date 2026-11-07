"""Half-scoped opponent frequencies. Priors keep the first hands on the default policy."""

from __future__ import annotations

from evaluator import STRAIGHT, TWO_PAIR, category_of, omaha_score
from game import Action, HandResult, Observation, Street

# Pseudocounts. They do not count toward the "enough real hands" gates.
_FOLD_PRIOR = 8
_CONTINUE_PRIOR = 12
_RAISE_PRIOR = 6
_NON_RAISE_PRIOR = 14
_WEAK_PRIOR = 3
_STRONG_PRIOR = 5

_MIN_ACTION_OBS = 25
_MIN_SHOWDOWNS = 15


def init_memory(memory: dict) -> None:
    memory.clear()
    memory["hands_played"] = 0
    memory["hole"] = None
    memory["hand_index"] = None
    memory["cursor_hand"] = None
    memory["cursor"] = 0
    memory["last_street"] = None
    memory["opp"] = {
        "pre": {"fold": 0, "continue": 0, "raise": 0, "non_raise": 0},
        "post": {"fold": 0, "continue": 0, "raise": 0, "non_raise": 0},
    }
    memory["show_weak"] = 0
    memory["show_strong"] = 0
    memory["show_n"] = 0
    memory["eq_cache"] = None


def _bucket(street: Street) -> str:
    return "pre" if street == Street.PREFLOP else "post"


def _bump(stats: dict, kind: str) -> None:
    stats[kind] = stats.get(kind, 0) + 1


def observe_actions(obs: Observation, memory: dict) -> None:
    """Count opponent actions we have not counted yet this hand."""
    if memory.get("cursor_hand") != obs.hand_index:
        memory["cursor_hand"] = obs.hand_index
        memory["cursor"] = 0
    start = int(memory.get("cursor") or 0)
    history = obs.action_history
    opp = memory.setdefault("opp", {"pre": {}, "post": {}})
    for action in history[start:]:
        if action.is_me:
            continue
        stats = opp[_bucket(action.street)]
        if action.action == Action.FOLD:
            _bump(stats, "fold")
            _bump(stats, "non_raise")
        elif action.action in (Action.RAISE_HALF, Action.RAISE_POT):
            _bump(stats, "raise")
            _bump(stats, "continue")
        elif action.action == Action.CHECK_CALL and action.amount > 0:
            _bump(stats, "call")
            _bump(stats, "continue")
            _bump(stats, "non_raise")
        elif action.action == Action.CHECK_CALL:
            _bump(stats, "non_raise")
    memory["cursor"] = len(history)
    memory["last_street"] = obs.street


def note_our_action(obs: Observation, action: Action, memory: dict) -> None:
    memory["we_bet"] = action in (Action.RAISE_HALF, Action.RAISE_POT)
    memory["last_street"] = obs.street


def observe_showdown(result: HandResult, memory: dict) -> None:
    """Label showdowns, and count a fold that ended the hand before we saw it."""
    if "opp" not in memory:
        return
    if not result.went_to_showdown and result.winner == "me" and memory.get("we_bet"):
        street = memory.get("last_street")
        if street is not None:
            stats = memory["opp"][_bucket(street)]
            _bump(stats, "fold")
            _bump(stats, "non_raise")
    if not result.went_to_showdown or result.opp_hole is None or len(result.board) != 5:
        return
    hole = memory.get("hole")
    if not hole or memory.get("hand_index") != result.hand_index:
        return
    try:
        opp_score = omaha_score(result.opp_hole, result.board)
    except ValueError:
        return
    cat = category_of(opp_score)
    memory["show_n"] = int(memory.get("show_n") or 0) + 1
    if cat >= STRAIGHT and result.winner == "opp":
        memory["show_strong"] = int(memory.get("show_strong") or 0) + 1
    elif cat <= TWO_PAIR or (result.winner == "me" and result.pot >= 20):
        memory["show_weak"] = int(memory.get("show_weak") or 0) + 1


def _rate(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def adjustments(memory: dict, to_call: int, is_button: bool) -> dict:
    """Threshold shifts. Empty-looking until a bucket has enough real samples."""
    opp = memory.get("opp") or {}
    post = opp.get("post") or {}
    # Postflop tells are the ones that change bet and call thresholds.
    # Preflop is folded in only when postflop is still thin.
    use = post if (post.get("fold", 0) + post.get("continue", 0)) >= _MIN_ACTION_OBS else None
    if use is None:
        pre = opp.get("pre") or {}
        if pre.get("fold", 0) + pre.get("continue", 0) >= _MIN_ACTION_OBS:
            use = pre
        else:
            use = post

    folds = use.get("fold", 0)
    continues = use.get("continue", 0)
    raises = use.get("raise", 0)
    non_raises = use.get("non_raise", 0)
    fold_obs = folds + continues
    raise_obs = raises + non_raises
    fold_rate = _rate(_FOLD_PRIOR + folds, _FOLD_PRIOR + _CONTINUE_PRIOR + fold_obs)
    raise_rate = _rate(_RAISE_PRIOR + raises, _RAISE_PRIOR + _NON_RAISE_PRIOR + raise_obs)

    show_n = int(memory.get("show_n") or 0)
    weak = int(memory.get("show_weak") or 0)
    strong = int(memory.get("show_strong") or 0)
    weak_rate = _rate(_WEAK_PRIOR + weak, _WEAK_PRIOR + _STRONG_PRIOR + weak + strong)
    strong_rate = _rate(_STRONG_PRIOR + strong, _WEAK_PRIOR + _STRONG_PRIOR + weak + strong)

    bet_delta = 0.0
    call_delta = 0.0
    value_delta = 0.0
    station = False

    if fold_obs >= _MIN_ACTION_OBS and fold_rate > 0.55 and to_call == 0:
        bet_delta -= 0.08
        if is_button:
            bet_delta -= 0.04
    station_fold = fold_obs >= _MIN_ACTION_OBS and fold_rate < 0.25
    station_show = show_n >= _MIN_SHOWDOWNS and weak_rate > 0.55
    if station_fold or station_show:
        bet_delta += 0.08
        value_delta -= 0.04
        station = True
    if raise_obs >= _MIN_ACTION_OBS and raise_rate > 0.55 and show_n >= _MIN_SHOWDOWNS:
        if strong_rate > 0.50:
            call_delta += 0.05
        if weak_rate > 0.50:
            call_delta -= 0.05

    return {
        "bet_delta": bet_delta,
        "call_delta": call_delta,
        "value_delta": value_delta,
        "station": station,
    }
