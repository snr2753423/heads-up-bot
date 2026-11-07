"""Equity versus the price of the pot, with a small opponent adjustment."""

from __future__ import annotations

from game import Action, Observation, Street

from strategy.equity import postflop_equity, preflop_equity
from strategy.identify import library_adjustments
from strategy.opponent import adjustments

# Every tuned number lives here. Match logs are the reason to change one.
THRESHOLDS = {
    "call_margin": 0.04,
    "facing_pot": 0.78,
    "facing_half": 0.68,
    "bet_pot": 0.72,
    "bet_half": 0.58,
    "button_bet_discount": 0.04,
    "preflop_raise_penalty": 0.06,
    "allin_extra": 0.05,
    "open_raise": 0.53,
    "open_call": 0.42,
    "bb_option_raise": 0.58,
    "min_value_bet": 0.55,
}


def fallback(obs: Observation) -> Action:
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return Action.FOLD


def _clamp(value: float) -> float:
    return max(0.30, min(0.90, value))


def _blend(live: float, extra: float, weight: float) -> float:
    """Keep the live shift when it already covers the library, or points the other way.

    The library only fills an axis the live model has left at zero, or replaces
    a smaller shift that points the same way. It does not stack on top.
    """
    nudge = weight * extra
    if nudge == 0.0 or live == 0.0:
        return live + nudge
    if (live > 0) == (nudge > 0):
        return live if abs(live) >= abs(nudge) else nudge
    return live


def equity_of(obs: Observation, memory: dict) -> float:
    key = (obs.hand_index, int(obs.street), obs.board)
    cached = memory.get("eq_cache")
    if cached is not None and cached[0] == key:
        return cached[1]
    if obs.street == Street.PREFLOP or len(obs.board) < 3:
        equity = preflop_equity(obs.hole_cards)
    else:
        equity = postflop_equity(obs.hole_cards, obs.board)
    memory["eq_cache"] = (key, equity)
    return equity


def _commit_extra(obs: Observation, action: Action) -> float:
    total = obs.raise_to.get(action)
    if total is not None and total >= obs.my_bet + obs.my_stack:
        return THRESHOLDS["allin_extra"]
    return 0.0


def _raise(obs: Observation, equity: float, pot_at: float, half_at: float) -> Action | None:
    for action, needed in ((Action.RAISE_POT, pot_at), (Action.RAISE_HALF, half_at)):
        if action not in obs.legal_actions:
            continue
        if equity >= needed + _commit_extra(obs, action):
            return action
    return None


def choose(obs: Observation, memory: dict) -> Action:
    equity = equity_of(obs, memory)
    shift = adjustments(memory, obs.to_call, obs.is_button)
    try:
        library = library_adjustments(memory)
    except Exception:
        library = {"weight": 0.0, "bet_delta": 0.0, "call_delta": 0.0, "value_delta": 0.0, "station": False}
    weight = library["weight"]
    bet_delta = _blend(shift["bet_delta"], library["bet_delta"], weight)
    call_delta = _blend(shift["call_delta"], library["call_delta"], weight)
    value_delta = _blend(shift["value_delta"], library["value_delta"], weight)
    # A confirmed check-folder folds to a bet. Widen past the generic fold exploit, with a cap.
    if library.get("profile_id") == "starter_checkfold" and shift["bet_delta"] < 0:
        bet_delta = max(-0.20, shift["bet_delta"] + weight * library["bet_delta"])
    if weight >= 0.5 and library["station"] and not shift["station"]:
        shift = {**shift, "station": True}
    postflop = obs.street != Street.PREFLOP
    position = THRESHOLDS["button_bet_discount"] if obs.is_button and postflop else 0.0

    bet_pot = _clamp(THRESHOLDS["bet_pot"] + bet_delta + value_delta - position)
    bet_half = _clamp(THRESHOLDS["bet_half"] + bet_delta - position)
    facing_pot = _clamp(THRESHOLDS["facing_pot"] + value_delta)
    facing_half = _clamp(THRESHOLDS["facing_half"] + bet_delta)
    if bet_half > bet_pot:
        bet_half = bet_pot
    if facing_half > facing_pot:
        facing_half = facing_pot

    # Opening or a free option. One chip into the blinds is not a real raise.
    if obs.street == Street.PREFLOP and obs.to_call <= obs.big_blind:
        if obs.is_button:
            open_raise = _clamp(THRESHOLDS["open_raise"] + bet_delta)
            if equity >= open_raise:
                raised = _raise(obs, equity, open_raise, open_raise)
                if raised is not None:
                    return raised
            if equity >= THRESHOLDS["open_call"] or Action.FOLD not in obs.legal_actions:
                return Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else fallback(obs)
            return Action.FOLD
        option = _clamp(THRESHOLDS["bb_option_raise"] + bet_delta + value_delta)
        if equity >= option:
            raised = _raise(obs, equity, option, option)
            if raised is not None:
                return raised
        return Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else fallback(obs)

    if obs.to_call <= 0:
        if shift["station"] and equity < THRESHOLDS["min_value_bet"]:
            return Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else fallback(obs)
        raised = _raise(obs, equity, bet_pot, bet_half)
        if raised is not None:
            return raised
        return Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else fallback(obs)

    penalty = THRESHOLDS["preflop_raise_penalty"] if obs.street == Street.PREFLOP else 0.0
    price = obs.to_call / (obs.pot + obs.to_call) if obs.pot + obs.to_call else 1.0
    need_call = price + THRESHOLDS["call_margin"] + call_delta + penalty
    raised = _raise(obs, equity, facing_pot + penalty, facing_half + penalty)
    if raised is not None:
        return raised
    if equity >= need_call:
        return Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else fallback(obs)
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return fallback(obs)
