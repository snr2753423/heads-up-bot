"""Profile matching stays quiet until a verified pattern is obvious."""

from strategy.identify import library_adjustments, live_features
from strategy.opponent import init_memory
from scouting.analyze import analyze_agent


STARTER = '''
def choose_action(obs, memory):
    if obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD
'''


def _memory(fold, cont, raises, non_raises, pot=0):
    memory = {}
    init_memory(memory)
    memory["opp"]["post"] = {
        "fold": fold,
        "continue": cont,
        "raise": raises,
        "non_raise": non_raises,
        "pot": pot,
        "half": max(0, raises - pot),
    }
    return memory


def test_starter_source_is_checkfold():
    found = analyze_agent(STARTER)
    assert found["archetype"] == "starter_checkfold"
    assert found["facts"]


def test_no_match_before_enough_hands():
    memory = _memory(fold=10, cont=1, raises=0, non_raises=11)
    assert live_features(memory) is None
    assert library_adjustments(memory)["weight"] == 0.0


def test_checkfold_profile_bluffs_wider():
    memory = _memory(fold=48, cont=2, raises=0, non_raises=50)
    found = library_adjustments(memory)
    assert found["profile_id"] == "starter_checkfold"
    assert found["weight"] > 0.4
    assert found["bet_delta"] < 0


def test_calling_station_profile_stops_bluffs():
    memory = _memory(fold=1, cont=50, raises=0, non_raises=51)
    found = library_adjustments(memory)
    assert found["profile_id"] == "calling_station"
    assert found["station"] is True


def test_ambiguous_rates_do_not_match():
    memory = _memory(fold=20, cont=20, raises=15, non_raises=25, pot=8)
    found = library_adjustments(memory)
    assert found["weight"] == 0.0
    assert found["profile_id"] is None


def test_library_does_not_stack_on_the_live_shift():
    from strategy.policy import _blend

    assert _blend(0.08, 0.06, 0.65) == 0.08
    assert _blend(-0.12, -0.10, 0.65) == -0.12
    assert _blend(-0.05, 0.05, 0.65) == -0.05
    assert abs(_blend(0.0, -0.10, 0.5) - (-0.05)) < 1e-9


def test_pot_raiser_is_not_labeled_a_half_raiser():
    memory = _memory(fold=2, cont=40, raises=36, non_raises=6, pot=34)
    found = library_adjustments(memory)
    assert found["profile_id"] == "always_pot"
