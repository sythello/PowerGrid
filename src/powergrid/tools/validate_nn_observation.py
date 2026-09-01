from __future__ import annotations

from dataclasses import replace
import json

from powergrid.ai.nn_rank_value.observation import (
    build_public_observation,
    encode_state_features,
)
from powergrid.session import GameSession


def main() -> None:
    session = GameSession.from_scenario("opening", seed=17)
    snapshot = session.snapshot()
    assert snapshot.active_request is not None
    state = snapshot.state
    mutated = replace(
        state,
        config=replace(state.config, seed=999_999),
        power_plant_draw_stack=tuple(reversed(state.power_plant_draw_stack)),
        power_plant_bottom_stack=tuple(reversed(state.power_plant_bottom_stack)),
    )
    original = build_public_observation(state, snapshot.active_request)
    hidden_changed = build_public_observation(mutated, snapshot.active_request)
    original_features, names = encode_state_features(original)
    changed_features, changed_names = encode_state_features(hidden_changed)

    serialized = json.dumps(original.to_dict(), sort_keys=True)
    assert original.to_dict() == hidden_changed.to_dict()
    assert original_features == changed_features
    assert names == changed_names
    assert '"seed"' not in serialized
    assert "power_plant_draw_stack" not in serialized
    assert "power_plant_bottom_stack" not in serialized

    print("Observation validation: PASS")
    print(f"  state features: {len(original_features)}")
    print("  seed invariance: PASS")
    print("  hidden deck-order invariance: PASS")


if __name__ == "__main__":
    main()
