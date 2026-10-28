"""Instrumented agents. Every policy sees only the official observation."""
import random
import time
from collections import Counter


def validate_action(obs, action):
    if not isinstance(action, list) or any(type(x) is not int for x in action):
        raise ValueError("Action must be list[int], not card IDs or bools")
    selection = obs.get("select")
    if selection is None:
        if len(action) != 60:
            raise ValueError("A deck must contain 60 cards")
        return
    if not selection["minCount"] <= len(action) <= selection["maxCount"]:
        raise ValueError("Selection count outside bounds")
    if len(action) != len(set(action)):
        raise ValueError("Duplicate option index")
    if any(x < 0 or x >= len(selection["option"]) for x in action):
        raise ValueError("Option index outside bounds")


class ProtocolAgent:
    def __init__(self, name, deck, seat, seed):
        if name not in ("random", "first"):
            raise ValueError(f"Unknown agent: {name}")
        self.name, self.deck, self.seat = name, list(deck), seat
        self.rng = random.Random(seed)
        self.durations = []
        self.contexts = Counter()
        self.round_calls = Counter()
        self.round_seconds = Counter()
        self.round_starts = {}
        self.invalid = 0

    def __call__(self, obs, configuration=None):
        start = time.perf_counter()
        try:
            selection = obs.get("select")
            if selection is None:
                action = self.choose(obs)
                round_id = 0
            else:
                current = obs["current"]
                if current["yourIndex"] != self.seat:
                    raise ValueError("Agent seat and observation.yourIndex disagree")
                round_id = int(current["round"])
                self.round_starts.setdefault(round_id, {
                    "yourIndex": current["yourIndex"], "turn": current["turn"],
                    "remainingOverageTime": obs["remainingOverageTime"],
                    "context": selection["context"],
                })
                self.contexts[str(selection["context"])] += 1
                action = self.choose(obs)
            validate_action(obs, action)
            self.round_calls[round_id] += 1
            return action
        except Exception:
            self.invalid += 1
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.durations.append(elapsed)
            self.round_seconds[locals().get("round_id", -1)] += elapsed

    def choose(self, obs):
        selection = obs.get('select')
        if selection is None:
            return list(self.deck)
        n, k = len(selection['option']), selection['maxCount']
        return self.rng.sample(range(n), k) if self.name == 'random' else list(range(k))

    def close(self):
        pass

