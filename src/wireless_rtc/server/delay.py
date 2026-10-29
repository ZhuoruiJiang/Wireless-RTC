"""Independent, reproducible application-level response-delay injection."""
import math
import random
import copy


class PatternResponseDelay:
    """Deterministic, request-indexed phases; repeat a cycle or hold its last value."""

    def __init__(self, config):
        if not isinstance(config,dict) or type(config.get("repeat", True)) is not bool:
            raise ValueError("Pattern must be an object with boolean repeat")
        phases = config.get("phases")
        if not isinstance(phases,list) or not phases:
            raise ValueError("Pattern requires nonempty phases")
        sequence = []
        for phase in phases:
            if not isinstance(phase,dict):
                raise ValueError("Each pattern phase must be an object")
            count, values = phase.get("requests"), phase.get("values_ms")
            if type(count) is not int or not 1<=count<=100000:
                raise ValueError("Phase requests must be an integer in [1,100000]")
            if not isinstance(values,list) or not values or any(
                    type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=10000 for v in values):
                raise ValueError("Phase values_ms must be nonempty, finite delays in [0,10000]")
            if len(sequence)+count>100000:
                raise ValueError("Pattern exceeds 100000 requests")
            sequence.extend(float(values[i%len(values)]) for i in range(count))
        self.config = copy.deepcopy(config)
        self.repeat = config.get("repeat",True)
        self.sequence = tuple(sequence)
        self.reset()

    def reset(self, episode_seed=0):
        self.index = 0

    def sample(self):
        index = self.index%len(self.sequence) if self.repeat else min(self.index,len(self.sequence)-1)
        self.index += 1
        return self.sequence[index]

    def describe(self):
        return dict(kind="deterministic_pattern",pattern=copy.deepcopy(self.config),
                    repeat=self.repeat,cycle_requests=len(self.sequence),
                    placement="Once per successful infer response, after inference",
                    clock="Request-indexed, not wall-clock; extra server wait, not target total RTT",
                    reset="Same pattern starts at request zero for every episode, independent of seed")


class NormalResponseDelay:
    def __init__(self, mean_ms=150., std_ms=50., max_ms=1000., seed=42):
        if (not all(math.isfinite(x) for x in [mean_ms,std_ms,max_ms])
                or not 0 <= mean_ms <= max_ms <= 10000 or max_ms == 0 or std_ms < 0):
            raise ValueError("Require 0 <= mean <= max <= 10000 ms, max > 0, and std >= 0; all finite")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Delay seed must be an integer in [0,2**32)")
        self.mean_ms, self.std_ms, self.max_ms, self.seed = mean_ms,std_ms,max_ms,seed
        self.reset()

    def reset(self, episode_seed=0):
        self.rng = random.Random(f"response-delay:{self.seed}:{episode_seed}")

    def sample(self):
        return min(self.max_ms,max(0.,self.rng.gauss(self.mean_ms,self.std_ms)))

    def describe(self):
        return dict(kind="clipped_normal", mean_ms=self.mean_ms, std_ms=self.std_ms,
                    max_ms=self.max_ms, seed=self.seed,
                    placement="Once per successful infer response, after inference",
                    reset="Independent seeded sequence reset for each episode seed")
