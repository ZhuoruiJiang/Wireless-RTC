"""Independent, reproducible application-level response-delay injection."""
import math
import random


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
