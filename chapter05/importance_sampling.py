import numpy as np

from typing import TypeAlias, Literal
SampleData: TypeAlias = Literal[1, 2, 3]
ProbDist: TypeAlias = dict[SampleData, float]

rng: np.random.Generator = np.random.default_rng()


xs: list[SampleData] = [1, 2, 3]
pi: ProbDist = {1: 0.1, 2: 0.1, 3: 0.8}
b: ProbDist = {1: 1/3, 2: 1/3, 3: 1/3}
p: ProbDist = {1: 0.2, 2: 0.2, 3: 0.6}


# true value
e: float = sum([x * prob for x, prob in pi.items()])
print(f"[True Value] E_pi[x] = {e:.3f}")


# estimated value: monte carlo
n: int = 100
samples: list[SampleData] = [rng.choice(xs, p=list(pi.values())) for _ in range(n)]
mean: float = np.mean(samples)
var: float = np.var(samples)
print(f"[Estimated Value (MC)] E_pi[x] ~ {mean:.3f} (var: {var:.3f})")


# estimated value: importance sampling
n = 100
samples = []
for _ in range(n):
    x: SampleData = rng.choice(list(b.keys()), p=list(b.values()))
    samples.append(pi[x]/b[x] * x)  # rho(x) * x = pi(x)/b(x) * x
mean = np.mean(samples)
var = np.var(samples)
print(f"[Estimated Value (IS)] E_pi[x] ~ {mean:.3f} (var: {var:.3f})")


# estimated value: importance sampling (low variance)
n = 100
samples = []
for _ in range(n):
    x: SampleData = rng.choice(list(p.keys()), p=list(p.values()))
    samples.append(pi[x]/p[x] * x)  # rho(x) * x = pi(x)/b(x) * x
mean = np.mean(samples)
var = np.var(samples)
print(f"[Estimated Value (IS)] E_pi[x] ~ {mean:.3f} (var: {var:.3f})")