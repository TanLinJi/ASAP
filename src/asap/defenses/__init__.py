"""ASAP baseline defense implementations.

- sor: Statistical Outlier Removal — removes points whose mean k-NN
  distance exceeds a threshold (mean + alpha * std of the population).
- ror: Radius Outlier Removal — removes points with fewer than
  min_neighbors within a fixed radius.
"""
