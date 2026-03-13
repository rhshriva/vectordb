"""Shared fixtures and helpers for Quiver tests."""

import random
import pytest

SAMPLE_DIM = 128


def random_vectors(n, dim=SAMPLE_DIM, seed=42):
    """Generate n random float32 vectors of the given dimension."""
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def random_vector(dim=SAMPLE_DIM, seed=None):
    """Generate a single random vector."""
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


ALL_METRICS = ["cosine", "l2", "dot_product"]

ALL_INDEX_TYPES = ["flat", "hnsw", "quantized_flat", "fp16_flat", "ivf", "ivf_pq", "mmap_flat", "binary_flat"]
