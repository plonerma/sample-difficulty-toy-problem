import numpy as np


def get_data(
    num_samples=1000, tail_portion=0.03, bulk_scale=0.5, tail_radius=4,
    tail_noise=1.0, seed=0
):
    rng = np.random.default_rng(seed=seed)

    label = rng.integers(2, size=num_samples)

    in_tail = (rng.random(num_samples) < tail_portion)[:, None]

    noise = rng.normal(size=(num_samples, 2))

    class_direction = (2.0*label - 1.0)[:, None]

    offset = class_direction * np.array([1, 0])

    tail_pos = rng.uniform(0, 1, size=num_samples)

    arc_pos = np.stack([
        np.cos(tail_pos * np.pi),
        np.sin(tail_pos * np.pi)
    ], axis=-1) * tail_radius

    pos = (
        (~in_tail) * (bulk_scale * noise + offset)
        +
        (in_tail) * (
            (arc_pos - np.array([tail_radius - 1, 0]))*class_direction
            + noise * tail_noise * tail_pos[:, None]
        )
    )

    return pos, label
