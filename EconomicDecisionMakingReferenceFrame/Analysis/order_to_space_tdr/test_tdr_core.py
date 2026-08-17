import unittest

import numpy as np

try:
    from .tdr_core import run_cross_validated_tdr
except ImportError:
    from tdr_core import run_cross_validated_tdr


class TargetedDimensionalityReductionTests(unittest.TestCase):
    def test_recovers_early_order_and_late_space(self):
        rng = np.random.default_rng(13)
        repeats = 20
        combinations = [
            (order, location, sequence)
            for order in (-1, 1)
            for location in (-1, 1)
            for sequence in (-1, 1)
        ]
        order = np.repeat([item[0] for item in combinations], repeats)
        location = np.repeat([item[1] for item in combinations], repeats)
        sequence = np.repeat([item[2] for item in combinations], repeats)
        space = order * location
        n_trials = len(order)
        time = np.arange(1500, 1710, 10)
        n_time = len(time)
        n_time_total = 171
        n_units = 12

        order_axis = np.zeros(n_units)
        order_axis[0] = 1
        location_axis = np.zeros(n_units)
        location_axis[1] = 1
        space_axis = np.zeros(n_units)
        space_axis[2] = 1
        order_amplitude = np.linspace(2.2, 0.2, n_time)
        space_amplitude = np.linspace(0.1, 2.4, n_time)
        location_amplitude = np.full(n_time, 0.8)

        state = rng.normal(0, 0.18, size=(n_trials, n_time_total, n_units))
        signal = np.zeros((n_trials, n_time, n_units))
        signal += (
            order[:, None, None]
            * order_amplitude[None, :, None]
            * order_axis[None, None, :]
        )
        signal += (
            location[:, None, None]
            * location_amplitude[None, :, None]
            * location_axis[None, None, :]
        )
        signal += (
            space[:, None, None]
            * space_amplitude[None, :, None]
            * space_axis[None, None, :]
        )
        state[:, 150:171] += signal

        labels = {
            "order": order,
            "location": location,
            "space": space,
            "sequence": sequence,
            "value1": rng.uniform(0, 1, n_trials),
            "value2": rng.uniform(0, 1, n_trials),
            "trial_index": np.arange(n_trials),
        }
        result = run_cross_validated_tdr(
            state,
            labels,
            dt=10,
            start_ms=1500,
            end_ms=1700,
            order_window=(1500, 1560),
            location_window=(1500, 1700),
            space_window=(1640, 1700),
            n_splits=5,
            random_state=4,
        )

        unique_r2 = result["unique_r2"]
        self.assertGreater(unique_r2[0, :5].mean(), unique_r2[2, :5].mean())
        self.assertGreater(unique_r2[2, -5:].mean(), unique_r2[0, -5:].mean())
        self.assertEqual(result["condition_trajectories"].shape, (4, n_time, 2))
        self.assertTrue(np.all(result["condition_counts"] == 40))
        self.assertLess(abs(result["metrics"]["axis_angle_order_space"] - 90), 10)


if __name__ == "__main__":
    unittest.main()
