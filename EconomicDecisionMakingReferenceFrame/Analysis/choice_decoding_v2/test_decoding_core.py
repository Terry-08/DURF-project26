import unittest

import numpy as np

from decoding_core import (
    balanced_matched_indices,
    make_grouped_splits,
    make_standard_splits,
)


class DecodingCoreTests(unittest.TestCase):
    def test_value_matching_balances_every_retained_stratum(self):
        labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        strata = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        selected, count = balanced_matched_indices(labels, strata, random_seed=4)
        self.assertEqual(count, 3)
        for stratum in np.unique(strata[selected]):
            retained = labels[selected][strata[selected] == stratum]
            self.assertEqual(np.sum(retained == 0), np.sum(retained == 1))

    def test_grouped_splits_never_share_offer_pairs(self):
        groups = np.repeat(np.arange(20), 6)
        labels = np.tile([0, 1, 0, 1, 0, 1], 20)
        splits = make_grouped_splits(labels, groups, 5, 2, 10)
        self.assertEqual(len(splits), 10)
        for train, test in splits:
            self.assertTrue(set(groups[train]).isdisjoint(set(groups[test])))

    def test_standard_splits_preserve_both_classes(self):
        labels = np.tile([0, 1], 50)
        splits = make_standard_splits(labels, 5, 2, 10)
        self.assertEqual(len(splits), 10)
        for train, test in splits:
            self.assertEqual(set(labels[train]), {0, 1})
            self.assertEqual(set(labels[test]), {0, 1})


if __name__ == "__main__":
    unittest.main()
