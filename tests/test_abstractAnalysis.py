import unittest

import numpy as np

from analysis.abstractAnalysis import AbstractAnalysis


class TestAbstractAnalysis(unittest.TestCase):
    def test_projection(self):
        # Arrange
        analysis = AbstractAnalysis([], [str(i) for i in range(7)], ["0", "3", "5", "6"])
        seed_clusters = [0, 0, 1, 4, 4, 3, 3]
        clusters = np.array([1, 1, 0, 0])
        # Act
        new_clusters = analysis.project_to_original(clusters, seed_clusters)
        # Assert
        self.assertListEqual(new_clusters, [1, 1, 2, 1, 1, 0, 0])
