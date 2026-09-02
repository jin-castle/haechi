from __future__ import annotations

import unittest

from firesight_ros2.mmwave_obstacle_mapper import (
    FrontPoint,
    GridSpec,
    StabilizationConfig,
    TemporalObstacleTracker,
    cluster_occupied_cells,
)


class TemporalObstacleTrackerTest(unittest.TestCase):
    def test_single_frame_point_is_not_stable_obstacle(self) -> None:
        config = StabilizationConfig(
            hit_increment=0.6,
            decay_per_frame=0.2,
            occupied_threshold=1.0,
            max_score=2.0,
            min_cluster_cells=1,
        )
        tracker = TemporalObstacleTracker(
            grid=GridSpec(
                width=40,
                height=40,
                side_extent_m=1.0,
                resolution_m=0.05,
            ),
            config=config,
        )

        stable_cells = tracker.update((_front_point(x=1.0, y=0.0),))

        self.assertEqual(stable_cells, ())

    def test_repeated_nearby_points_become_one_cluster(self) -> None:
        config = StabilizationConfig(
            hit_increment=0.6,
            decay_per_frame=0.1,
            occupied_threshold=1.0,
            max_score=2.0,
            min_cluster_cells=1,
        )
        tracker = TemporalObstacleTracker(
            grid=GridSpec(
                width=40,
                height=40,
                side_extent_m=1.0,
                resolution_m=0.05,
            ),
            config=config,
        )

        tracker.update((_front_point(x=1.0, y=0.0),))
        stable_cells = tracker.update((_front_point(x=1.02, y=0.02),))
        clusters = cluster_occupied_cells(stable_cells, min_cluster_cells=1)

        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0].center_x_m, 1.0, delta=0.08)
        self.assertAlmostEqual(clusters[0].center_y_m, 0.0, delta=0.08)


def _front_point(x: float, y: float) -> FrontPoint:
    return FrontPoint(
        x=x,
        y=y,
        z=0.0,
        range_m=(x**2 + y**2) ** 0.5,
        angle_deg=0.0,
    )


if __name__ == "__main__":
    unittest.main()
