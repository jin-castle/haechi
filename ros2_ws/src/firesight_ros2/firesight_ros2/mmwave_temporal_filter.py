from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Final, Protocol, TypeGuard


class IndexedPoint(Protocol):
    def __getitem__(self, index: int, /) -> object: ...


@dataclass(frozen=True, slots=True)
class FrontPoint:
    x: float
    y: float
    z: float
    range_m: float
    angle_deg: float


@dataclass(frozen=True, slots=True)
class GridSpec:
    width: int
    height: int
    side_extent_m: float
    resolution_m: float


@dataclass(frozen=True, slots=True)
class StabilizationConfig:
    hit_increment: float
    decay_per_frame: float
    occupied_threshold: float
    max_score: float
    min_cluster_cells: int


@dataclass(frozen=True, slots=True)
class GridCell:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class OccupiedCell:
    cell: GridCell
    center_x_m: float
    center_y_m: float
    score: float


@dataclass(frozen=True, slots=True)
class ObstacleCluster:
    center_x_m: float
    center_y_m: float
    range_m: float
    score: float
    cell_count: int


class PointShapeError(TypeError):
    def __str__(self) -> str:
        return "PointCloud2 point must expose x/y/z fields or index access"


class TemporalObstacleTracker:
    def __init__(self, grid: GridSpec, config: StabilizationConfig) -> None:
        self._grid = grid
        self._config = config
        self._scores: dict[GridCell, float] = {}

    def update(self, points: tuple[FrontPoint, ...]) -> tuple[OccupiedCell, ...]:
        decayed = {
            cell: max(0.0, score - self._config.decay_per_frame)
            for cell, score in self._scores.items()
        }
        for point in points:
            cell = _point_cell(point, self._grid)
            if cell is None:
                continue
            decayed[cell] = min(
                self._config.max_score,
                decayed.get(cell, 0.0) + self._config.hit_increment,
            )
        self._scores = {
            cell: score
            for cell, score in decayed.items()
            if score >= self._config.decay_per_frame
        }
        return tuple(
            _occupied_cell(cell, score, self._grid)
            for cell, score in sorted(
                self._scores.items(),
                key=lambda item: (item[0].y, item[0].x),
            )
            if score >= self._config.occupied_threshold
        )


def cluster_occupied_cells(
    cells: tuple[OccupiedCell, ...],
    min_cluster_cells: int,
) -> tuple[ObstacleCluster, ...]:
    by_cell = {occupied.cell: occupied for occupied in cells}
    remaining = set(by_cell)
    clusters: list[ObstacleCluster] = []
    while remaining:
        seed = remaining.pop()
        component = _connected_component(seed, by_cell, remaining)
        if len(component) < min_cluster_cells:
            continue
        clusters.append(_cluster(component))
    return tuple(sorted(clusters, key=lambda cluster: cluster.range_m))


def point_xyz(raw_point: object) -> tuple[float, float, float]:
    if hasattr(raw_point, "x") and hasattr(raw_point, "y") and hasattr(raw_point, "z"):
        return float(raw_point.x), float(raw_point.y), float(raw_point.z)
    if _is_indexed_point(raw_point):
        return float(raw_point[0]), float(raw_point[1]), float(raw_point[2])
    raise PointShapeError


def range_color(range_m: float) -> tuple[float, float, float]:
    if range_m <= 0.5:
        return 1.0, 0.05, 0.05
    if range_m <= 1.0:
        return 1.0, 0.55, 0.05
    return 0.1, 0.55, 1.0


def _connected_component(
    seed: GridCell,
    by_cell: dict[GridCell, OccupiedCell],
    remaining: set[GridCell],
) -> tuple[OccupiedCell, ...]:
    queue: deque[GridCell] = deque([seed])
    component: list[OccupiedCell] = []
    while queue:
        cell = queue.popleft()
        component.append(by_cell[cell])
        for neighbor in _neighbors(cell):
            if neighbor in remaining:
                remaining.remove(neighbor)
                queue.append(neighbor)
    return tuple(component)


def _neighbors(cell: GridCell) -> tuple[GridCell, ...]:
    offsets: Final = (-1, 0, 1)
    return tuple(
        GridCell(cell.x + offset_x, cell.y + offset_y)
        for offset_x in offsets
        for offset_y in offsets
        if offset_x != 0 or offset_y != 0
    )


def _cluster(component: tuple[OccupiedCell, ...]) -> ObstacleCluster:
    total_score = sum(cell.score for cell in component)
    center_x = sum(cell.center_x_m * cell.score for cell in component) / total_score
    center_y = sum(cell.center_y_m * cell.score for cell in component) / total_score
    return ObstacleCluster(
        center_x_m=center_x,
        center_y_m=center_y,
        range_m=math.hypot(center_x, center_y),
        score=total_score / len(component),
        cell_count=len(component),
    )


def _point_cell(point: FrontPoint, grid: GridSpec) -> GridCell | None:
    cell = GridCell(
        x=int(point.x / grid.resolution_m),
        y=int((point.y + grid.side_extent_m) / grid.resolution_m),
    )
    if 0 <= cell.x < grid.width and 0 <= cell.y < grid.height:
        return cell
    return None


def _is_indexed_point(raw_point: object) -> TypeGuard[IndexedPoint]:
    return hasattr(raw_point, "__getitem__")


def _occupied_cell(cell: GridCell, score: float, grid: GridSpec) -> OccupiedCell:
    return OccupiedCell(
        cell=cell,
        center_x_m=(cell.x + 0.5) * grid.resolution_m,
        center_y_m=((cell.y + 0.5) * grid.resolution_m) - grid.side_extent_m,
        score=score,
    )
