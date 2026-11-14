"""Point resampling utilities shared by the current conversion commands."""

import math


def resample_polyline(points, num_points, method="legacy"):
    """Return a fixed number of points using the selected resampling method."""
    if method not in {"legacy", "arc_length"}:
        raise ValueError("method must be 'legacy' or 'arc_length'")
    if num_points < 1:
        raise ValueError("num_points must be at least 1")
    if not points:
        return [[0.0, 0.0]] * num_points
    if len(points) == 1:
        return [list(points[0]) for _ in range(num_points)]
    if num_points == 1:
        return [list(points[0])]

    if method == "legacy":
        if len(points) >= num_points:
            return [list(point) for point in points[:num_points]]
        return [list(point) for point in points] + [
            list(points[-1]) for _ in range(num_points - len(points))
        ]

    cumulative_lengths = [0.0]
    for first, second in zip(points, points[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        cumulative_lengths.append(cumulative_lengths[-1] + length)

    total_length = cumulative_lengths[-1]
    if total_length == 0:
        return [list(points[0]) for _ in range(num_points)]

    resampled = []
    segment_index = 0
    for index in range(num_points):
        target = total_length * index / (num_points - 1)
        while (
            segment_index < len(cumulative_lengths) - 2
            and cumulative_lengths[segment_index + 1] < target
        ):
            segment_index += 1

        start = points[segment_index]
        end = points[segment_index + 1]
        start_length = cumulative_lengths[segment_index]
        segment_length = cumulative_lengths[segment_index + 1] - start_length
        ratio = (target - start_length) / segment_length if segment_length else 0
        resampled.append([
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        ])

    return resampled


def resample_visibility(visibility, source_count, target_count):
    """Map visibility values to the indices selected by point resampling."""
    if source_count < 1:
        return [0] * target_count
    if not visibility:
        return [1] * target_count
    visibility = list(visibility[:source_count])
    if len(visibility) < source_count:
        visibility.extend([visibility[-1]] * (source_count - len(visibility)))
    if source_count <= 1 or target_count <= 1:
        return [visibility[0]] * target_count

    return [
        visibility[round(index * (source_count - 1) / (target_count - 1))]
        for index in range(target_count)
    ]
