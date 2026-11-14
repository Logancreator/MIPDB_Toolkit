"""Resample Labelme linestrip points for the current preprocessing commands."""

import argparse
import json
import os

from point_resampling import resample_polyline, resample_visibility


def process_directory(input_dir, output_dir, num_points, method):
    os.makedirs(output_dir, exist_ok=True)
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with open(input_path, "r", encoding="utf-8") as source:
            data = json.load(source)

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "linestrip":
                continue
            points = shape.get("points", [])
            visibility = shape.get("visibility", [1] * len(points))
            shape["points"] = resample_polyline(points, num_points, method)
            shape["visibility"] = resample_visibility(
                visibility, len(points), num_points
            )

        with open(output_path, "w", encoding="utf-8") as target:
            json.dump(data, target, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Resample Labelme linestrip points."
    )
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_points", type=int, default=17)
    parser.add_argument(
        "--interpolation_method",
        choices=["legacy", "arc_length"],
        default="arc_length",
    )
    args = parser.parse_args()
    process_directory(
        args.json_dir,
        args.output_dir,
        args.num_points,
        args.interpolation_method,
    )


if __name__ == "__main__":
    main()
