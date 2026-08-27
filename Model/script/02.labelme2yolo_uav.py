from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
from io import BytesIO
from typing import Any, Optional

from PIL import Image
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Tile large UAV images with overlap and convert LabelMe 17-keypoint annotations "
			"to YOLO-Pose labels."
		)
	)
	parser.add_argument(
		"--json_input",
		required=True,
		help="LabelMe JSON path or a directory that contains *.json",
	)
	parser.add_argument(
		"--image_input",
		default=None,
		help=(
			"Image path or directory. If omitted, uses imagePath in JSON or decodes imageData."
		),
	)
	parser.add_argument(
		"--output_dir",
		required=True,
		help="Output root directory. Images go to output_dir/images and labels to output_dir/labels",
	)
	parser.add_argument("--tile_size", type=int, default=2048, help="Tile size in pixels")
	parser.add_argument("--overlap", type=int, default=256, help="Tile overlap in pixels")
	parser.add_argument(
		"--bbox_padding",
		type=float,
		default=25.0,
		help="Padding in pixels applied to bbox built from keypoints",
	)
	parser.add_argument(
		"--num_keypoints",
		type=int,
		default=17,
		help="Expected number of keypoints in each linestrip object",
	)
	parser.add_argument(
		"--min_keypoints_in_tile",
		type=int,
		default=15,
		help="Keep annotation when at least this many keypoints are inside a tile",
	)
	parser.add_argument(
		"--output_image_format",
		default="auto",
		choices=["auto", "jpg", "png"],
		help="Output tile image format",
	)
	parser.add_argument(
		"--jpg_quality",
		type=int,
		default=95,
		help="JPEG quality when output_image_format is jpg",
	)
	parser.add_argument(
		"--save_index",
		action="store_true",
		help="Save per-image tile index JSON",
	)
	return parser.parse_args()


def _parse_label(label: str) -> int:
	# Keep category mapping aligned with existing scripts.
	if "-y" in label:
		return 1  # leaf
	return 0  # stem


def _to_uint8_for_common_image(arr: np.ndarray) -> np.ndarray:
	if arr.dtype == np.uint8:
		return arr
	arr = arr.astype(np.float32)
	finite_mask = np.isfinite(arr)
	if not np.any(finite_mask):
		return np.zeros(arr.shape, dtype=np.uint8)
	vmin = float(np.min(arr[finite_mask]))
	vmax = float(np.max(arr[finite_mask]))
	if vmax <= vmin:
		return np.zeros(arr.shape, dtype=np.uint8)
	arr = (arr - vmin) / (vmax - vmin) * 255.0
	return np.clip(arr, 0, 255).astype(np.uint8)


def _collect_json_files(json_input: str) -> list[str]:
	if os.path.isdir(json_input):
		files = sorted(glob.glob(os.path.join(json_input, "*.json")))
	else:
		files = [json_input] if json_input.lower().endswith(".json") and os.path.isfile(json_input) else []
	return files


def _resolve_image_path(image_input: Optional[str], json_path: str, labelme_data: dict[str, Any]) -> Optional[str]:
	if image_input and os.path.isfile(image_input):
		return image_input

	base_name = os.path.splitext(os.path.basename(json_path))[0]
	image_path_in_json = labelme_data.get("imagePath")

	candidates: list[str] = []

	if image_input and os.path.isdir(image_input):
		if image_path_in_json:
			candidates.append(os.path.join(image_input, os.path.basename(image_path_in_json)))
		for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]:
			candidates.append(os.path.join(image_input, base_name + ext))

	json_dir = os.path.dirname(json_path)
	if image_path_in_json:
		candidates.append(os.path.join(json_dir, image_path_in_json))
		candidates.append(os.path.join(json_dir, os.path.basename(image_path_in_json)))

	for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]:
		candidates.append(os.path.join(json_dir, base_name + ext))

	for p in candidates:
		if os.path.isfile(p):
			return p
	return None


def _load_common_image_from_json_data(image_data_b64: str) -> np.ndarray:
	image_bytes = base64.b64decode(image_data_b64)
	img = Image.open(BytesIO(image_bytes))
	if img.mode == "RGBA":
		img = img.convert("RGB")
	return np.asarray(img)


def _read_labelme_objects(labelme_data: dict[str, Any], num_keypoints: int) -> list[dict[str, Any]]:
	objects: list[dict[str, Any]] = []
	for shape in labelme_data.get("shapes", []):
		if shape.get("shape_type") != "linestrip":
			continue
		label = shape.get("label")
		points = shape.get("points", [])
		if not label or not points:
			continue
		if len(points) != num_keypoints:
			continue

		vis = shape.get("visibility", [1] * num_keypoints)
		if len(vis) < num_keypoints:
			vis = vis + [1] * (num_keypoints - len(vis))
		elif len(vis) > num_keypoints:
			vis = vis[:num_keypoints]

		obj = {
			"class_id": _parse_label(str(label)),
			"points": [(float(p[0]), float(p[1])) for p in points],
			"visibility": [int(v) for v in vis],
		}
		objects.append(obj)
	return objects


def _all_points_inside_tile(
	points: list[tuple[float, float]],
	x0: int,
	y0: int,
	valid_w: int,
	valid_h: int,
) -> bool:
	x1 = x0 + valid_w
	y1 = y0 + valid_h
	for px, py in points:
		if px < x0 or px > x1 or py < y0 or py > y1:
			return False
	return True


def _select_and_pad_points_for_tile(
	points: list[tuple[float, float]],
	visibility: list[int],
	x0: int,
	y0: int,
	valid_w: int,
	valid_h: int,
	num_keypoints: int,
	min_keypoints_in_tile: int,
) -> tuple[list[tuple[float, float]], list[int]] | None:
	"""Select in-tile keypoints and pad the missing ones, preserving order.

	A keypoint is kept at its original index (in tile-local coordinates) when it
	falls inside the tile's valid region. Keypoints outside the region are
	treated as *missing* and filled with this object's last valid in-tile
	keypoint, so the output always has exactly ``num_keypoints`` points in their
	original order. Padded points are marked with visibility 0 (not labeled) so
	they are ignored by the pose loss during training.

	Returns None when fewer than ``min_keypoints_in_tile`` keypoints are inside.
	"""
	x1 = x0 + valid_w
	y1 = y0 + valid_h

	# Map each keypoint to its tile-local coordinate, or None when outside.
	local_points: list[tuple[float, float] | None] = []
	for px, py in points:
		if x0 <= px < x1 and y0 <= py < y1:
			local_points.append((px - x0, py - y0))
		else:
			local_points.append(None)

	inside_count = sum(1 for p in local_points if p is not None)
	if inside_count < min_keypoints_in_tile:
		return None

	# "Last valid point in this tile" = the highest-index in-tile keypoint.
	last_valid_point: tuple[float, float] | None = None
	for p in local_points:
		if p is not None:
			last_valid_point = p
	# inside_count >= min_keypoints_in_tile >= 1, so this is never None.

	out_points: list[tuple[float, float]] = []
	out_visibility: list[int] = []
	for idx in range(num_keypoints):
		p = local_points[idx] if idx < len(local_points) else None
		if p is not None:
			out_points.append(p)
			out_visibility.append(visibility[idx] if idx < len(visibility) else 1)
		else:
			out_points.append(last_valid_point)
			out_visibility.append(0)  # padded -> not labeled / ignored in loss

	return out_points, out_visibility


def _build_yolo_line(
	class_id: int,
	local_points: list[tuple[float, float]],
	visibility: list[int],
	tile_size: int,
	bbox_padding: float,
) -> str:
	xs = [p[0] for p in local_points]
	ys = [p[1] for p in local_points]

	x_min = max(0.0, min(xs) - bbox_padding)
	y_min = max(0.0, min(ys) - bbox_padding)
	x_max = min(float(tile_size), max(xs) + bbox_padding)
	y_max = min(float(tile_size), max(ys) + bbox_padding)

	bw = max(0.0, x_max - x_min)
	bh = max(0.0, y_max - y_min)
	cx = (x_min + x_max) / 2.0
	cy = (y_min + y_max) / 2.0

	row: list[float] = [
		float(class_id),
		cx / tile_size,
		cy / tile_size,
		bw / tile_size,
		bh / tile_size,
	]

	for idx, (kx, ky) in enumerate(local_points):
		kx = min(max(kx, 0.0), float(tile_size))
		ky = min(max(ky, 0.0), float(tile_size))
		vis = visibility[idx] if idx < len(visibility) else 1
		if vis <= 0:
			v = 0  # padded / out-of-tile keypoint -> not labeled
		elif vis == 1:
			v = 2  # visible
		else:
			v = 1  # labeled but not visible
		row.extend([kx / tile_size, ky / tile_size, float(v)])

	return " ".join(f"{x:.6f}" if i != 0 else str(int(x)) for i, x in enumerate(row))


def _write_tile_image(
	out_path: str,
	tile_array: np.ndarray,
	out_format: str,
	jpg_quality: int,
) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	data = tile_array

	if data.ndim == 2:
		img_arr = _to_uint8_for_common_image(data)
		pil_img = Image.fromarray(img_arr, mode="L")
	else:
		if data.shape[2] > 4:
			data = data[:, :, :3]
		img_arr = _to_uint8_for_common_image(data)
		if img_arr.shape[2] == 4 and out_format == "jpg":
			pil_img = Image.fromarray(img_arr, mode="RGBA").convert("RGB")
		elif img_arr.shape[2] == 4:
			pil_img = Image.fromarray(img_arr, mode="RGBA")
		elif img_arr.shape[2] == 1:
			pil_img = Image.fromarray(img_arr[:, :, 0], mode="L")
		else:
			pil_img = Image.fromarray(img_arr[:, :, :3], mode="RGB")

	if out_format == "jpg":
		pil_img.save(out_path, format="JPEG", quality=jpg_quality, subsampling=0)
	else:
		pil_img.save(out_path, format="PNG")


def _infer_output_format(input_ext: str, preferred: str) -> str:
	if preferred != "auto":
		return preferred
	ext = input_ext.lower()
	if ext in {".jpg", ".jpeg"}:
		return "jpg"
	return "png"


def process_one_json(json_path: str, args: argparse.Namespace) -> dict[str, int]:
	with open(json_path, "r", encoding="utf-8") as f:
		labelme_data = json.load(f)

	image_path = _resolve_image_path(args.image_input, json_path, labelme_data)
	image_data_b64 = labelme_data.get("imageData")

	if image_path is None and not image_data_b64:
		print(f"[WARN] No image found for {json_path}, skipped.")
		return {"saved_tiles": 0, "skipped_tiles": 0, "saved_labels": 0, "skipped_objects": 0}

	src_ext = os.path.splitext(image_path)[1].lower() if image_path else ".png"
	out_format = _infer_output_format(src_ext, args.output_image_format)

	objects = _read_labelme_objects(labelme_data, args.num_keypoints)
	if not objects:
		print(f"[WARN] No valid linestrip object with {args.num_keypoints} keypoints in {json_path}")

	os.makedirs(args.output_dir, exist_ok=True)
	image_out_dir = os.path.join(args.output_dir, "images")
	label_out_dir = os.path.join(args.output_dir, "labels")
	os.makedirs(image_out_dir, exist_ok=True)
	os.makedirs(label_out_dir, exist_ok=True)

	stem = os.path.splitext(os.path.basename(json_path))[0]

	saved_tiles = 0
	skipped_tiles = 0
	saved_labels = 0
	skipped_objects = 0

	step = args.tile_size - args.overlap
	if step <= 0:
		raise ValueError("overlap must be smaller than tile_size")

	tile_index: dict[str, Any] = {
		"json": os.path.basename(json_path),
		"image": os.path.basename(image_path) if image_path else f"{stem}(from_imageData)",
		"tile_size": args.tile_size,
		"overlap": args.overlap,
		"step": step,
		"tiles": [],
	}

	if image_path:
		img = Image.open(image_path)
		if img.mode not in ("RGB", "RGBA", "L"):
			img = img.convert("RGB")
		full_arr = np.asarray(img)
	else:
		full_arr = _load_common_image_from_json_data(image_data_b64)

	height, width = full_arr.shape[0], full_arr.shape[1]
	n_cols = max(1, math.ceil((width - args.overlap) / step))
	n_rows = max(1, math.ceil((height - args.overlap) / step))

	for row in range(n_rows):
		for col in range(n_cols):
			x0 = col * step
			y0 = row * step
			read_w = min(args.tile_size, width - x0)
			read_h = min(args.tile_size, height - y0)
			if read_w <= 0 or read_h <= 0:
				continue

			if full_arr.ndim == 2:
				tile = np.zeros((args.tile_size, args.tile_size), dtype=full_arr.dtype)
				patch = full_arr[y0:y0 + read_h, x0:x0 + read_w]
				tile[:read_h, :read_w] = patch
			else:
				channels = full_arr.shape[2]
				tile = np.zeros((args.tile_size, args.tile_size, channels), dtype=full_arr.dtype)
				patch = full_arr[y0:y0 + read_h, x0:x0 + read_w, :]
				tile[:read_h, :read_w, :] = patch

			ann_lines = []
			for obj in objects:
				selected = _select_and_pad_points_for_tile(
					points=obj["points"],
					visibility=obj["visibility"],
					x0=x0,
					y0=y0,
					valid_w=read_w,
					valid_h=read_h,
					num_keypoints=args.num_keypoints,
					min_keypoints_in_tile=args.min_keypoints_in_tile,
				)
				if selected is None:
					skipped_objects += 1
					continue
				local_points, local_visibility = selected
				ann_lines.append(
					_build_yolo_line(
						class_id=obj["class_id"],
						local_points=local_points,
						visibility=local_visibility,
						tile_size=args.tile_size,
						bbox_padding=args.bbox_padding,
					)
				)

			ext = out_format
			tile_name = f"{stem}_r{row:04d}_c{col:04d}.{ext}"
			label_name = f"{stem}_r{row:04d}_c{col:04d}.txt"
			out_img_path = os.path.join(image_out_dir, tile_name)
			out_label_path = os.path.join(label_out_dir, label_name)

			_write_tile_image(
				out_path=out_img_path,
				tile_array=tile,
				out_format=out_format,
				jpg_quality=args.jpg_quality,
			)

			with open(out_label_path, "w", encoding="utf-8") as lf:
				if ann_lines:
					lf.write("\n".join(ann_lines) + "\n")
					saved_labels += len(ann_lines)

			tile_index["tiles"].append(
				{
					"tile": tile_name,
					"label": label_name,
					"row": row,
					"col": col,
					"pixel_offset": [x0, y0],
					"actual_size": [read_w, read_h],
					"objects": len(ann_lines),
				}
			)
			saved_tiles += 1

	if args.save_index:
		index_dir = os.path.join(args.output_dir, "tile_index")
		os.makedirs(index_dir, exist_ok=True)
		index_path = os.path.join(index_dir, f"{stem}.json")
		with open(index_path, "w", encoding="utf-8") as f:
			json.dump(tile_index, f, ensure_ascii=False, indent=2)

	print(
		f"[DONE] {os.path.basename(json_path)} | saved_tiles={saved_tiles}, "
		f"skipped_tiles={skipped_tiles}, saved_labels={saved_labels}, skipped_objects={skipped_objects}"
	)
	return {
		"saved_tiles": saved_tiles,
		"skipped_tiles": skipped_tiles,
		"saved_labels": saved_labels,
		"skipped_objects": skipped_objects,
	}


def main() -> None:
	args = parse_args()

	if args.tile_size <= 0:
		raise ValueError("tile_size must be > 0")
	if args.overlap < 0 or args.overlap >= args.tile_size:
		raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
	if args.num_keypoints <= 0:
		raise ValueError("num_keypoints must be > 0")
	if args.min_keypoints_in_tile <= 0 or args.min_keypoints_in_tile > args.num_keypoints:
		raise ValueError("min_keypoints_in_tile must be in [1, num_keypoints]")

	json_files = _collect_json_files(args.json_input)
	if not json_files:
		raise FileNotFoundError(f"No JSON files found from: {args.json_input}")

	total_saved_tiles = 0
	total_skipped_tiles = 0
	total_saved_labels = 0
	total_skipped_objects = 0

	for idx, json_path in enumerate(json_files, start=1):
		print(f"[INFO] ({idx}/{len(json_files)}) Processing {json_path}")
		stats = process_one_json(json_path, args)
		total_saved_tiles += stats["saved_tiles"]
		total_skipped_tiles += stats["skipped_tiles"]
		total_saved_labels += stats["saved_labels"]
		total_skipped_objects += stats["skipped_objects"]

	print(
		"[SUMMARY] "
		f"saved_tiles={total_saved_tiles}, "
		f"skipped_tiles={total_skipped_tiles}, "
		f"saved_labels={total_saved_labels}, "
		f"skipped_objects={total_skipped_objects}"
	)


if __name__ == "__main__":
	main()