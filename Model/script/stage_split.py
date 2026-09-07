"""Build a file-level stage table from an Excel mapping and LabelMe JSON files.

Example:

    python stage_split.py --json-root /public/share/ruanjue/wangmengtao/Project/keypoint/source_data --input stage_file.xlsx --output stage_file2.txt

The JSON root may contain nested directories. Category and date are read from
the JSON's ``mipdb.category_name`` and ``mipdb.picture_date`` fields, matching
the export format used by the companion grouping scripts. Each output row
represents one JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


def normalize_date(value: Any) -> date | None:
	"""Convert Excel dates and ISO/DMY strings to a calendar date."""
	if isinstance(value, datetime):
		return value.date()
	if isinstance(value, date):
		return value
	if value is None:
		return None

	text = str(value).strip()
	if not text:
		return None

	for parser in (
		lambda item: datetime.fromisoformat(item.replace("Z", "+00:00")).date(),
		lambda item: date(*map(int, item.split("/", 2)[::-1])),
		lambda item: date(*map(int, item.split("-", 2))),
	):
		try:
			return parser(text)
		except (ValueError, TypeError, IndexError):
			continue
	return None


def load_json_records(json_root: Path) -> dict[tuple[str, date], list[tuple[str, str]]]:
	"""Index image and JSON filenames by category metadata and picture date."""
	records: dict[tuple[str, date], list[tuple[str, str]]] = defaultdict(list)

	for json_path in json_root.rglob("*.json"):
		relative_path = json_path.relative_to(json_root)
		try:
			with json_path.open("r", encoding="utf-8") as file:
				data = json.load(file)
		except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
			print(f"Warning: skipped invalid JSON {json_path}: {error}", file=sys.stderr)
			continue

		metadata = data.get("mipdb") or {}
		category = str(metadata.get("category_name") or "").strip()
		if not category:
			category = relative_path.parts[0] if len(relative_path.parts) > 1 else ""

		picture_date = normalize_date(metadata.get("picture_date") or data.get("picture_date"))
		if picture_date is None:
			print(f"Warning: skipped JSON without a valid picture_date: {json_path}", file=sys.stderr)
			continue

		image_path = str(data.get("imagePath") or "").strip()
		image_filename = Path(image_path).name if image_path else json_path.name
		records[(category, picture_date)].append((image_filename, json_path.name))

	return records


def build_stage_file(input_path: Path, output_path: Path, json_root: Path) -> tuple[int, int]:
	"""Create a UTF-8 tab-separated file and return (written rows, unmatched rows)."""
	try:
		from openpyxl import load_workbook
	except ImportError as error:
		raise RuntimeError("This script requires openpyxl. Install it with: pip install openpyxl") from error

	workbook = load_workbook(input_path, read_only=True, data_only=True)
	sheet = workbook.active
	if sheet is None:
		raise ValueError("The input workbook has no active worksheet")
	rows = sheet.iter_rows(values_only=True)
	headers = [str(value).strip().lower() if value is not None else "" for value in next(rows)]

	required_headers = {"category", "date", "stage"}
	missing_headers = required_headers - set(headers)
	if missing_headers:
		raise ValueError(f"Missing required Excel columns: {', '.join(sorted(missing_headers))}")

	category_index = headers.index("category")
	date_index = headers.index("date")
	stage_index = headers.index("stage")

	records = load_json_records(json_root)
	written_rows = 0
	unmatched_rows = 0
	with output_path.open("w", encoding="utf-8", newline="") as output_file:
		output_file.write("category\tdate\tcount\tstage\tfilename\tjson_filename\n")
		for row in rows:
			category = str(row[category_index]).strip() if row[category_index] is not None else ""
			stage = row[stage_index]
			target_date = normalize_date(row[date_index])
			if not category or target_date is None:
				unmatched_rows += 1
				print(f"Warning: skipped Excel row with invalid category/date: {row}", file=sys.stderr)
				continue

			filenames = records.get((category, target_date), [])
			if not filenames:
				unmatched_rows += 1
				print(f"Warning: no JSON matched category={category!r}, date={target_date}", file=sys.stderr)
				continue

			for filename, json_filename in filenames:
				values = [
					category,
					target_date.strftime("%d/%m/%Y"),
					"1",
					"" if stage is None else str(stage).strip(),
					filename,
					json_filename,
				]
				output_file.write("\t".join(value.replace("\t", " ") for value in values) + "\n")
				written_rows += 1

	workbook.close()
	return written_rows, unmatched_rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--json-root", required=True, type=Path, help="Server directory containing category folders")
	parser.add_argument("--input", type=Path, default=Path("stage_file.xlsx"), help="Input Excel mapping")
	parser.add_argument("--output", type=Path, default=Path("stage_file2.txt"), help="Output UTF-8 tab-separated text file")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	if not args.input.is_file():
		print(f"Error: input workbook does not exist: {args.input}", file=sys.stderr)
		return 1
	if not args.json_root.is_dir():
		print(f"Error: JSON root does not exist or is not a directory: {args.json_root}", file=sys.stderr)
		return 1

	try:
		written_rows, unmatched_rows = build_stage_file(args.input, args.output, args.json_root)
	except (OSError, ValueError, RuntimeError) as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	print(f"Created {args.output} with {written_rows} file rows.")
	if unmatched_rows:
		print(f"Unmatched or invalid Excel rows: {unmatched_rows}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
