"""
通用正射图 / 大图分块工具
=========================

支持输入：
    - GeoTIFF / TIFF：保留 CRS、GeoTransform、世界坐标信息
    - JPG / JPEG：普通像素图像
    - PNG：普通像素图像

核心功能：
    1. 大图按 tile_size 分块
    2. 相邻瓦片支持 overlap
    3. 自动跳过空白瓦片
    4. 边缘瓦片自动 padding
    5. 保存 tile_index.json
    6. 检测结果可从 tile 局部坐标映射回原图全局像素坐标
    7. GeoTIFF 输入额外支持映射到世界坐标
    8. JPG/PNG 输入不强行生成世界坐标（因为通常没有 CRS / GeoTransform）

用法：
    python ortho_tiler.py input.tif -o output_dir --tile-size 2048 --overlap 256
    python ortho_tiler.py input.jpg -o output_dir --tile-size 2048 --overlap 256
    python ortho_tiler.py input.png -o output_dir --tile-size 2048 --overlap 256

依赖：
    TIFF: rasterio, numpy
    JPG/PNG: PIL/Pillow, numpy
"""

import os
import json
import math
import time
import argparse
from typing import List, Dict, Any, Optional

import numpy as np


# =========================================================================
# 可选依赖
# =========================================================================

def _import_rasterio():
    import rasterio
    import rasterio.windows
    return rasterio


def _import_pil():
    from PIL import Image
    return Image


# =========================================================================
# 工具函数
# =========================================================================

SUPPORTED_EXTENSIONS = {
    ".tif": "tiff",
    ".tiff": "tiff",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
}


def detect_input_type(path: str) -> str:
    """根据扩展名判断输入类型。"""
    ext = os.path.splitext(path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的输入格式: {ext}\n"
            f"支持: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    return SUPPORTED_EXTENSIONS[ext]


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """
    将 PIL / raster 数据转换为适合保存 JPG/PNG 的 uint8。

    对普通 RGB/RGBA uint8 数据不做修改。
    对其他 dtype 做 min-max 拉伸。
    """
    if data.dtype == np.uint8:
        return data

    data = data.astype(np.float32)

    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros(data.shape, dtype=np.uint8)

    vmin = np.min(data[finite])
    vmax = np.max(data[finite])

    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint8)

    data = (data - vmin) / (vmax - vmin) * 255.0
    return np.clip(data, 0, 255).astype(np.uint8)


def _array_to_pil(data: np.ndarray):
    """将 [H,W] 或 [H,W,C] ndarray 转成 PIL Image。"""
    Image = _import_pil()

    data = _normalize_to_uint8(data)

    if data.ndim == 2:
        return Image.fromarray(data, mode="L")

    if data.ndim == 3:
        channels = data.shape[2]

        if channels == 1:
            return Image.fromarray(data[:, :, 0], mode="L")
        elif channels == 3:
            return Image.fromarray(data, mode="RGB")
        elif channels == 4:
            return Image.fromarray(data, mode="RGBA")

    raise ValueError(f"无法保存的数据形状: {data.shape}")


def _is_empty_image(data: np.ndarray, empty_threshold: float = 0.0) -> bool:
    """
    判断普通 JPG/PNG 瓦片是否为空。

    empty_threshold:
        0 表示只有全 0 才认为为空。
        例如 0.95 可用于更激进地跳过接近全黑区域。
    """
    if data.size == 0:
        return True

    if data.ndim == 3:
        # 所有通道均为 0 的像素
        empty_pixels = np.all(data == 0, axis=0)
    else:
        empty_pixels = data == 0

    return float(np.mean(empty_pixels)) > empty_threshold


# =========================================================================
# 通用分块主函数
# =========================================================================

def tile_orthomosaic(
    ortho_path: str,
    output_dir: str,
    tile_size: int = 2048,
    overlap: int = 256,
    nodata_skip_ratio: float = 0.95,
    output_format: Optional[str] = None,
    jpg_quality: int = 95,
) -> Dict[str, Any]:
    """
    将大型 GeoTIFF / JPG / PNG 切成小瓦片。

    TIFF：
        - 使用 rasterio
        - 保留 CRS / GeoTransform
        - 支持世界坐标映射

    JPG/PNG：
        - 使用 Pillow
        - 不依赖 GeoTransform / CRS
        - 仅支持像素坐标映射

    Parameters
    ----------
    ortho_path : str
        输入图片路径。
    output_dir : str
        输出文件夹。
    tile_size : int
        瓦片边长（像素），默认 2048。
    overlap : int
        相邻瓦片重叠像素数，默认 256。
    nodata_skip_ratio : float
        TIFF 中 nodata 比例超过该值则跳过。
        JPG/PNG 中则表示黑色像素比例超过该值时跳过。
    output_format : str or None
        输出格式：
            None -> 根据输入自动选择
            tif / jpg / jpeg / png
    jpg_quality : int
        JPG 保存质量，默认 95。

    Returns
    -------
    dict
        瓦片索引信息。
    """
    if not os.path.isfile(ortho_path):
        raise FileNotFoundError(f"找不到输入文件: {ortho_path}")

    if tile_size <= 0:
        raise ValueError("tile_size 必须 > 0")

    if overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"overlap ({overlap}) 必须满足 0 <= overlap < tile_size ({tile_size})"
        )

    input_type = detect_input_type(ortho_path)
    input_ext = os.path.splitext(ortho_path)[1].lower()

    # 自动决定输出格式
    if output_format is None:
        if input_type == "tiff":
            output_format = "tif"
        elif input_ext in [".jpg", ".jpeg"]:
            output_format = "jpg"
        else:
            output_format = "png"

    output_format = output_format.lower().lstrip(".")

    allowed_output_formats = {"tif", "tiff", "jpg", "jpeg", "png"}
    if output_format not in allowed_output_formats:
        raise ValueError(
            f"不支持的输出格式: {output_format}. "
            f"支持: {sorted(allowed_output_formats)}"
        )

    # ---------------------------------------------------------------------
    # 读取源图像信息
    # ---------------------------------------------------------------------

    rasterio = None
    src = None
    pil_img = None

    crs = None
    transform = None
    px_w = None
    px_h = None
    nodata_val = None

    if input_type == "tiff":
        rasterio = _import_rasterio()
        src = rasterio.open(ortho_path)

        img_w = src.width
        img_h = src.height
        n_bands = src.count
        dtype = src.dtypes[0]

        nodata_val = src.nodata
        transform = src.transform
        crs = src.crs

        px_w = abs(transform.a)
        px_h = abs(transform.e)

        print(
            f"[INFO] 输入类型: GeoTIFF | "
            f"{img_w}×{img_h} px | {n_bands} bands | "
            f"pixel {px_w:.6f}×{px_h:.6f}"
        )
        print(f"[INFO] CRS: {crs}")

    else:
        Image = _import_pil()
        pil_img = Image.open(ortho_path)

        img_w, img_h = pil_img.size

        # 统一读取成 RGB/RGBA，避免 PNG palette / grayscale 等特殊模式
        if pil_img.mode in ("RGB", "RGBA", "L"):
            pass
        elif "A" in pil_img.mode:
            pil_img = pil_img.convert("RGBA")
        else:
            pil_img = pil_img.convert("RGB")

        n_bands = len(pil_img.getbands())
        dtype = np.uint8

        print(
            f"[INFO] 输入类型: {input_ext.upper()} | "
            f"{img_w}×{img_h} px | {n_bands} channels"
        )
        print("[INFO] JPG/PNG 没有 GeoTransform/CRS，将只记录像素坐标。")

    # ---------------------------------------------------------------------
    # 计算网格
    # ---------------------------------------------------------------------

    step = tile_size - overlap

    n_cols = max(1, math.ceil((img_w - overlap) / step))
    n_rows = max(1, math.ceil((img_h - overlap) / step))
    total_tiles = n_cols * n_rows

    print(
        f"[INFO] 瓦片: {tile_size}×{tile_size} px | "
        f"重叠: {overlap} px"
    )
    print(f"[INFO] 步长: {step} px")
    print(f"[INFO] 网格: {n_cols}×{n_rows} = {total_tiles} 块")

    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # tile index
    # ---------------------------------------------------------------------

    tile_index = {
        "source": os.path.basename(ortho_path),
        "source_type": input_type,
        "source_size": [img_w, img_h],
        "tile_size": tile_size,
        "overlap": overlap,
        "step": step,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "crs": str(crs) if crs else None,
        "pixel_size": [px_w, px_h] if px_w is not None else None,
        "coordinate_system": (
            "world + pixel" if input_type == "tiff"
            else "pixel_only"
        ),
        "tiles": [],
    }

    saved = 0
    skipped = 0
    t0 = time.time()

    # ---------------------------------------------------------------------
    # 开始分块
    # ---------------------------------------------------------------------

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):

            # 原图中的左上角像素偏移
            px_col = col_idx * step
            px_row = row_idx * step

            # 实际读取尺寸
            read_w = min(tile_size, img_w - px_col)
            read_h = min(tile_size, img_h - px_row)

            if read_w <= 0 or read_h <= 0:
                continue

            # =============================================================
            # TIFF
            # =============================================================
            if input_type == "tiff":

                window = rasterio.windows.Window(
                    px_col,
                    px_row,
                    read_w,
                    read_h
                )

                data = src.read(window=window)  # [B,H,W]

                # nodata 检查
                if nodata_val is not None:
                    nodata_ratio = np.mean(data[0] == nodata_val)

                    if nodata_ratio > nodata_skip_ratio:
                        skipped += 1
                        continue
                else:
                    if np.all(data == 0):
                        skipped += 1
                        continue

                # padding
                if read_w < tile_size or read_h < tile_size:
                    fill = nodata_val if nodata_val is not None else 0

                    padded = np.full(
                        (n_bands, tile_size, tile_size),
                        fill,
                        dtype=data.dtype
                    )

                    padded[:, :read_h, :read_w] = data
                    data = padded

                    actual_w, actual_h = read_w, read_h
                else:
                    actual_w, actual_h = tile_size, tile_size

                # tile GeoTransform
                tile_transform = rasterio.windows.transform(
                    window,
                    transform
                )

                # 世界坐标范围
                left, top = tile_transform * (0, 0)
                right, bottom = tile_transform * (
                    tile_size,
                    tile_size
                )

                x_min = min(left, right)
                x_max = max(left, right)
                y_min = min(top, bottom)
                y_max = max(top, bottom)

            # =============================================================
            # JPG / PNG
            # =============================================================
            else:

                # Pillow crop
                data_pil = pil_img.crop(
                    (
                        px_col,
                        px_row,
                        px_col + read_w,
                        px_row + read_h
                    )
                )

                data = np.asarray(data_pil)

                # 黑色区域检查
                if _is_empty_image(
                    data,
                    empty_threshold=nodata_skip_ratio
                ):
                    skipped += 1
                    continue

                # padding
                if read_w < tile_size or read_h < tile_size:

                    if data.ndim == 2:
                        padded = np.zeros(
                            (tile_size, tile_size),
                            dtype=np.uint8
                        )
                        padded[:read_h, :read_w] = data

                    else:
                        channels = data.shape[2]

                        padded = np.zeros(
                            (
                                tile_size,
                                tile_size,
                                channels
                            ),
                            dtype=np.uint8
                        )

                        padded[
                            :read_h,
                            :read_w,
                            :
                        ] = data

                    data = padded

                    actual_w, actual_h = read_w, read_h

                else:
                    actual_w, actual_h = tile_size, tile_size

                tile_transform = None

                # JPG/PNG 没有世界坐标
                x_min = x_max = y_min = y_max = None

            # -----------------------------------------------------------------
            # 保存 tile
            # -----------------------------------------------------------------

            tile_ext = output_format

            if tile_ext == "tiff":
                tile_ext = "tif"

            if tile_ext == "jpeg":
                tile_ext = "jpg"

            tile_name = (
                f"tile_{row_idx:04d}_{col_idx:04d}.{tile_ext}"
            )

            tile_path = os.path.join(
                tiles_dir,
                tile_name
            )

            # -------------------------------------------------------------
            # TIFF output
            # -------------------------------------------------------------

            if output_format in ("tif", "tiff"):

                if input_type != "tiff":
                    raise ValueError(
                        "JPG/PNG 可以输出 JPG/PNG；"
                        "如果需要输出 GeoTIFF，必须提供 GeoTransform/CRS。"
                    )

                with rasterio.open(
                    tile_path,
                    "w",
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=n_bands,
                    dtype=data.dtype,
                    crs=crs,
                    transform=tile_transform,
                    nodata=nodata_val,
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                ) as dst:
                    dst.write(data)

            # -------------------------------------------------------------
            # JPG / PNG output
            # -------------------------------------------------------------

            else:

                if data.ndim == 3:
                    # rasterio [B,H,W] -> [H,W,B]
                    # 只有 TIFF 会走到这里
                    if data.shape[0] in (1, 3, 4):
                        data_save = np.transpose(data, (1, 2, 0))
                    else:
                        raise ValueError(
                            f"无法把 {data.shape} 保存成普通 JPG/PNG。"
                        )
                else:
                    data_save = data

                image = _array_to_pil(data_save)

                if output_format in ("jpg", "jpeg"):

                    # JPG 不支持 RGBA
                    if image.mode == "RGBA":
                        image = image.convert("RGB")

                    image.save(
                        tile_path,
                        format="JPEG",
                        quality=jpg_quality,
                        subsampling=0,
                    )

                else:
                    image.save(
                        tile_path,
                        format="PNG"
                    )

            # -----------------------------------------------------------------
            # 保存 index
            # -----------------------------------------------------------------

            tile_record = {
                "name": tile_name,
                "row": row_idx,
                "col": col_idx,

                # 原图中的像素偏移
                "pixel_offset": [
                    px_col,
                    px_row
                ],

                # 实际有效区域，不包括 padding
                "actual_size": [
                    actual_w,
                    actual_h
                ],

                # 坐标类型
                "coordinate_system": (
                    "world + pixel"
                    if input_type == "tiff"
                    else "pixel_only"
                ),

                # TIFF 才有 world bounds
                "world_bounds": (
                    {
                        "x_min": round(x_min, 6),
                        "x_max": round(x_max, 6),
                        "y_min": round(y_min, 6),
                        "y_max": round(y_max, 6),
                    }
                    if x_min is not None
                    else None
                ),
            }

            tile_index["tiles"].append(tile_record)

            saved += 1

        # 进度
        done = min(
            (row_idx + 1) * n_cols,
            total_tiles
        )

        if (
            done % max(1, total_tiles // 10) < n_cols
            or row_idx == n_rows - 1
        ):
            print(
                f"[INFO]   {done}/{total_tiles} "
                f"({100 * done / total_tiles:.0f}%) | "
                f"saved: {saved} | skipped: {skipped}"
            )

    # ---------------------------------------------------------------------
    # 关闭资源
    # ---------------------------------------------------------------------

    if src is not None:
        src.close()

    if pil_img is not None:
        pil_img.close()

    # ---------------------------------------------------------------------
    # 保存索引
    # ---------------------------------------------------------------------

    index_path = os.path.join(
        output_dir,
        "tile_index.json"
    )

    with open(
        index_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            tile_index,
            f,
            indent=2,
            ensure_ascii=False
        )

    dt = time.time() - t0

    print(
        f"[INFO] 完成: {saved} 块保存, "
        f"{skipped} 块跳过 | {dt:.1f}s"
    )

    print(f"[INFO] 输出: {tiles_dir}")
    print(f"[INFO] 索引: {index_path}")

    return tile_index


# =========================================================================
# 检测结果坐标转换
# =========================================================================

class TileDetectionMapper:
    """
    将分块检测结果：

        tile local bbox
              ↓
        global pixel bbox
              ↓
        world bbox（仅 GeoTIFF）

    输入：
        output_dir/tile_index.json

    detections 格式：

        [
            {
                "bbox": [x1, y1, x2, y2],
                "label": "...",
                "confidence": 0.9
            }
        ]
    """

    def __init__(self, index_path: str):

        with open(
            index_path,
            encoding="utf-8"
        ) as f:
            self.index = json.load(f)

        self.tile_lookup = {
            t["name"]: t
            for t in self.index["tiles"]
        }

        self.overlap = self.index["overlap"]
        self.tile_size = self.index["tile_size"]

        self.pixel_size = self.index.get("pixel_size")

        self.coordinate_system = self.index.get(
            "coordinate_system",
            "pixel_only"
        )

    # ---------------------------------------------------------------------
    # tile bbox → global pixel bbox
    # ---------------------------------------------------------------------

    def tile_bbox_to_global_pixel(
        self,
        tile_name: str,
        bbox: list
    ) -> list:

        """
        bbox:
            [x_min, y_min, x_max, y_max]

        返回：
            原图全局像素坐标
        """

        tile = self.tile_lookup[tile_name]

        ox, oy = tile["pixel_offset"]

        return [
            bbox[0] + ox,
            bbox[1] + oy,
            bbox[2] + ox,
            bbox[3] + oy,
        ]

    # ---------------------------------------------------------------------
    # tile bbox → world bbox
    # ---------------------------------------------------------------------

    def tile_bbox_to_world(
        self,
        tile_name: str,
        bbox: list
    ) -> Optional[dict]:

        """
        只有 GeoTIFF 输入才可以得到世界坐标。

        JPG / PNG：
            返回 None。
        """

        tile = self.tile_lookup[tile_name]

        wb = tile.get("world_bounds")

        if wb is None:
            return None

        if self.pixel_size is None:
            return None

        px_w, px_h = self.pixel_size

        tile_x_origin = wb["x_min"]
        tile_y_origin = wb["y_max"]

        x_min = tile_x_origin + bbox[0] * px_w
        x_max = tile_x_origin + bbox[2] * px_w

        y_max = tile_y_origin - bbox[1] * px_h
        y_min = tile_y_origin - bbox[3] * px_h

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }

    # ---------------------------------------------------------------------
    # 单 tile detection mapping
    # ---------------------------------------------------------------------

    def map_detections(
        self,
        tile_name: str,
        detections: list
    ) -> list:

        mapped = []

        for det in detections:

            d = dict(det)

            d["tile"] = tile_name

            # 局部 → 全局像素
            d["bbox_global_pixel"] = (
                self.tile_bbox_to_global_pixel(
                    tile_name,
                    det["bbox"]
                )
            )

            # TIFF → 世界坐标
            world_bbox = self.tile_bbox_to_world(
                tile_name,
                det["bbox"]
            )

            if world_bbox is not None:
                d["bbox_world"] = world_bbox

            mapped.append(d)

        return mapped

    # ---------------------------------------------------------------------
    # 合并所有 tile detection
    # ---------------------------------------------------------------------

    def merge_all_detections(
        self,
        all_detections: Dict[str, list],
        iou_threshold: float = 0.5,
    ) -> list:

        """
        合并所有瓦片检测。

        注意：
            NMS 使用全局像素坐标，而不是世界坐标。
            因此 JPG/PNG 和 TIFF 都可以使用。

        返回：
            [
                {
                    "bbox_global_pixel": [...],
                    "bbox_world": {...},   # TIFF 才有
                    "label": "...",
                    "confidence": ...,
                    "tile": "..."
                }
            ]
        """

        # 收集所有 detection
        all_global = []

        for tile_name, dets in all_detections.items():

            mapped = self.map_detections(
                tile_name,
                dets
            )

            all_global.extend(mapped)

        if not all_global:
            return []

        print(
            f"[INFO] 合并前: "
            f"{len(all_global)} detections "
            f"from {len(all_detections)} tiles"
        )

        # 按置信度排序
        all_global.sort(
            key=lambda d: d.get(
                "confidence",
                0
            ),
            reverse=True
        )

        keep = []
        suppressed = set()

        for i, det_i in enumerate(all_global):

            if i in suppressed:
                continue

            keep.append(det_i)

            bbox_i = det_i[
                "bbox_global_pixel"
            ]

            for j in range(
                i + 1,
                len(all_global)
            ):

                if j in suppressed:
                    continue

                bbox_j = all_global[j][
                    "bbox_global_pixel"
                ]

                iou = self._compute_iou_pixel(
                    bbox_i,
                    bbox_j
                )

                if iou > iou_threshold:
                    suppressed.add(j)

        print(
            f"[INFO] 合并后: {len(keep)} detections "
            f"(去掉 {len(all_global) - len(keep)} 重复)"
        )

        return keep

    # ---------------------------------------------------------------------
    # pixel bbox IoU
    # ---------------------------------------------------------------------

    @staticmethod
    def _compute_iou_pixel(
        a: list,
        b: list
    ) -> float:

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (
            (ix2 - ix1)
            *
            (iy2 - iy1)
        )

        area_a = (
            (ax2 - ax1)
            *
            (ay2 - ay1)
        )

        area_b = (
            (bx2 - bx1)
            *
            (by2 - by1)
        )

        union = area_a + area_b - inter

        return (
            inter / union
            if union > 0
            else 0.0
        )


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "通用大图分块工具："
            "支持 GeoTIFF / TIFF / JPG / JPEG / PNG"
        )
    )

    parser.add_argument(
        "ortho",
        help=(
            "输入图像："
            ".tif/.tiff/.jpg/.jpeg/.png"
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        default="tiled_ortho",
        help="输出文件夹"
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="瓦片边长(px)，默认 2048"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="瓦片重叠像素数，默认 256"
    )

    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.95,
        help=(
            "空白比例超过此值则跳过，"
            "默认 0.95"
        )
    )

    parser.add_argument(
        "--format",
        choices=[
            "tif",
            "tiff",
            "jpg",
            "jpeg",
            "png"
        ],
        default=None,
        help=(
            "输出格式；默认跟随输入。"
            "TIFF→TIF，JPG→JPG，PNG→PNG"
        )
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG 输出质量，默认 95"
    )

    args = parser.parse_args()

    tile_orthomosaic(
        args.ortho,
        args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        nodata_skip_ratio=args.skip_ratio,
        output_format=args.format,
        jpg_quality=args.jpg_quality,
    )
