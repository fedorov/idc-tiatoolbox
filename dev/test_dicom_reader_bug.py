"""
Standalone test: DICOMWSIReader coordinate bug in TIAToolbox 1.6.0.

This script demonstrates that TIAToolbox's DICOMWSIReader passes unscaled
baseline coordinates to wsidicom when reading from non-baseline pyramid levels,
causing WsiDicomOutOfBoundsError for any region not near the slide origin.

The bug affects ALL TIAToolbox APIs that read at resolutions mapping to a
non-baseline level: read_bounds, read_rect, PointsPatchExtractor,
SlidingWindowPatchExtractor, and TilePyramidGenerator.

Tested with: tiatoolbox 1.6.0, wsidicom 0.22.0, Python 3.12
IDC data: cptac_luad, smallest 20x slide (C3L-03262, ~22 MB)

Usage:
    pip install tiatoolbox idc-index openslide-bin "numcodecs<0.16"
    python dev/test_dicom_reader_bug.py
"""
import math
import os
import sys
import numpy as np

from idc_index import IDCClient
from tiatoolbox.wsicore.wsireader import WSIReader


def setup_slide():
    """Download test slide from IDC if needed, return (reader, info, baseline_center)."""
    idc_client = IDCClient()
    idc_client.fetch_index("sm_index")

    candidates = idc_client.sql_query("""
        SELECT i.SeriesInstanceUID, s.ObjectiveLensPower,
               s.min_PixelSpacing_2sf as pixel_spacing_mm
        FROM sm_index s
        JOIN index i ON s.SeriesInstanceUID = i.SeriesInstanceUID
        WHERE i.collection_id = 'cptac_luad' AND s.ObjectiveLensPower = 20
        ORDER BY i.series_size_MB ASC LIMIT 1
    """)
    selected = candidates.iloc[0]
    series_uid = selected['SeriesInstanceUID']
    slide_path = os.path.join('./slides', series_uid)

    if not os.path.exists(slide_path):
        os.makedirs('./slides', exist_ok=True)
        idc_client.download_from_selection(
            downloadDir='./slides', seriesInstanceUID=[series_uid],
            dirTemplate='%SeriesInstanceUID'
        )

    reader = WSIReader.open(slide_path)
    info = reader.info
    info.objective_power = float(selected['ObjectiveLensPower'])
    info.mpp = np.array([float(selected['pixel_spacing_mm']) * 1000] * 2)

    # Find tissue center
    thumbnail = reader.slide_thumbnail(resolution=1.25, units="power")
    thumb_gray = np.mean(thumbnail, axis=2)
    tissue_coords = np.argwhere(thumb_gray < 200)
    cy, cx = tissue_coords.mean(axis=0).astype(int)
    baseline_x = int(cx * info.slide_dimensions[0] / thumbnail.shape[1])
    baseline_y = int(cy * info.slide_dimensions[1] / thumbnail.shape[0])

    return reader, info, (baseline_x, baseline_y)


def test_read_bounds(reader, info, center):
    """Test read_bounds at different resolutions."""
    print("\n" + "=" * 60)
    print("TEST: read_bounds")
    print("=" * 60)
    bx, by = center
    bounds = (bx - 1024, by - 1024, bx + 1024, by + 1024)
    print(f"  Bounds: {bounds} (baseline coords)")

    results = {}
    for power in [5, 10, 20]:
        try:
            region = reader.read_bounds(bounds=bounds, resolution=power, units="power")
            print(f"  {power:>3}x: OK  shape={region.shape}")
            results[power] = True
        except Exception as e:
            print(f"  {power:>3}x: FAIL {type(e).__name__}: {e}")
            results[power] = False
    return results


def test_read_rect(reader, info, center):
    """Test read_rect at different resolutions."""
    print("\n" + "=" * 60)
    print("TEST: read_rect (coord_space='resolution')")
    print("=" * 60)
    bx, by = center

    results = {}
    for power in [5, 10, 20]:
        scale = power / info.objective_power
        loc = (int(bx * scale) - 128, int(by * scale) - 128)
        try:
            patch = reader.read_rect(
                location=loc, size=(256, 256),
                resolution=power, units="power",
                coord_space="resolution"
            )
            print(f"  {power:>3}x loc={loc}: OK  shape={patch.shape}")
            results[power] = True
        except Exception as e:
            print(f"  {power:>3}x loc={loc}: FAIL {type(e).__name__}: {e}")
            results[power] = False
    return results


def test_points_patch_extractor(reader, info, center):
    """Test PointsPatchExtractor at different resolutions."""
    print("\n" + "=" * 60)
    print("TEST: PointsPatchExtractor")
    print("=" * 60)
    from tiatoolbox.tools.patchextraction import PointsPatchExtractor
    bx, by = center

    results = {}
    for power in [5, 10, 20]:
        scale = power / info.objective_power
        loc = np.array([[int(bx * scale), int(by * scale)]])
        try:
            ext = PointsPatchExtractor(
                input_img=reader, locations_list=loc,
                patch_size=(256, 256), resolution=power, units="power"
            )
            patch = next(iter(ext))
            print(f"  {power:>3}x loc={loc[0].tolist()}: OK  shape={patch.shape}")
            results[power] = True
        except Exception as e:
            print(f"  {power:>3}x loc={loc[0].tolist()}: FAIL {type(e).__name__}: {e}")
            results[power] = False
    return results


def test_sliding_window_extractor(reader, info):
    """Test SlidingWindowPatchExtractor - extract ALL patches."""
    print("\n" + "=" * 60)
    print("TEST: SlidingWindowPatchExtractor (full slide)")
    print("=" * 60)
    from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor

    results = {}
    for power in [5, 20]:
        try:
            ext = SlidingWindowPatchExtractor(
                input_img=reader, patch_size=(256, 256),
                stride=(256, 256), resolution=power, units="power"
            )
            count = 0
            for patch in ext:
                count += 1
            print(f"  {power:>3}x: OK  {count} patches extracted")
            results[power] = True
        except Exception as e:
            print(f"  {power:>3}x: FAIL after some patches: {type(e).__name__}: {e}")
            results[power] = False
    return results


def test_tile_pyramid_generator(reader, info):
    """Test TilePyramidGenerator - center tiles at each level."""
    print("\n" + "=" * 60)
    print("TEST: TilePyramidGenerator (center tiles)")
    print("=" * 60)
    from tiatoolbox.tools.pyramid import TilePyramidGenerator

    gen = TilePyramidGenerator(reader, tile_size=256, downsample=2)
    results = {}
    for level in range(gen.level_count):
        grid = gen.tile_grid_size(level)
        ds = gen.level_downsample(level)
        cx, cy = max(0, grid[0] // 2 - 1), max(0, grid[1] // 2 - 1)
        try:
            tile = gen.get_tile(level=level, x=cx, y=cy)
            print(f"  Level {level} (ds={ds:>2}, grid={grid}): "
                  f"tile({cx},{cy}) OK  size={tile.size}")
            results[level] = True
        except Exception as e:
            print(f"  Level {level} (ds={ds:>2}, grid={grid}): "
                  f"tile({cx},{cy}) FAIL {type(e).__name__}")
            results[level] = False
    return results


def test_workaround(reader, info, center):
    """Test the working workaround: read at native + PIL resize."""
    print("\n" + "=" * 60)
    print("WORKAROUND: read_bounds at native + PIL resize")
    print("=" * 60)
    from PIL import Image
    bx, by = center
    bounds = (
        max(0, bx - 1024),
        max(0, by - 1024),
        min(info.slide_dimensions[0], bx + 1024),
        min(info.slide_dimensions[1], by + 1024),
    )

    region_native = reader.read_bounds(
        bounds=bounds, resolution=info.objective_power, units="power"
    )
    print(f"  Native ({info.objective_power}x): {region_native.shape}")

    for power in [5, 10, 20]:
        scale = power / info.objective_power
        if scale < 1:
            new_w = int(region_native.shape[1] * scale)
            new_h = int(region_native.shape[0] * scale)
            region = np.array(Image.fromarray(region_native).resize(
                (new_w, new_h), Image.LANCZOS
            ))
        else:
            region = region_native
        print(f"  {power:>3}x: OK  shape={region.shape}")


def main():
    import tiatoolbox
    import wsidicom
    print(f"tiatoolbox {tiatoolbox.__version__}, wsidicom {wsidicom.__version__}")

    reader, info, center = setup_slide()
    print(f"\nSlide: {info.slide_dimensions} at {info.objective_power}x")
    print(f"Pyramid levels: {info.level_dimensions}")
    print(f"Tissue center (baseline): {center}")

    test_read_bounds(reader, info, center)
    test_read_rect(reader, info, center)
    test_points_patch_extractor(reader, info, center)
    test_sliding_window_extractor(reader, info)
    test_tile_pyramid_generator(reader, info)
    test_workaround(reader, info, center)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Bug: DICOMWSIReader passes unscaled baseline coordinates to wsidicom
when the requested resolution maps to a non-baseline pyramid level.

Affected: ALL TIAToolbox read APIs at resolutions using non-baseline levels
Working:  - Any API at native resolution (reads from level 0)
          - Any API at resolutions that read from level 0 + post-process
          - Workaround: read at native, resize with PIL
""")


if __name__ == "__main__":
    main()
