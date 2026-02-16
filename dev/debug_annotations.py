#!/usr/bin/env python3
"""Debug script for notebook 07 IDC annotation rendering.

Traces the full coordinate pipeline to identify why
IDC Pan-Cancer annotations don't appear on the rendered tile.

Known facts from ann_index/ann_group_index:
  - AnnotationCoordinateType: 2D (SCOORD = pixel coordinates)
  - GraphicType: POLYGON (nucleus boundary polygons)

Reference: IDC-Tutorials/notebooks/pathomics/microscopy_dicom_ann_intro.ipynb
uses DICOM ANN polygon coordinates directly as pixel coordinates.
"""

import os
import sys
import numpy as np
import pydicom
import highdicom as hd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from idc_index import IDCClient
from tiatoolbox.wsicore.wsireader import WSIReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOWNLOAD_DIR = "./slides"
ANN_DIR = "./annotations"
OUTPUT_DIR = "./debug_output"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Query IDC metadata (ann_index + ann_group_index) — no download needed
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Query IDC metadata for annotation details")
print("=" * 70)

idc_client = IDCClient()
idc_client.fetch_index("sm_index")
idc_client.fetch_index("ann_index")
idc_client.fetch_index("ann_group_index")

# Check coordinate type and graphic type from indexes
ann_metadata = idc_client.sql_query("""
    SELECT
        ann.SeriesInstanceUID as ann_series_uid,
        ann.referenced_SeriesInstanceUID as slide_series_uid,
        ann.AnnotationCoordinateType,
        ag.GraphicType,
        ag.NumberOfAnnotations,
        ag.AnnotationGroupLabel,
        i_slide.PatientID,
        ROUND(i_slide.series_size_MB, 1) as slide_size_mb,
        ROUND(i_ann.series_size_MB, 1) as ann_size_mb,
        s.ObjectiveLensPower,
        s.min_PixelSpacing_2sf as pixel_spacing_mm
    FROM ann_index ann
    JOIN ann_group_index ag ON ann.SeriesInstanceUID = ag.SeriesInstanceUID
    JOIN index i_ann ON ann.SeriesInstanceUID = i_ann.SeriesInstanceUID
    JOIN index i_slide ON ann.referenced_SeriesInstanceUID = i_slide.SeriesInstanceUID
    JOIN sm_index s ON ann.referenced_SeriesInstanceUID = s.SeriesInstanceUID
    WHERE i_ann.analysis_result_id = 'Pan-Cancer-Nuclei-Seg-DICOM'
        AND i_slide.collection_id = 'tcga_luad'
        AND s.ObjectiveLensPower >= 20
    ORDER BY i_slide.series_size_MB ASC
    LIMIT 5
""")

print(f"Found {len(ann_metadata)} candidates")
print(ann_metadata[['PatientID', 'AnnotationCoordinateType', 'GraphicType',
                     'NumberOfAnnotations', 'slide_size_mb']].to_string())

selected = ann_metadata.iloc[0]
slide_series_uid = selected["slide_series_uid"]
ann_series_uid = selected["ann_series_uid"]

print(f"\nSelected: Patient={selected['PatientID']}")
print(f"  AnnotationCoordinateType: {selected['AnnotationCoordinateType']}")
print(f"  GraphicType: {selected['GraphicType']}")
print(f"  NumberOfAnnotations: {selected['NumberOfAnnotations']}")
print(f"  Slide: {slide_series_uid} ({selected['slide_size_mb']} MB)")
print(f"  ANN: {ann_series_uid} ({selected['ann_size_mb']} MB)")

# ---------------------------------------------------------------------------
# Step 2: Get slide dimensions from sm_instance_index
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Get slide dimensions")
print("=" * 70)

idc_client.fetch_index("sm_instance_index")
pixel_info = idc_client.sql_query(f"""
    SELECT
        TotalPixelMatrixColumns as width,
        TotalPixelMatrixRows as height,
        PixelSpacing_0 as pixel_spacing_mm
    FROM sm_instance_index
    WHERE SeriesInstanceUID = '{slide_series_uid}'
    ORDER BY TotalPixelMatrixColumns DESC
    LIMIT 1
""")

slide_width_px = int(pixel_info.iloc[0]["width"])
slide_height_px = int(pixel_info.iloc[0]["height"])
px_spacing = pixel_info.iloc[0]["pixel_spacing_mm"]

print(f"Slide dimensions: {slide_width_px} x {slide_height_px} px")
print(f"Pixel spacing: {px_spacing:.6f} mm ({px_spacing*1000:.4f} um)")

# ---------------------------------------------------------------------------
# Step 3: Download slide + annotations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Download slide and annotations")
print("=" * 70)

slide_path = os.path.join(DOWNLOAD_DIR, slide_series_uid)
if os.path.isdir(slide_path) and any(f.endswith(".dcm") for f in os.listdir(slide_path)):
    print(f"Slide already downloaded: {slide_path}")
else:
    idc_client.download_from_selection(
        downloadDir=DOWNLOAD_DIR,
        seriesInstanceUID=[slide_series_uid],
        dirTemplate="%SeriesInstanceUID",
    )
    print(f"Downloaded slide to {slide_path}")

ann_files = [f for f in os.listdir(ANN_DIR) if f.endswith(".dcm")]
if ann_files:
    print(f"Annotation file(s) already present: {ann_files}")
else:
    idc_client.download_from_selection(
        downloadDir=ANN_DIR,
        seriesInstanceUID=[ann_series_uid],
        dirTemplate=None,
    )
    ann_files = [f for f in os.listdir(ANN_DIR) if f.endswith(".dcm")]
    print(f"Downloaded {len(ann_files)} annotation file(s)")

# ---------------------------------------------------------------------------
# Step 4: Inspect raw DICOM ANN data with pydicom
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Inspect raw DICOM ANN data (pydicom)")
print("=" * 70)

ann_path = os.path.join(ANN_DIR, ann_files[0])
ann_ds = pydicom.dcmread(ann_path)

print(f"SOP Class UID: {ann_ds.SOPClassUID}")
print(f"Modality: {ann_ds.Modality}")
coord_type_dicom = getattr(ann_ds, "AnnotationCoordinateType", "NOT PRESENT")
print(f"AnnotationCoordinateType: '{coord_type_dicom}'")

if hasattr(ann_ds, "AnnotationGroupSequence"):
    for i, grp in enumerate(ann_ds.AnnotationGroupSequence):
        label = getattr(grp, "AnnotationGroupLabel", "N/A")
        gtype = getattr(grp, "GraphicType", "N/A")
        n_ann = getattr(grp, "NumberOfAnnotations", "N/A")
        print(f"\nAnnotationGroup[{i}]: label='{label}', type={gtype}, n={n_ann}")

        # Read raw coordinate data
        raw_data = None
        dtype_label = None
        if hasattr(grp, "PointCoordinatesData"):
            raw_data = np.frombuffer(grp.PointCoordinatesData, dtype=np.float32)
            dtype_label = "float32 (PointCoordinatesData)"
        elif hasattr(grp, "DoublePointCoordinatesData"):
            raw_data = np.frombuffer(grp.DoublePointCoordinatesData, dtype=np.float64)
            dtype_label = "float64 (DoublePointCoordinatesData)"

        if raw_data is not None:
            n_coords = 3 if coord_type_dicom == "3D" else 2
            print(f"  Raw data: {len(raw_data)} values, {dtype_label}")
            print(f"  -> {len(raw_data) // n_coords} coordinate pairs ({n_coords}D)")

            # Show first few raw coordinate pairs
            for k in range(min(5, len(raw_data) // n_coords)):
                pt = raw_data[k * n_coords : (k + 1) * n_coords]
                print(f"     [{k}] {pt}")

            coords_reshaped = raw_data.reshape(-1, n_coords)
            for c in range(n_coords):
                axis_name = ["col/X", "row/Y", "Z"][c]
                cmin, cmax = coords_reshaped[:, c].min(), coords_reshaped[:, c].max()
                print(f"  {axis_name} range: [{cmin:.2f}, {cmax:.2f}]")

            # Check if ranges match slide dimensions
            print(f"\n  Slide pixel dimensions: {slide_width_px} (cols) x {slide_height_px} (rows)")
            col_max = coords_reshaped[:, 0].max()
            row_max = coords_reshaped[:, 1].max()
            print(f"  Max col/X ({col_max:.0f}) vs slide width ({slide_width_px}): "
                  f"{'WITHIN' if col_max <= slide_width_px else 'EXCEEDS'}")
            print(f"  Max row/Y ({row_max:.0f}) vs slide height ({slide_height_px}): "
                  f"{'WITHIN' if row_max <= slide_height_px else 'EXCEEDS'}")

            # Check swapped interpretation
            print(f"  Max col/X ({col_max:.0f}) vs slide height ({slide_height_px}): "
                  f"{'WITHIN' if col_max <= slide_height_px else 'EXCEEDS'} (if swapped)")
            print(f"  Max row/Y ({row_max:.0f}) vs slide width ({slide_width_px}): "
                  f"{'WITHIN' if row_max <= slide_width_px else 'EXCEEDS'} (if swapped)")
        else:
            print("  No coordinate data found!")

# ---------------------------------------------------------------------------
# Step 5: Parse with highdicom and extract centroids
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Parse with highdicom, extract polygon centroids")
print("=" * 70)

ann = hd.ann.annread(ann_path)

print(f"highdicom coordinate_type: {ann.annotation_coordinate_type}")
print(f"  .value = '{ann.annotation_coordinate_type.value}'")

ann_groups = ann.get_annotation_groups()
for i, group in enumerate(ann_groups):
    print(f"  Group {i+1}: label='{group.label}', "
          f"graphic_type={group.graphic_type}, "
          f"n={group.number_of_annotations}")

annotation_group = ann.get_annotation_group(number=1)

# Test BOTH approaches: notebook 07 vs reference notebook
print("\n--- Approach A: notebook 07 (coordinate_type=ann.annotation_coordinate_type) ---")
nuclei_ann_a = annotation_group.get_graphic_data(
    coordinate_type=ann.annotation_coordinate_type,
)
print(f"  Returned: {len(nuclei_ann_a)} items")

print("\n--- Approach B: reference notebook (coordinate_type='2D' hardcoded string) ---")
nuclei_ann_b = annotation_group.get_graphic_data(coordinate_type='2D')
print(f"  Returned: {len(nuclei_ann_b)} items")

# Compare
if len(nuclei_ann_a) == len(nuclei_ann_b):
    same = all(np.array_equal(a, b) for a, b in zip(nuclei_ann_a[:10], nuclei_ann_b[:10]))
    print(f"\n  Same count. First 10 elements identical: {same}")
else:
    print(f"\n  DIFFERENT counts! {len(nuclei_ann_a)} vs {len(nuclei_ann_b)}")

nuclei_annotations = nuclei_ann_b  # Use reference approach
print(f"\nUsing reference approach: {len(nuclei_annotations)} annotations")

if len(nuclei_annotations) == 0:
    print("ERROR: No annotations from get_graphic_data!")
    sys.exit(1)

# Show first few
for j in range(min(3, len(nuclei_annotations))):
    a = nuclei_annotations[j]
    print(f"  [{j}] shape={a.shape}, dtype={a.dtype}")
    if a.ndim == 1:
        print(f"       values: {a}")
    else:
        print(f"       first 3 vertices: {a[:3]}")
        print(f"       col/X range: [{a[:, 0].min():.1f}, {a[:, 0].max():.1f}]")
        print(f"       row/Y range: [{a[:, 1].min():.1f}, {a[:, 1].max():.1f}]")

# Extract centroids
centroids = []
for ann_data in nuclei_annotations:
    if ann_data.ndim == 1:
        centroids.append(ann_data[:2])
    else:
        centroids.append(ann_data[:, :2].mean(axis=0))

centroids = np.array(centroids, dtype=np.float64)
print(f"\nExtracted {len(centroids)} centroids")
print(f"Centroid col/X range: [{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}]")
print(f"Centroid row/Y range: [{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}]")

# Check in-bounds both ways
in_bounds_normal = (
    (centroids[:, 0] >= 0) & (centroids[:, 0] < slide_width_px) &
    (centroids[:, 1] >= 0) & (centroids[:, 1] < slide_height_px)
).sum()

in_bounds_swapped = (
    (centroids[:, 1] >= 0) & (centroids[:, 1] < slide_width_px) &
    (centroids[:, 0] >= 0) & (centroids[:, 0] < slide_height_px)
).sum()

print(f"\nIn slide bounds (col=X, row=Y): {in_bounds_normal} / {len(centroids)}")
print(f"In slide bounds (swapped axes): {in_bounds_swapped} / {len(centroids)}")

# Also test with Shapely R-tree (reference notebook approach)
print("\n--- Shapely R-tree test (reference notebook approach) ---")
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.strtree import STRtree
import shapely

shapely_polys = []
for ann_data in nuclei_annotations:
    if ann_data.ndim == 2 and len(ann_data) >= 3:
        p = ShapelyPolygon(ann_data)
        if shapely.is_valid(p):
            shapely_polys.append(p)

print(f"  Valid Shapely polygons: {len(shapely_polys)} / {len(nuclei_annotations)}")
if shapely_polys:
    r_tree = STRtree(shapely_polys)
    # Test with a tile in the center of the slide
    test_box = shapely_box(
        slide_width_px // 2 - 256,
        slide_height_px // 2 - 256,
        slide_width_px // 2 + 256,
        slide_height_px // 2 + 256,
    )
    hits = r_tree.query(test_box, predicate='intersects')
    print(f"  Center tile (512x512) Shapely hits: {len(hits)}")

    # Also check a range of tiles
    n_hits_total = 0
    for tx in range(0, slide_width_px, 2048):
        for ty in range(0, slide_height_px, 2048):
            tb = shapely_box(tx, ty, tx + 2048, ty + 2048)
            h = r_tree.query(tb, predicate='intersects')
            n_hits_total += len(h)
    print(f"  Total hits across all 2048x2048 tiles: {n_hits_total}")

# ---------------------------------------------------------------------------
# Step 6: Open slide, pick tissue tile, filter annotations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Open slide, pick tissue tile, check annotation overlap")
print("=" * 70)

reader = WSIReader.open(slide_path)
info = reader.info

if info.objective_power is None:
    info.objective_power = float(selected["ObjectiveLensPower"])
if info.mpp is None:
    pixel_spacing_um = float(selected["pixel_spacing_mm"]) * 1000
    info.mpp = np.array([pixel_spacing_um, pixel_spacing_um])

slide_w, slide_h = info.slide_dimensions
print(f"WSIReader dimensions: {slide_w} x {slide_h}")
print(f"  (width={slide_w} should match TotalPixelMatrixColumns={slide_width_px})")
print(f"  (height={slide_h} should match TotalPixelMatrixRows={slide_height_px})")
print(f"MPP: {info.mpp}, Objective: {info.objective_power}")

thumbnail = reader.slide_thumbnail(resolution=1.25, units="power")
print(f"Thumbnail shape: {thumbnail.shape}")

gray = np.mean(thumbnail, axis=2)
tissue_mask = gray < 200
tissue_coords = np.argwhere(tissue_mask)
center_y, center_x = tissue_coords.mean(axis=0).astype(int)

baseline_x = int(center_x * slide_w / thumbnail.shape[1])
baseline_y = int(center_y * slide_h / thumbnail.shape[0])
print(f"Tissue center (thumbnail): row={center_y}, col={center_x}")
print(f"Tissue center (baseline): x={baseline_x}, y={baseline_y}")

tile_size = 2048
baseline_mpp = float(info.mpp[0])
target_mpp = 0.5
baseline_extent = int(tile_size * target_mpp / baseline_mpp)

bounds = (
    max(0, baseline_x - baseline_extent // 2),
    max(0, baseline_y - baseline_extent // 2),
    min(slide_w, baseline_x + baseline_extent // 2),
    min(slide_h, baseline_y + baseline_extent // 2),
)

print(f"Tile bounds: x=[{bounds[0]}, {bounds[2]}], y=[{bounds[1]}, {bounds[3]}]")
print(f"Tile size in baseline pixels: {bounds[2]-bounds[0]} x {bounds[3]-bounds[1]}")

# Test filtering with centroids as (col, row) = (x, y) — normal
in_tile_normal = (
    (centroids[:, 0] >= bounds[0]) & (centroids[:, 0] < bounds[2]) &
    (centroids[:, 1] >= bounds[1]) & (centroids[:, 1] < bounds[3])
)

# Test filtering with swapped axes
in_tile_swapped = (
    (centroids[:, 1] >= bounds[0]) & (centroids[:, 1] < bounds[2]) &
    (centroids[:, 0] >= bounds[1]) & (centroids[:, 0] < bounds[3])
)

print(f"\nAnnotations in tile (normal col=X, row=Y): {in_tile_normal.sum()}")
print(f"Annotations in tile (swapped):             {in_tile_swapped.sum()}")

# Use whichever works better
if in_tile_swapped.sum() > in_tile_normal.sum():
    print("\n>>> SWAPPED axes gives more annotations — coordinates are likely (row, col)!")
    idc_centroids_px = centroids[:, ::-1].copy()  # swap to (col, row) = (x, y)
    in_tile = in_tile_swapped
    axis_order = "SWAPPED"
else:
    idc_centroids_px = centroids.copy()
    in_tile = in_tile_normal
    axis_order = "NORMAL"

n_in_tile = in_tile.sum()
print(f"\nUsing {axis_order} axis order: {n_in_tile} annotations in tile")

if n_in_tile == 0:
    print("\nDIAGNOSTIC: Zero annotations in tile!")
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    dists = np.sqrt((idc_centroids_px[:, 0] - cx)**2 + (idc_centroids_px[:, 1] - cy)**2)
    nearest = np.argmin(dists)
    print(f"  Tile center: ({cx:.0f}, {cy:.0f})")
    print(f"  Nearest annotation: ({idc_centroids_px[nearest, 0]:.0f}, "
          f"{idc_centroids_px[nearest, 1]:.0f}), dist={dists[nearest]:.0f} px")

# ---------------------------------------------------------------------------
# Step 7: Render diagnostic images
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Render diagnostic images")
print("=" * 70)

tile = reader.read_bounds(
    bounds=bounds,
    resolution=info.objective_power,
    units="power",
)
if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
    tile = np.array(Image.fromarray(tile).resize((tile_size, tile_size), Image.LANCZOS))

scale = tile_size / baseline_extent

tile_baseline_x = bounds[0]
tile_baseline_y = bounds[1]

region_centroids = idc_centroids_px[in_tile]
tile_coords = (region_centroids - np.array([tile_baseline_x, tile_baseline_y])) * scale

# Tile with annotations
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(tile)
axes[0].set_title("Tile (no annotations)")
axes[0].axis("off")

axes[1].imshow(tile)
if len(tile_coords) > 0:
    axes[1].scatter(tile_coords[:, 0], tile_coords[:, 1],
                    s=4, c="cyan", alpha=0.5, marker=".")
    axes[1].set_title(f"Tile + IDC ({len(tile_coords)} nuclei) [{axis_order}]")
else:
    axes[1].set_title(f"Tile + IDC (NONE FOUND) [{axis_order}]")
axes[1].axis("off")

out1 = os.path.join(OUTPUT_DIR, "debug_tile.png")
plt.tight_layout()
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved: {out1}")

# Thumbnail with all annotations — both axis interpretations
thumb_h, thumb_w = thumbnail.shape[:2]
sx = thumb_w / slide_w
sy = thumb_h / slide_h

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Normal: centroids[:, 0] = col/X, centroids[:, 1] = row/Y
axes[0].imshow(thumbnail)
axes[0].scatter(centroids[:, 0] * sx, centroids[:, 1] * sy,
                s=0.3, c="red", alpha=0.3)
rect = plt.Rectangle((bounds[0] * sx, bounds[1] * sy),
                      (bounds[2]-bounds[0]) * sx, (bounds[3]-bounds[1]) * sy,
                      linewidth=2, edgecolor="lime", facecolor="none")
axes[0].add_patch(rect)
axes[0].set_xlim(0, thumb_w)
axes[0].set_ylim(thumb_h, 0)
axes[0].set_title(f"Normal axis order (col=X, row=Y)\n"
                  f"{in_bounds_normal}/{len(centroids)} in slide")
axes[0].axis("off")

# Swapped: centroids[:, 0] = row/Y, centroids[:, 1] = col/X
axes[1].imshow(thumbnail)
axes[1].scatter(centroids[:, 1] * sx, centroids[:, 0] * sy,
                s=0.3, c="red", alpha=0.3)
rect2 = plt.Rectangle((bounds[0] * sx, bounds[1] * sy),
                       (bounds[2]-bounds[0]) * sx, (bounds[3]-bounds[1]) * sy,
                       linewidth=2, edgecolor="lime", facecolor="none")
axes[1].add_patch(rect2)
axes[1].set_xlim(0, thumb_w)
axes[1].set_ylim(thumb_h, 0)
axes[1].set_title(f"Swapped axis order (row=Y, col=X)\n"
                  f"{in_bounds_swapped}/{len(centroids)} in slide")
axes[1].axis("off")

out2 = os.path.join(OUTPUT_DIR, "debug_thumbnail.png")
plt.tight_layout()
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"AnnotationCoordinateType (ann_index): {selected['AnnotationCoordinateType']}")
print(f"AnnotationCoordinateType (DICOM):     {coord_type_dicom}")
print(f"GraphicType: {selected['GraphicType']}")
print(f"Total annotations: {selected['NumberOfAnnotations']}")
print(f"Centroids extracted: {len(centroids)}")
print(f"Slide: {slide_width_px} x {slide_height_px} px")
print()
print(f"Centroid ranges:")
print(f"  [:, 0]: [{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}]")
print(f"  [:, 1]: [{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}]")
print()
print(f"In-bounds (normal):  {in_bounds_normal} / {len(centroids)}")
print(f"In-bounds (swapped): {in_bounds_swapped} / {len(centroids)}")
print(f"In-tile (normal):    {in_tile_normal.sum()}")
print(f"In-tile (swapped):   {in_tile_swapped.sum()}")
print()
print(f"Tile bounds: x=[{bounds[0]}, {bounds[2]}], y=[{bounds[1]}, {bounds[3]}]")

if n_in_tile == 0:
    if in_tile_swapped.sum() == 0 and in_tile_normal.sum() == 0:
        print("\nNO ANNOTATIONS IN TILE with either axis order.")
        print("Check debug_thumbnail.png to see where annotations actually fall.")
    elif in_tile_swapped.sum() > 0:
        print("\nROOT CAUSE: Axis order is (row, col) not (col, row).")
        print("FIX: Swap annotation axes before filtering.")
    else:
        print("\nAnnotations exist in tile with normal axes but something else is wrong.")
else:
    print(f"\n{n_in_tile} annotations found in tile. Check debug_tile.png for rendering.")

print(f"\nOutput saved to {OUTPUT_DIR}/")
print("Done.")
