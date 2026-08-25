# ADR 0002 — Geospatial truthfulness

**Status:** Accepted

## Decision

DepthWizard never invents geospatial or metric semantics.

- PNG/JPG/TIFF without usable CRS + transform -> **rDSM**, explicitly dimensionless/relative.
- Georeferenced imagery -> metric DSM only after a valid metric calibration path.
- GeoTIFF exports preserve the target CRS and affine transform exactly.
- Reference/DEM rasters are reprojected to the prediction grid before metric computation.
- No shape-only comparison is accepted as scientific validation.

## Consequence

The UI may show `Absolute DSM` only when the source and calibration state support that claim. This rule is enforced in code and in presentation language.
