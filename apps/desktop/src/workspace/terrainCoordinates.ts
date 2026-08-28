import type { NormalizedPoint } from "../api";

function clampUnit(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Convert the UV returned by the loaded terrain GLB to DepthWizard's
 * normalized raster coordinate convention.
 *
 * The GLB loader exposes the terrain V coordinate in the same top-to-bottom
 * raster convention consumed by the analytical APIs. Do not apply a second
 * vertical inversion here.
 */
export function rasterPointFromTerrainUv(u: number, v: number): NormalizedPoint {
  return {
    x: clampUnit(u),
    y: clampUnit(v),
  };
}
