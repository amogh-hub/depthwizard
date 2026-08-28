import { describe, expect, it } from "vitest";
import { rasterPointFromTerrainUv } from "./terrainCoordinates";

describe("3D terrain UV to raster coordinates", () => {
  it("preserves the terrain V coordinate instead of vertically mirroring it", () => {
    const point = rasterPointFromTerrainUv(407 / 1023, 259 / 1023);

    expect(point.x).toBeCloseTo(407 / 1023, 12);
    expect(point.y).toBeCloseTo(259 / 1023, 12);
    expect(point.y).not.toBeCloseTo(763 / 1023, 3);
  });

  it("clamps raycast interpolation noise to the normalized raster domain", () => {
    expect(rasterPointFromTerrainUv(-0.01, 1.01)).toEqual({ x: 0, y: 1 });
  });
});
