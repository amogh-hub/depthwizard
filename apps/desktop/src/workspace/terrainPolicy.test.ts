import { describe, expect, it } from "vitest";
import {
  lodPressureDelta,
  nextAutoLod,
  validTerrainTelemetry,
} from "./terrainPolicy";
import type { TerrainPerformance, TerrainRenderState } from "./TerrainViewport";

const ready: TerrainRenderState = {
  phase: "ready",
  message: "ready",
  triangles: 100,
  drawCalls: 2,
};

describe("terrain readiness and auto LOD", () => {
  it("rejects fps that comes from an empty renderer", () => {
    const empty: TerrainPerformance = { fps: 30, triangles: 0, drawCalls: 0 };
    expect(validTerrainTelemetry(ready, empty)).toBe(false);
    expect(lodPressureDelta(empty)).toBe(0);
  });

  it("does not degrade quality merely because the environment is capped near 30 fps", () => {
    const recorded: TerrainPerformance = { fps: 30, triangles: 250000, drawCalls: 3 };
    expect(validTerrainTelemetry(ready, recorded)).toBe(true);
    expect(lodPressureDelta(recorded)).toBe(0);
  });

  it("requires sustained severe pressure before degrading a LOD", () => {
    expect(nextAutoLod(0, 3, 3)).toBe(0);
    expect(nextAutoLod(0, 3, 4)).toBe(1);
  });

  it("requires a ready renderer before telemetry can drive policy", () => {
    const loading: TerrainRenderState = { ...ready, phase: "loading" };
    expect(validTerrainTelemetry(loading, { fps: 60, triangles: 250000, drawCalls: 3 })).toBe(false);
  });
});
