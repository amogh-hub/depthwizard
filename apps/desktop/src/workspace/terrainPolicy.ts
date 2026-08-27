import type { TerrainPerformance, TerrainRenderState } from "./TerrainViewport";

export const AUTO_LOD_DEGRADE_FPS = 24;
export const AUTO_LOD_UPGRADE_FPS = 52;
export const AUTO_LOD_SUSTAINED_SAMPLES = 4;

export function validTerrainTelemetry(
  renderState: TerrainRenderState,
  performance: TerrainPerformance | null,
): performance is TerrainPerformance {
  return Boolean(
    renderState.phase === "ready"
      && performance
      && Number.isFinite(performance.fps)
      && performance.fps > 0
      && performance.triangles > 0
      && performance.drawCalls > 0,
  );
}

export function lodPressureDelta(performance: TerrainPerformance): -1 | 0 | 1 {
  if (performance.triangles <= 0 || performance.drawCalls <= 0 || !Number.isFinite(performance.fps)) return 0;
  if (performance.fps < AUTO_LOD_DEGRADE_FPS) return 1;
  if (performance.fps > AUTO_LOD_UPGRADE_FPS) return -1;
  return 0;
}

export function nextAutoLod(
  currentLod: number,
  lastLod: number,
  pressure: number,
): number {
  if (pressure >= AUTO_LOD_SUSTAINED_SAMPLES) return Math.min(lastLod, currentLod + 1);
  if (pressure <= -AUTO_LOD_SUSTAINED_SAMPLES) return Math.max(0, currentLod - 1);
  return currentLod;
}
