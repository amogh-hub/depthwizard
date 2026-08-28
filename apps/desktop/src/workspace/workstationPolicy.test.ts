import { describe, expect, it } from "vitest";
import {
  activeWorkspaceStatus,
  compareToolAvailable,
  terrainControlsEnabled,
  terrainStageState,
} from "./workstationPolicy";

describe("workstation truthfulness policy", () => {
  it("does not enable comparison until a real validation reference exists", () => {
    expect(compareToolAvailable(true, false)).toBe(false);
    expect(compareToolAvailable(false, true)).toBe(false);
    expect(compareToolAvailable(true, true)).toBe(true);
  });

  it("does not call an existing mesh a completed renderer", () => {
    expect(terrainStageState(true, "loading")).toBe("active");
    expect(terrainStageState(true, "error")).toBe("failed");
    expect(terrainStageState(true, "ready")).toBe("complete");
    expect(terrainControlsEnabled("loading")).toBe(false);
    expect(terrainControlsEnabled("ready")).toBe(true);
  });

  it("keeps 2D status contextual instead of leaking 3D or old export readiness", () => {
    expect(activeWorkspaceStatus({
      activeView: "DSM",
      previewLoading: false,
      terrainPhase: "ready",
      processing: false,
      waitingForCalibration: false,
      calibrationReady: true,
      geometryReady: true,
      analysisBusy: false,
      exporting: false,
      buildingMesh: false,
      projectExportMiB: 24.2,
    })).toBe("DSM ready · metric DSM");
  });

  it("surfaces terrain loading and failure only in the terrain workspace", () => {
    expect(activeWorkspaceStatus({
      activeView: "3D Terrain",
      previewLoading: false,
      terrainPhase: "loading",
      terrainMessage: "Loading terrain…",
      processing: false,
      waitingForCalibration: false,
      calibrationReady: true,
      geometryReady: true,
      analysisBusy: false,
      exporting: false,
      buildingMesh: false,
    })).toBe("Loading terrain…");
  });
});
