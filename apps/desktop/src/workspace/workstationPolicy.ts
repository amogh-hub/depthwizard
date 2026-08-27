import type { TerrainRenderPhase } from "./TerrainViewport";
import type { StageState } from "../components/StatusPipeline";

export function compareToolAvailable(calibrationReady: boolean, hasReferenceValidation: boolean): boolean {
  return calibrationReady && hasReferenceValidation;
}

export function terrainStageState(
  meshArtifactReady: boolean,
  renderPhase: TerrainRenderPhase,
): StageState {
  if (renderPhase === "ready") return "complete";
  if (renderPhase === "error") return "failed";
  if (meshArtifactReady) return "active";
  return "pending";
}

export function terrainControlsEnabled(renderPhase: TerrainRenderPhase): boolean {
  return renderPhase === "ready";
}

export function activeWorkspaceStatus(input: {
  activeView: string;
  previewLoading: boolean;
  terrainPhase: TerrainRenderPhase;
  terrainMessage?: string;
  processing: boolean;
  waitingForCalibration: boolean;
  calibrationReady: boolean;
  geometryReady: boolean;
  analysisBusy: boolean;
  exporting: boolean;
  buildingMesh: boolean;
  projectError?: string | null;
  projectExportMiB?: number | null;
  validationRmseM?: number | null;
}): string {
  if (input.projectError) return input.projectError;
  if (input.exporting) return "Building deterministic project export…";
  if (input.buildingMesh) return "Building persistent terrain LODs…";
  if (input.analysisBusy) return "Sampling persisted analytical products…";
  if (input.activeView === "3D Terrain") {
    if (input.terrainPhase === "error") return input.terrainMessage ?? "Terrain renderer failed";
    if (input.terrainPhase === "loading") return input.terrainMessage ?? "Preparing 3D terrain…";
    if (input.terrainPhase === "ready") return "Interactive 3D terrain ready";
  }
  if (input.previewLoading) return `Loading ${input.activeView} scientific layer…`;
  if (input.projectExportMiB != null) return `Export ready · ${input.projectExportMiB.toFixed(2)} MiB`;
  if (input.validationRmseM != null && ["Validation", "Reference", "Residual"].includes(input.activeView)) {
    return `Reference validation ready · RMSE ${input.validationRmseM.toFixed(3)} m`;
  }
  if (input.waitingForCalibration) return "Geometry ready · metric evidence required";
  if (input.processing) return "Production processing…";
  if (input.calibrationReady) return `${input.activeView} ready · metric DSM`;
  if (input.geometryReady) return `${input.activeView} ready · relative DSM`;
  return "Ready · local processing";
}
