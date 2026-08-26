import type {
  ProjectProbeResult,
  ProjectProfileResult,
  RasterMetadata,
  ReferenceValidationReport,
} from "../api";
import { AnalysisInspector } from "./AnalysisInspector";
import { StatusPipeline } from "./StatusPipeline";

function gsdLabel(meta: RasterMetadata | null): string {
  if (!meta?.ground_sample_distance_x || !meta.ground_sample_distance_y) return "—";
  return `${meta.ground_sample_distance_x.toFixed(3)} × ${meta.ground_sample_distance_y.toFixed(3)} m`;
}

export type ValidationEvidence = {
  dataset: string;
  protocol: string;
  anchorCount: number;
  heldoutPixels: number;
  rmseM: number;
  maeM: number;
  pearsonR: number | null;
};

type InspectorProps = {
  metadata: RasterMetadata | null;
  geometryReady?: boolean;
  meshReady?: boolean;
  calibrationReady?: boolean;
  elevationMode?: string;
  modelId?: string;
  tileCount?: number;
  harmonizedTiles?: number;
  validationEvidence?: ValidationEvidence | null;
  projectValidation?: ReferenceValidationReport | null;
  activeTool?: string;
  probe?: ProjectProbeResult | null;
  measurement?: ProjectProfileResult | null;
  profile?: ProjectProfileResult | null;
  analysisBusy?: boolean;
};

export function Inspector({
  metadata,
  geometryReady = false,
  meshReady = false,
  calibrationReady = false,
  elevationMode,
  modelId,
  tileCount,
  harmonizedTiles,
  validationEvidence,
  projectValidation,
  activeTool = "Project",
  probe,
  measurement,
  profile,
  analysisBusy = false,
}: InspectorProps) {
  const hasInput = metadata !== null;
  const georeferenced = Boolean(metadata?.crs);
  const mode = elevationMode ?? (hasInput ? (georeferenced ? "Calibration eligible" : "Relative DSM") : "—");
  const benchmarkDataset = validationEvidence?.dataset.split("/")[0]?.trim() ?? "—";
  const referenceName = projectValidation?.reference_path.split(/[\\/]/).pop() ?? "—";

  return (
    <aside className="dw-inspector" aria-label="Analysis inspector">
      <header className="dw-inspector-header">
        <h2>Scene inspector</h2>
        <p>{geometryReady ? "Reconstructed terrain loaded" : hasInput ? "Source imagery loaded" : "No scene loaded"}</p>
      </header>

      <section className="dw-section">
        <div className="dw-section-title">Project state</div>
        <StatusPipeline stages={[
          { label: "Input", state: hasInput ? "complete" : "pending", detail: hasInput ? "ready" : "" },
          { label: "Geometry", state: geometryReady ? "complete" : hasInput ? "active" : "pending", detail: geometryReady ? (modelId ?? "DA3") : "" },
          { label: "Calibration", state: calibrationReady ? "complete" : "pending", detail: calibrationReady ? "metric evidence" : georeferenced ? "DEM/GCP" : "relative" },
          { label: "DSM", state: geometryReady ? "complete" : "pending", detail: geometryReady ? (calibrationReady ? "absolute" : "relative") : "" },
          { label: "Validation", state: projectValidation ? "complete" : "pending", detail: projectValidation ? `${projectValidation.valid_pixels.toLocaleString()} px` : "scene reference" },
          { label: "3D export", state: meshReady ? "complete" : "pending", detail: meshReady ? "GLB LOD" : "" },
        ]} />
      </section>

      <section className="dw-section">
        <div className="dw-section-title">Geospatial metadata</div>
        <dl className="dw-property-list">
          <div className="dw-property"><dt>Source</dt><dd title={metadata?.path}>{metadata?.path.split(/[\\/]/).pop() ?? "Not loaded"}</dd></div>
          <div className="dw-property"><dt>CRS</dt><dd>{metadata?.crs ?? "—"}</dd></div>
          <div className="dw-property"><dt>GSD</dt><dd>{gsdLabel(metadata)}</dd></div>
          <div className="dw-property"><dt>Raster size</dt><dd>{metadata ? `${metadata.width} × ${metadata.height}` : "—"}</dd></div>
          <div className="dw-property"><dt>Elevation product</dt><dd>{mode}</dd></div>
          <div className="dw-property"><dt>Geometry model</dt><dd>{modelId ?? (geometryReady ? "DA3" : "—")}</dd></div>
          <div className="dw-property"><dt>Tiling</dt><dd>{tileCount === undefined ? "—" : `${tileCount} tiles · ${harmonizedTiles ?? 0} harmonized`}</dd></div>
        </dl>
      </section>

      <AnalysisInspector
        activeTool={activeTool}
        probe={probe}
        measurement={measurement}
        profile={profile}
        analysisBusy={analysisBusy}
      />

      <section className="dw-section">
        <div className="dw-section-title">Project reference validation</div>
        {projectValidation ? (
          <>
            <dl className="dw-property-list">
              <div className="dw-property"><dt>Reference</dt><dd title={projectValidation.reference_path}>{referenceName}</dd></div>
              <div className="dw-property"><dt>RMSE</dt><dd>{projectValidation.elevation.rmse_m.toFixed(3)} m</dd></div>
              <div className="dw-property"><dt>MAE</dt><dd>{projectValidation.elevation.mae_m.toFixed(3)} m</dd></div>
              <div className="dw-property"><dt>Bias</dt><dd>{projectValidation.elevation.mean_bias_m.toFixed(3)} m</dd></div>
              <div className="dw-property"><dt>P95 error</dt><dd>{projectValidation.elevation.p95_abs_error_m.toFixed(3)} m</dd></div>
              <div className="dw-property"><dt>Pearson r</dt><dd>{projectValidation.elevation.pearson_r === null ? "—" : projectValidation.elevation.pearson_r.toFixed(3)}</dd></div>
              <div className="dw-property"><dt>Slope RMSE</dt><dd>{projectValidation.slope.rmse_degrees.toFixed(3)}°</dd></div>
              <div className="dw-property"><dt>Coverage</dt><dd>{(100 * projectValidation.coverage_fraction).toFixed(2)}%</dd></div>
              <div className="dw-property"><dt>Valid pixels</dt><dd>{projectValidation.valid_pixels.toLocaleString()}</dd></div>
              <div className="dw-property"><dt>Reliability</dt><dd>{projectValidation.reliability.available ? "measured" : "unavailable"}</dd></div>
            </dl>
            <div className="dw-validation-empty">
              <strong>Reference values stay evaluation-only</strong>
              <p>The reference is aligned downstream of reconstruction and calibration. The exact-file check prevents byte-identical calibration DEM reuse; it does not by itself prove geographic or sensor independence.</p>
            </div>
          </>
        ) : (
          <div className="dw-validation-empty">
            <strong>Reference DSM required</strong>
            <p>Load LiDAR or another metric reference surface to compute residuals, RMSE, MAE, bias, P95, correlation and slope diagnostics.</p>
          </div>
        )}
      </section>

      {validationEvidence && (
        <section className="dw-section">
          <div className="dw-section-title">Held-out model evidence</div>
          <dl className="dw-property-list">
            <div className="dw-property"><dt>Dataset</dt><dd title={validationEvidence.dataset}>{benchmarkDataset}</dd></div>
            <div className="dw-property"><dt>Protocol</dt><dd>{validationEvidence.anchorCount} sparse anchors</dd></div>
            <div className="dw-property"><dt>RMSE</dt><dd>{validationEvidence.rmseM.toFixed(3)} m</dd></div>
            <div className="dw-property"><dt>MAE</dt><dd>{validationEvidence.maeM.toFixed(3)} m</dd></div>
            <div className="dw-property"><dt>Pearson r</dt><dd>{validationEvidence.pearsonR === null ? "—" : validationEvidence.pearsonR.toFixed(3)}</dd></div>
            <div className="dw-property"><dt>Held-out pixels</dt><dd>{validationEvidence.heldoutPixels.toLocaleString()}</dd></div>
          </dl>
          <div className="dw-validation-empty">
            <strong>Separate benchmark scene</strong>
            <p>These metrics belong to the declared sparse-anchor OrthoLoC protocol; they are not reference validation of the currently displayed project.</p>
          </div>
        </section>
      )}
    </aside>
  );
}
