import type { RasterMetadata } from "../api";
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
}: InspectorProps) {
  const hasInput = metadata !== null;
  const georeferenced = Boolean(metadata?.crs);
  const mode = elevationMode ?? (hasInput ? (georeferenced ? "Calibration eligible" : "Relative DSM") : "—");
  const benchmarkDataset = validationEvidence?.dataset.split("/")[0]?.trim() ?? "—";

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
          { label: "Calibration", state: calibrationReady ? "complete" : "pending", detail: calibrationReady ? "DEM evidence" : georeferenced ? "DEM/GCP" : "relative" },
          { label: "DSM", state: geometryReady ? "complete" : "pending", detail: geometryReady ? (calibrationReady ? "absolute" : "relative") : "" },
          { label: "Validation", state: "pending", detail: "scene reference" },
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

      <section className="dw-section">
        <div className="dw-section-title">Held-out model evidence</div>
        {validationEvidence ? (
          <>
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
              <p>Metrics use the declared sparse-anchor OrthoLoC holdout protocol; they are not reference validation of the displayed Joshimath terrain.</p>
            </div>
          </>
        ) : (
          <div className="dw-validation-empty">
            <strong>Reference DSM required</strong>
            <p>Load LiDAR or another reference surface to compute RMSE, MAE, correlation and residual diagnostics.</p>
          </div>
        )}
      </section>
    </aside>
  );
}
