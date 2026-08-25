import type { RasterMetadata } from "../api";
import { StatusPipeline } from "./StatusPipeline";

function gsdLabel(meta: RasterMetadata | null): string {
  if (!meta?.ground_sample_distance_x || !meta.ground_sample_distance_y) return "—";
  return `${meta.ground_sample_distance_x.toFixed(3)} × ${meta.ground_sample_distance_y.toFixed(3)} m`;
}

type InspectorProps = {
  metadata: RasterMetadata | null;
  geometryReady?: boolean;
  meshReady?: boolean;
  elevationMode?: string;
};

export function Inspector({
  metadata,
  geometryReady = false,
  meshReady = false,
  elevationMode,
}: InspectorProps) {
  const hasInput = metadata !== null;
  const georeferenced = Boolean(metadata?.crs);
  const mode = elevationMode ?? (hasInput ? (georeferenced ? "Calibration eligible" : "Relative DSM") : "—");

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
          { label: "Geometry", state: geometryReady ? "complete" : hasInput ? "active" : "pending", detail: geometryReady ? "DA3" : "" },
          { label: "Calibration", state: "pending", detail: georeferenced ? "DEM/GCP" : "relative" },
          { label: "DSM", state: geometryReady ? "complete" : "pending", detail: geometryReady ? "rDSM" : "" },
          { label: "Validation", state: "pending", detail: "reference" },
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
        </dl>
      </section>

      <section className="dw-section">
        <div className="dw-section-title">Official validation</div>
        <div className="dw-validation-empty">
          <strong>Reference DSM required</strong>
          <p>Load LiDAR or another reference surface to compute RMSE, MAE, correlation and residual diagnostics.</p>
        </div>
      </section>
    </aside>
  );
}
