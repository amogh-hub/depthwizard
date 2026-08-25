import type { RasterMetadata } from "../api";
import { StatusPipeline } from "./StatusPipeline";

function gsdLabel(meta: RasterMetadata | null): string {
  if (!meta?.ground_sample_distance_x || !meta.ground_sample_distance_y) return "—";
  return `${meta.ground_sample_distance_x.toFixed(3)} × ${meta.ground_sample_distance_y.toFixed(3)}`;
}

export function Inspector({ metadata }: { metadata: RasterMetadata | null }) {
  const hasInput = metadata !== null;
  const georeferenced = Boolean(metadata?.crs && metadata.transform);
  return (
    <aside className="dw-inspector" aria-label="Analysis inspector">
      <header className="dw-inspector-header">
        <h2>Scene inspector</h2>
        <p>{hasInput ? "Source imagery loaded" : "No analysis point selected"}</p>
      </header>

      <section className="dw-section">
        <div className="dw-section-title">Project state</div>
        <StatusPipeline stages={[
          { label: "Input", state: hasInput ? "complete" : "pending", detail: hasInput ? "ready" : "" },
          { label: "Geometry", state: "pending" },
          { label: "Calibration", state: "pending" },
          { label: "DSM", state: "pending" },
          { label: "Validation", state: "pending" },
          { label: "Export", state: "pending" },
        ]} />
      </section>

      <section className="dw-section">
        <div className="dw-section-title">Geospatial metadata</div>
        <dl className="dw-property-list">
          <div className="dw-property"><dt>Source</dt><dd title={metadata?.path}>{metadata?.path.split(/[\\/]/).pop() ?? "Not loaded"}</dd></div>
          <div className="dw-property"><dt>CRS</dt><dd>{metadata?.crs ?? "—"}</dd></div>
          <div className="dw-property"><dt>GSD</dt><dd>{gsdLabel(metadata)}</dd></div>
          <div className="dw-property"><dt>Raster size</dt><dd>{metadata ? `${metadata.width} × ${metadata.height}` : "—"}</dd></div>
          <div className="dw-property"><dt>Elevation mode</dt><dd>{hasInput ? (georeferenced ? "Absolute DSM" : "Relative DSM") : "—"}</dd></div>
        </dl>
      </section>

      <section className="dw-section">
        <div className="dw-section-title">Official validation</div>
        <div className="dw-metric-grid">
          <div className="dw-metric"><span>RMSE</span><strong>—</strong></div>
          <div className="dw-metric"><span>MAE</span><strong>—</strong></div>
          <div className="dw-metric"><span>Correlation</span><strong>—</strong></div>
          <div className="dw-metric"><span>Valid pixels</span><strong>—</strong></div>
        </div>
      </section>
    </aside>
  );
}
