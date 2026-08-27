import type {
  GroundControlPointFileReport,
  ProjectExportReport,
  ProjectProbeResult,
  ProjectProfileResult,
  ProjectStructureHeightResult,
  RasterMetadata,
  ReferenceValidationReport,
} from "../api";
import type { TerrainRenderState } from "../workspace/TerrainViewport";
import { AnalysisInspector } from "./AnalysisInspector";
import { StatusPipeline, type StageState } from "./StatusPipeline";

function gsdLabel(meta: RasterMetadata | null): string {
  if (!meta?.ground_sample_distance_x || !meta.ground_sample_distance_y) return "—";
  return `${meta.ground_sample_distance_x.toFixed(3)} × ${meta.ground_sample_distance_y.toFixed(3)} m`;
}

function byteLabel(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
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

type RendererTelemetry = {
  fps: number;
  triangles: number;
  drawCalls: number;
};

type InspectorProps = {
  metadata: RasterMetadata | null;
  geometryReady?: boolean;
  meshArtifactReady?: boolean;
  rendererReady?: boolean;
  terrainRenderState?: TerrainRenderState;
  activeView?: string;
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
  structureHeight?: ProjectStructureHeightResult | null;
  structureVertexCount?: number;
  gcpEvidence?: GroundControlPointFileReport | null;
  analysisBusy?: boolean;
  projectExport?: ProjectExportReport | null;
  meshLod?: number;
  autoLod?: boolean;
  terrainPerformance?: RendererTelemetry | null;
};

export function Inspector({
  metadata,
  geometryReady = false,
  meshArtifactReady = false,
  rendererReady = false,
  terrainRenderState,
  activeView = "DSM",
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
  structureHeight,
  structureVertexCount = 0,
  gcpEvidence,
  analysisBusy = false,
  projectExport,
  meshLod = 0,
  autoLod = false,
  terrainPerformance,
}: InspectorProps) {
  const hasInput = metadata !== null;
  const georeferenced = Boolean(metadata?.crs);
  const mode = elevationMode ?? (hasInput ? (georeferenced ? "Calibration eligible" : "Relative DSM") : "—");
  const benchmarkDataset = validationEvidence?.dataset.split("/")[0]?.trim() ?? "—";
  const referenceName = projectValidation?.reference_path.split(/[\\/]/).pop() ?? "—";
  const exportName = projectExport?.bundle_path.split(/[\\/]/).pop() ?? "—";
  const gcpName = gcpEvidence?.source_path.split(/[\\/]/).pop() ?? "—";
  const renderFailed = terrainRenderState?.phase === "error";
  const terrainStageState: StageState = rendererReady
    ? "complete"
    : renderFailed
      ? "failed"
      : meshArtifactReady
        ? "active"
        : "pending";
  const showAnalysis = ["Measure", "Profiles", "Validation", "Compare"].includes(activeTool) || Boolean(probe);
  const showValidation = ["Validation", "Compare"].includes(activeTool) || ["Reference", "Residual"].includes(activeView);
  const showExport = activeTool === "Export" || Boolean(projectExport);
  const subtitle = activeView === "3D Terrain"
    ? rendererReady ? "Interactive terrain renderer ready" : renderFailed ? "Terrain renderer requires attention" : meshArtifactReady ? "Preparing terrain renderer" : "Terrain mesh not built"
    : geometryReady ? `${activeView} analytical workspace` : hasInput ? "Source imagery loaded" : "No scene loaded";

  return (
    <aside className="dw-inspector" aria-label="Analysis inspector">
      <header className="dw-inspector-header">
        <h2>Scene inspector</h2>
        <p>{subtitle}</p>
      </header>

      <section className="dw-section">
        <div className="dw-section-title">Project state</div>
        <StatusPipeline stages={[
          { label: "Input", state: hasInput ? "complete" : "pending", detail: hasInput ? "ready" : "" },
          { label: "Geometry", state: geometryReady ? "complete" : hasInput ? "active" : "pending", detail: geometryReady ? (modelId ?? "DA3") : "" },
          { label: "Calibration", state: calibrationReady ? "complete" : "pending", detail: calibrationReady ? "metric evidence" : georeferenced ? "DEM/GCP" : "relative" },
          { label: "DSM", state: geometryReady ? "complete" : "pending", detail: geometryReady ? (calibrationReady ? "absolute" : "relative") : "" },
          { label: "Validation", state: projectValidation ? "complete" : "pending", detail: projectValidation ? `${projectValidation.valid_pixels.toLocaleString()} px` : "scene reference" },
          {
            label: "3D terrain",
            state: terrainStageState,
            detail: rendererReady ? "rendered GLB" : renderFailed ? "renderer failed" : meshArtifactReady ? "loading renderer" : "",
          },
          { label: "Export", state: projectExport ? "complete" : "pending", detail: projectExport ? byteLabel(projectExport.bundle_bytes) : "hash-audited ZIP" },
        ]} />
      </section>

      <details className="dw-inspector-disclosure" open={activeTool === "Project" || activeTool === "Layers"}>
        <summary>Geospatial metadata</summary>
        <section className="dw-section dw-section--nested">
          <dl className="dw-property-list">
            <div className="dw-property"><dt>Source</dt><dd title={metadata?.path}>{metadata?.path.split(/[\\/]/).pop() ?? "Not loaded"}</dd></div>
            <div className="dw-property"><dt>CRS</dt><dd title={metadata?.crs ?? undefined}>{metadata?.crs ?? "—"}</dd></div>
            <div className="dw-property"><dt>GSD</dt><dd>{gsdLabel(metadata)}</dd></div>
            <div className="dw-property"><dt>Raster size</dt><dd>{metadata ? `${metadata.width.toLocaleString()} × ${metadata.height.toLocaleString()}` : "—"}</dd></div>
            <div className="dw-property"><dt>Elevation product</dt><dd>{mode}</dd></div>
            <div className="dw-property"><dt>Geometry model</dt><dd title={modelId}>{modelId ?? (geometryReady ? "DA3" : "—")}</dd></div>
            <div className="dw-property"><dt>Tiling</dt><dd>{tileCount === undefined ? "—" : `${tileCount} tiles · ${harmonizedTiles ?? 0} harmonized`}</dd></div>
          </dl>
        </section>
      </details>

      {gcpEvidence && activeTool === "Project" && (
        <section className="dw-section">
          <div className="dw-section-title">Sparse GCP evidence</div>
          <dl className="dw-property-list">
            <div className="dw-property"><dt>Evidence file</dt><dd title={gcpEvidence.source_path}>{gcpName}</dd></div>
            <div className="dw-property"><dt>Control points</dt><dd>{gcpEvidence.point_count}</dd></div>
            <div className="dw-property"><dt>Elevation span</dt><dd>{gcpEvidence.minimum_elevation_m.toFixed(2)}–{gcpEvidence.maximum_elevation_m.toFixed(2)} m</dd></div>
            <div className="dw-property"><dt>SHA-256</dt><dd title={gcpEvidence.sha256}>{gcpEvidence.sha256.slice(0, 16)}…</dd></div>
          </dl>
          <div className="dw-validation-empty">
            <strong>Coordinates are never guessed</strong>
            <p>The CSV parser preserves x/y/elevation values exactly. Calibration interprets coordinates in the source raster CRS and verifies the file hash again before a metric claim.</p>
          </div>
        </section>
      )}

      {activeView === "3D Terrain" && meshArtifactReady && (
        <section className="dw-section">
          <div className="dw-section-title">3D renderer</div>
          <dl className="dw-property-list">
            <div className="dw-property"><dt>Renderer state</dt><dd>{terrainRenderState?.phase ?? "loading"}</dd></div>
            <div className="dw-property"><dt>Active LOD</dt><dd>LOD {meshLod}</dd></div>
            <div className="dw-property"><dt>LOD policy</dt><dd>{autoLod ? "adaptive" : "manual"}</dd></div>
            <div className="dw-property"><dt>Frame rate</dt><dd>{rendererReady && terrainPerformance ? `${terrainPerformance.fps.toFixed(1)} fps` : "—"}</dd></div>
            <div className="dw-property"><dt>Triangles</dt><dd>{rendererReady && terrainPerformance ? terrainPerformance.triangles.toLocaleString() : "—"}</dd></div>
            <div className="dw-property"><dt>Draw calls</dt><dd>{rendererReady && terrainPerformance ? terrainPerformance.drawCalls.toLocaleString() : "—"}</dd></div>
          </dl>
          {renderFailed ? (
            <div className="dw-validation-empty dw-validation-empty--danger">
              <strong>Renderer did not reach a valid frame</strong>
              <p>{terrainRenderState?.message}</p>
            </div>
          ) : (
            <div className="dw-validation-empty">
              <strong>Visualization is not the measurement source</strong>
              <p>LOD, analytical overlays, camera motion and vertical exaggeration change only the rendered workstation view. Probe and profile values continue to come from persisted project rasters.</p>
            </div>
          )}
        </section>
      )}

      {showAnalysis && (
        <AnalysisInspector
          activeTool={activeTool}
          probe={probe}
          measurement={measurement}
          profile={profile}
          analysisBusy={analysisBusy}
        />
      )}

      {activeTool === "Structures" && (
        <section className="dw-section">
          <div className="dw-section-title">Structural height</div>
          {structureHeight ? (
            <>
              <dl className="dw-property-list">
                <div className="dw-property"><dt>Structure height</dt><dd>{structureHeight.structure_height_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Robust top</dt><dd>{structureHeight.top_elevation_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Local ground</dt><dd>{structureHeight.ground_elevation_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Footprint pixels</dt><dd>{structureHeight.structure_pixels.toLocaleString()}</dd></div>
                <div className="dw-property"><dt>Ground-ring pixels</dt><dd>{structureHeight.ground_pixels.toLocaleString()}</dd></div>
                <div className="dw-property"><dt>Ring radius</dt><dd>{structureHeight.ring_pixels} px</dd></div>
              </dl>
              <div className="dw-validation-empty">
                <strong>Analyst-selected footprint</strong>
                <p>Height is the robust median DSM elevation inside the explicit footprint minus the robust surrounding ground ring. DepthWizard does not claim automatic building classification.</p>
              </div>
              {structureHeight.warnings.map((warning) => (
                <div className="dw-validation-empty dw-warning-note" key={warning}><strong>Selection warning</strong><p>{warning}</p></div>
              ))}
            </>
          ) : (
            <div className="dw-validation-empty">
              <strong>{structureVertexCount >= 3 ? "Footprint ready" : "Select a footprint"}</strong>
              <p>{structureVertexCount >= 3 ? `${structureVertexCount} vertices selected. Drag any numbered vertex to refine the footprint, then measure.` : "Click at least three vertices around one structure on the metric DSM. Vertices stay explicit, draggable and undoable before measurement."}</p>
            </div>
          )}
        </section>
      )}

      {showValidation && (
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
      )}

      {showExport && (
        <section className="dw-section">
          <div className="dw-section-title">Scientific export</div>
          {projectExport ? (
            <>
              <dl className="dw-property-list">
                <div className="dw-property"><dt>Bundle</dt><dd title={projectExport.bundle_path}>{exportName}</dd></div>
                <div className="dw-property"><dt>Size</dt><dd>{byteLabel(projectExport.bundle_bytes)}</dd></div>
                <div className="dw-property"><dt>Artifacts</dt><dd>{projectExport.files.length}</dd></div>
                <div className="dw-property"><dt>Source bytes</dt><dd>{projectExport.include_source ? "included" : "excluded"}</dd></div>
                <div className="dw-property"><dt>Mesh</dt><dd>{projectExport.include_mesh ? "included" : "excluded"}</dd></div>
                <div className="dw-property"><dt>Validation</dt><dd>{projectExport.include_validation ? "included" : "excluded"}</dd></div>
                <div className="dw-property"><dt>SHA-256</dt><dd title={projectExport.bundle_sha256}>{projectExport.bundle_sha256.slice(0, 16)}…</dd></div>
              </dl>
              <div className="dw-validation-empty">
                <strong>Transport derivative only</strong>
                <p>The ZIP re-hashes persisted products before packaging. Export does not rerun reconstruction, calibration, validation, or alter model evidence.</p>
              </div>
            </>
          ) : (
            <div className="dw-validation-empty">
              <strong>Export is deterministic</strong>
              <p>Use Export to create a hash-audited ZIP from the persisted project products. Source imagery remains excluded by default.</p>
            </div>
          )}
        </section>
      )}

      {validationEvidence && (
        <details className="dw-inspector-disclosure">
          <summary>Held-out model evidence</summary>
          <section className="dw-section dw-section--nested">
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
        </details>
      )}
    </aside>
  );
}
