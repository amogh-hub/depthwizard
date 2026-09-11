import type {
  GroundControlPointFileReport,
  ProjectExportReport,
  ProjectProbeResult,
  ProjectProfileResult,
  ProjectStructureHeightResult,
  RasterMetadata,
  ReferenceValidationReport,
} from "../api";
import type {
  TerrainPerformance,
  TerrainRenderState,
} from "../workspace/TerrainViewport";
import { AnalysisInspector } from "./AnalysisInspector";
import { StatusPipeline, type StageState } from "./StatusPipeline";

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
  geometryReady: boolean;
  meshArtifactReady: boolean;
  rendererReady: boolean;
  terrainRenderState: TerrainRenderState;
  activeView: string;
  calibrationReady: boolean;
  elevationMode?: string;
  modelId?: string;
  tileCount?: number;
  harmonizedTiles?: number;
  validationEvidence?: ValidationEvidence | null;
  projectValidation?: ReferenceValidationReport | null;
  activeTool: string;
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
  terrainPerformance?: TerrainPerformance | null;
};

function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

function byteLabel(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function qualityLabel(quality: ProjectStructureHeightResult["measurement_quality"]): string {
  if (quality === "high") return "High";
  if (quality === "moderate") return "Moderate";
  return "Low";
}

export function Inspector({
  metadata,
  geometryReady,
  meshArtifactReady,
  rendererReady,
  terrainRenderState,
  activeView,
  calibrationReady,
  elevationMode,
  modelId,
  tileCount,
  harmonizedTiles,
  validationEvidence,
  projectValidation,
  activeTool,
  probe,
  measurement,
  profile,
  structureHeight,
  structureVertexCount = 0,
  gcpEvidence,
  analysisBusy = false,
  projectExport,
  meshLod = 0,
  autoLod = true,
  terrainPerformance,
}: InspectorProps) {
  const hasInput = metadata !== null;
  const georeferenced = Boolean(metadata?.crs);
  const sourceName = metadata?.path ? fileName(metadata.path) : "—";
  const showValidation = activeTool === "Validation";
  const showExport = activeTool === "Export";
  const benchmarkDataset = validationEvidence?.dataset ? fileName(validationEvidence.dataset) : "—";
  const referenceName = projectValidation?.reference_path ? fileName(projectValidation.reference_path) : "—";
  const exportName = projectExport?.bundle_path ? fileName(projectExport.bundle_path) : "—";
  const renderFailed = terrainRenderState.phase === "error";
  const terrainStageState: StageState = rendererReady
    ? "complete"
    : renderFailed
      ? "failed"
      : meshArtifactReady
        ? "active"
        : "pending";
  const subtitle = activeView === "3D Terrain"
    ? rendererReady
      ? "Interactive terrain renderer ready"
      : renderFailed
        ? "Terrain renderer requires attention"
        : meshArtifactReady
          ? "Preparing terrain renderer"
          : "Terrain mesh not built"
    : geometryReady
      ? `${activeView} analytical workspace`
      : hasInput
        ? "Source imagery loaded"
        : "No scene loaded";

  return (
    <aside className="dw-inspector" aria-label="Scene inspector">
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
            detail: rendererReady ? `rendered LOD ${meshLod}` : renderFailed ? "renderer failed" : meshArtifactReady ? "loading renderer" : "",
          },
          { label: "Export", state: projectExport ? "complete" : "pending", detail: projectExport ? byteLabel(projectExport.bundle_bytes) : "hash-audited ZIP" },
        ]} />
      </section>

      <details className="dw-inspector-disclosure">
        <summary>Geospatial metadata</summary>
        <section className="dw-section dw-section--nested">
          <dl className="dw-property-list">
            <div className="dw-property"><dt>Source</dt><dd title={metadata?.path ?? undefined}>{sourceName}</dd></div>
            <div className="dw-property"><dt>Dimensions</dt><dd>{metadata ? `${metadata.width.toLocaleString()} × ${metadata.height.toLocaleString()}` : "—"}</dd></div>
            <div className="dw-property"><dt>Bands</dt><dd>{metadata?.count ?? "—"}</dd></div>
            <div className="dw-property"><dt>CRS</dt><dd>{metadata?.crs ?? "None"}</dd></div>
            <div className="dw-property"><dt>Ground GSD X</dt><dd>{metadata?.ground_sample_distance_x == null ? "—" : `${metadata.ground_sample_distance_x.toFixed(3)} m`}</dd></div>
            <div className="dw-property"><dt>Ground GSD Y</dt><dd>{metadata?.ground_sample_distance_y == null ? "—" : `${metadata.ground_sample_distance_y.toFixed(3)} m`}</dd></div>
            <div className="dw-property"><dt>Elevation mode</dt><dd>{elevationMode ?? "Not reconstructed"}</dd></div>
            <div className="dw-property"><dt>Source quality</dt><dd>{metadata?.quality.status ?? "—"}</dd></div>
            {metadata?.quality.flags.length ? (
              <div className="dw-property"><dt>Quality flags</dt><dd>{metadata.quality.flags.join(" · ")}</dd></div>
            ) : null}
            {metadata?.quality.off_nadir_degrees != null && (
              <div className="dw-property"><dt>Off-nadir angle</dt><dd>{metadata.quality.off_nadir_degrees.toFixed(1)}°</dd></div>
            )}
            {modelId && <div className="dw-property"><dt>Geometry prior</dt><dd>{modelId}</dd></div>}
            {tileCount !== undefined && <div className="dw-property"><dt>Inference tiles</dt><dd>{tileCount}</dd></div>}
            {harmonizedTiles !== undefined && <div className="dw-property"><dt>Harmonized tiles</dt><dd>{harmonizedTiles}</dd></div>}
            {gcpEvidence && <div className="dw-property"><dt>GCP evidence</dt><dd>{gcpEvidence.point_count} points · SHA {gcpEvidence.sha256.slice(0, 12)}…</dd></div>}
            {meshArtifactReady && <div className="dw-property"><dt>Terrain LOD</dt><dd>{meshLod} · {autoLod ? "auto" : "manual"}</dd></div>}
            {terrainPerformance && rendererReady && (
              <div className="dw-property"><dt>Renderer</dt><dd>{terrainPerformance.fps.toFixed(1)} fps · {terrainPerformance.triangles.toLocaleString()} triangles · {terrainPerformance.drawCalls.toLocaleString()} calls</dd></div>
            )}
          </dl>
        </section>
      </details>

      {(activeTool === "Measure" || activeTool === "Profiles" || probe || analysisBusy) && (
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
                <div className="dw-property"><dt>Measurement quality</dt><dd>{qualityLabel(structureHeight.measurement_quality)}</dd></div>
                <div className="dw-property"><dt>Robust top</dt><dd>{structureHeight.top_elevation_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Local ground</dt><dd>{structureHeight.ground_elevation_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Roof inset</dt><dd>{structureHeight.roof_inset_m.toFixed(2)} m</dd></div>
                <div className="dw-property"><dt>Ground support</dt><dd>{structureHeight.ground_inner_buffer_m.toFixed(2)}–{structureHeight.ground_outer_buffer_m.toFixed(2)} m</dd></div>
                <div className="dw-property"><dt>Ground inliers</dt><dd>{structureHeight.ground_pixels.toLocaleString()} / {structureHeight.ground_candidate_pixels.toLocaleString()} · {(100 * structureHeight.ground_inlier_fraction).toFixed(1)}%</dd></div>
                <div className="dw-property"><dt>Ground coverage</dt><dd>{Math.round(4 * structureHeight.ground_sector_coverage)}/4 sectors</dd></div>
                <div className="dw-property"><dt>Roof dispersion</dt><dd>{structureHeight.roof_dispersion_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Ground fit σ</dt><dd>{structureHeight.ground_residual_sigma_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Height dispersion</dt><dd>{structureHeight.local_height_dispersion_m.toFixed(3)} m</dd></div>
                <div className="dw-property"><dt>Roof-core pixels</dt><dd>{structureHeight.structure_pixels.toLocaleString()}</dd></div>
              </dl>
              <div className="dw-validation-empty">
                <strong>Physical local-ground evidence</strong>
                <p>DepthWizard excludes the roof edge, samples surrounding terrain in metres using trusted GSD, robustly rejects high-object contamination, fits a local ground plane, and evaluates the roof above that fitted terrain. The legacy pixel-ring field is compatibility-only and does not define scientific support.</p>
              </div>
              {structureHeight.warnings.map((warning) => (
                <div className="dw-validation-empty dw-warning-note" key={warning}><strong>Selection warning</strong><p>{warning}</p></div>
              ))}
            </>
          ) : (
            <div className="dw-validation-empty">
              <strong>{structureVertexCount >= 3 ? "Footprint ready" : "Select a footprint"}</strong>
              <p>{structureVertexCount >= 3 ? `${structureVertexCount} vertices selected. Use Undo vertex or Backspace to refine, then measure.` : "Click at least three vertices around one structure on the metric DSM. The selection remains explicit and undoable before measurement."}</p>
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
                <div className="dw-property"><dt>Reliability</dt><dd>{projectValidation.reliability.available ? "measured · native confidence" : "unavailable"}</dd></div>
                {projectValidation.reliability.available && (
                  <div className="dw-property"><dt>Confidence ↔ |error| r</dt><dd>{projectValidation.reliability.confidence_abs_error_pearson_r === null ? "—" : projectValidation.reliability.confidence_abs_error_pearson_r.toFixed(3)}</dd></div>
                )}
              </dl>
              <div className="dw-validation-empty">
                <strong>Reference values stay evaluation-only</strong>
                <p>The reference is aligned downstream of reconstruction and calibration. The exact-file check prevents byte-identical calibration DEM reuse; it does not by itself prove geographic or sensor independence. Native-confidence reliability diagnostics are empirical associations, not calibrated correctness probabilities.</p>
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
