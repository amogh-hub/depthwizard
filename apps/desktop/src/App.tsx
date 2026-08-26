import { useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  getProjectJob,
  getProjectManifest,
  inspectRaster,
  submitProject,
  type ProjectJobState,
  type ProjectManifest,
  type RasterMetadata,
} from "./api";
import { Inspector, type ValidationEvidence } from "./components/Inspector";
import { ToolRail } from "./components/ToolRail";
import { UploadIcon } from "./components/icons";
import { TerrainViewport, type CameraMode } from "./workspace/TerrainViewport";

const views = ["Optical", "DSM", "3D Terrain", "Reference", "Residual", "Confidence"] as const;
const layers = ["Texture", "DSM", "Slope", "Confidence", "Residual"] as const;
const cameraModes: { id: CameraMode; label: string }[] = [
  { id: "orbit", label: "Orbit" },
  { id: "fly", label: "Fly" },
  { id: "firstPerson", label: "First person" },
  { id: "topDown", label: "Top down" },
];

const terminalJobStates = new Set(["waiting_for_calibration", "complete", "failed"]);

type AbsoluteDemoReport = {
  status: string;
  scene: string;
  purpose: string;
  model: string;
  device: string;
  shape: [number, number];
  tile_count: number;
  harmonized_tiles: number;
  crs: string | null;
  gsd_x_m: number;
  gsd_y_m: number;
  dsm: string;
  imagery: {
    source: string;
    path: string;
  };
  calibration: {
    method?: string;
    scale?: number;
    offset?: number;
    orientation_flipped?: boolean;
    anchor_correlation_before?: number;
    anchor_correlation_after?: number;
  };
};

type BenchmarkReport = {
  dataset: string;
  protocol: string;
  results: Array<{
    anchor_count: number;
    heldout_pixels: number;
    metrics: {
      rmse_m: number;
      mae_m: number;
      pearson_r: number | null;
    };
  }>;
};

function stageNumber(manifest: ProjectManifest | null, stage: string, key: string): number | undefined {
  const value = manifest?.stages[stage]?.details[key];
  return typeof value === "number" ? value : undefined;
}

function estimatorModel(manifest: ProjectManifest | null): string | undefined {
  const value = manifest?.estimator.selected_model_id;
  return typeof value === "string" ? value : undefined;
}

export function App() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const [activeTool, setActiveTool] = useState("Project");
  const [activeView, setActiveView] = useState<(typeof views)[number]>("3D Terrain");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [activeLayer, setActiveLayer] = useState<(typeof layers)[number]>("Texture");
  const [metadata, setMetadata] = useState<RasterMetadata | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [projectJob, setProjectJob] = useState<ProjectJobState | null>(null);
  const [projectManifest, setProjectManifest] = useState<ProjectManifest | null>(null);
  const [submittingProject, setSubmittingProject] = useState(false);
  const [demoReport, setDemoReport] = useState<AbsoluteDemoReport | null>(null);
  const [validationEvidence, setValidationEvidence] = useState<ValidationEvidence | null>(null);
  const meshUrl: string | undefined = demoMode ? "/demo/terrain.glb" : undefined;

  useEffect(() => {
    if (!demoMode) return;
    let cancelled = false;
    Promise.all([
      fetch("/demo/absolute_demo_report.json").then((response) => {
        if (!response.ok) throw new Error("Unable to load absolute-DSM engineering report");
        return response.json() as Promise<AbsoluteDemoReport>;
      }),
      fetch("/demo/benchmark_report.json").then((response) => {
        if (!response.ok) throw new Error("Unable to load held-out benchmark report");
        return response.json() as Promise<BenchmarkReport>;
      }),
    ])
      .then(([absoluteReport, benchmark]) => {
        if (cancelled) return;
        setDemoReport(absoluteReport);
        setMetadata({
          path: absoluteReport.imagery.path,
          width: absoluteReport.shape[1],
          height: absoluteReport.shape[0],
          count: 3,
          dtype: "source RGB",
          crs: absoluteReport.crs,
          transform: null,
          nodata: null,
          ground_sample_distance_x: absoluteReport.gsd_x_m,
          ground_sample_distance_y: absoluteReport.gsd_y_m,
        });
        const preferred = benchmark.results.find((item) => item.anchor_count === 64)
          ?? benchmark.results.at(-1);
        if (preferred) {
          setValidationEvidence({
            dataset: benchmark.dataset,
            protocol: benchmark.protocol,
            anchorCount: preferred.anchor_count,
            heldoutPixels: preferred.heldout_pixels,
            rmseM: preferred.metrics.rmse_m,
            maeM: preferred.metrics.mae_m,
            pearsonR: preferred.metrics.pearson_r,
          });
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setImportError(error instanceof Error ? error.message : "Unable to load demo evidence");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [demoMode]);

  useEffect(() => {
    if (!projectJob || terminalJobStates.has(projectJob.status)) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getProjectJob(projectJob.job_id)
        .then(async (next) => {
          if (cancelled) return;
          setProjectJob(next);
          if (terminalJobStates.has(next.status)) {
            window.clearInterval(timer);
            const manifest = await getProjectManifest(next.project_dir);
            if (!cancelled) setProjectManifest(manifest);
          }
        })
        .catch((error: unknown) => {
          if (!cancelled) {
            window.clearInterval(timer);
            setImportError(error instanceof Error ? error.message : "Unable to read project status");
          }
        });
    }, 750);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [projectJob]);

  const geometryReady = demoMode
    ? Boolean(meshUrl)
    : Boolean(projectManifest?.artifacts.rdsm);
  const calibrationReady = demoMode
    ? Boolean(meshUrl)
    : Boolean(projectManifest?.artifacts.dsm);
  const meshReady = Boolean(meshUrl);
  const processing = projectJob?.status === "queued" || projectJob?.status === "running";
  const waitingForCalibration = projectJob?.status === "waiting_for_calibration";

  const projectName = useMemo(
    () => (
      demoMode
        ? "Joshimath absolute DSM"
        : metadata?.path.split(/[\\/]/).pop() ?? "Untitled reconstruction"
    ),
    [demoMode, metadata],
  );

  const importImagery = async () => {
    setImportError(null);
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Remote-sensing imagery", extensions: ["png", "jpg", "jpeg", "tif", "tiff"] }],
    });
    if (!selected || Array.isArray(selected)) return;
    try {
      setImporting(true);
      const nextMetadata = await inspectRaster(selected);
      setMetadata(nextMetadata);
      setProjectDir(null);
      setProjectJob(null);
      setProjectManifest(null);
      setValidationEvidence(null);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to inspect imagery");
    } finally {
      setImporting(false);
    }
  };

  const reconstruct = async () => {
    if (!metadata) return;
    setImportError(null);
    const selectedDir = await open({ multiple: false, directory: true });
    if (!selectedDir || Array.isArray(selectedDir)) return;
    try {
      setSubmittingProject(true);
      const next = await submitProject({
        source: metadata.path,
        output_dir: selectedDir,
        requested_output: metadata.crs ? null : "rdsm",
      });
      setProjectDir(selectedDir);
      setProjectManifest(null);
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start reconstruction");
    } finally {
      setSubmittingProject(false);
    }
  };

  const addDemEvidence = async () => {
    if (!metadata || !projectDir) return;
    setImportError(null);
    const dem = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Metric DEM", extensions: ["tif", "tiff"] }],
    });
    if (!dem || Array.isArray(dem)) return;
    try {
      setSubmittingProject(true);
      const next = await submitProject({
        source: metadata.path,
        output_dir: projectDir,
        dem_path: dem,
        requested_output: "dsm",
      });
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start metric calibration");
    } finally {
      setSubmittingProject(false);
    }
  };

  const normalStatus = projectJob?.error
    ?? (projectJob?.status === "waiting_for_calibration"
      ? "Geometry ready · metric evidence required"
      : projectJob?.status === "complete"
        ? "Production products ready"
        : projectJob?.status === "failed"
          ? "Processing failed"
          : processing
            ? "Production processing…"
            : geometryReady
              ? "Reconstruction loaded · local processing"
              : metadata
                ? "Input ready · local processing"
                : "Ready · local processing");

  return (
    <main className="dw-app">
      <header className="dw-topbar">
        <div className="dw-brand">
          <span className="dw-mark">DW</span>
          <span>DepthWizard</span>
        </div>
        <div className="dw-project-title">
          <strong>{projectName}</strong>
          <span>ISRO · SIH26175</span>
        </div>
        <div className="dw-top-actions">
          <button className="dw-btn" onClick={importImagery} disabled={importing || processing}>
            <UploadIcon /> {importing ? "Inspecting…" : "Import imagery"}
          </button>
          {!demoMode && metadata && !projectDir && (
            <button className="dw-btn dw-btn--primary" onClick={reconstruct} disabled={submittingProject}>
              {submittingProject ? "Starting…" : "Reconstruct"}
            </button>
          )}
          {!demoMode && waitingForCalibration && (
            <button className="dw-btn dw-btn--primary" onClick={addDemEvidence} disabled={submittingProject}>
              {submittingProject ? "Starting…" : "Add DEM evidence"}
            </button>
          )}
          <button className="dw-btn dw-btn--primary" disabled>Export</button>
        </div>
      </header>

      <ToolRail active={activeTool} onChange={setActiveTool} />

      <section className="dw-workspace" aria-label="Scientific workspace">
        <div className="dw-workspace-bar">
          <div className="dw-segmented" role="tablist" aria-label="Data view">
            {views.map((view) => {
              const available = view === "3D Terrain" && meshReady;
              return (
                <button
                  key={view}
                  data-active={activeView === view}
                  disabled={!available}
                  title={available ? undefined : "Enabled only when its real analysis artifact is available"}
                  onClick={() => available && setActiveView(view)}
                >
                  {view}
                </button>
              );
            })}
          </div>
          <div className="dw-toolbar-group">
            {activeView === "3D Terrain" && cameraModes.map((mode) => (
              <button
                className="dw-chip"
                key={mode.id}
                data-active={cameraMode === mode.id}
                disabled={!meshReady}
                onClick={() => meshReady && setCameraMode(mode.id)}
              >
                {mode.label}
              </button>
            ))}
            <span className="dw-toolbar-divider" aria-hidden="true" />
            {layers.map((layer) => {
              const available = layer === "Texture" && meshReady;
              return (
                <button
                  className="dw-chip"
                  key={layer}
                  data-active={activeLayer === layer}
                  disabled={!available}
                  title={available ? undefined : "Layer is enabled only after its real product is loaded"}
                  onClick={() => available && setActiveLayer(layer)}
                >
                  {layer}
                </button>
              );
            })}
          </div>
        </div>

        <div className="dw-canvas">
          {activeView === "3D Terrain" && meshReady && (
            <TerrainViewport meshUrl={meshUrl} cameraMode={cameraMode} />
          )}
          {meshUrl && (
            <>
              <div className="dw-canvas-context">
                <strong>{demoMode ? "Absolute DSM" : "Relative DSM"}</strong>
                <span>
                  {demoMode
                    ? `${demoReport?.scene ?? "India scene"} · ${demoReport?.model ?? "DA3MONO-LARGE"}`
                    : "DA3MONO-LARGE · textured terrain"}
                </span>
              </div>
              <div className="dw-north-indicator" aria-label="North indicator"><strong>N</strong><span>↑</span></div>
              <div className="dw-scene-badge">
                <strong>{demoMode ? "Metric elevation" : "Relative elevation"}</strong>
                <span>
                  {demoMode
                    ? "DEM-calibrated · metres · engineering path"
                    : "dimensionless relative surface height · not metric height"}
                </span>
              </div>
            </>
          )}
          {!meshUrl && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>
                  {processing
                    ? "Reconstructing scene"
                    : waitingForCalibration
                      ? "Relative geometry complete"
                      : calibrationReady
                        ? "Metric DSM products ready"
                        : geometryReady
                          ? "Relative DSM ready"
                          : metadata
                            ? "Source accepted"
                            : "Load a reconstruction project"}
                </h2>
                <p>
                  {importError
                    ? importError
                    : waitingForCalibration
                      ? "This georeferenced project is intentionally paused before any metric-height claim. Add a DEM now; sparse GCP workflow will be exposed by the calibration workspace."
                      : calibrationReady
                        ? "DepthWizard completed evidence-calibrated metric elevation. Analytical raster and 3D views remain disabled until the corresponding real artifacts are integrated."
                        : geometryReady
                          ? "DepthWizard completed a truthful dimensionless relative surface model. No metric elevation has been invented."
                          : metadata
                            ? `${metadata.crs ? "Georeferenced input detected. Reconstruct once, then DepthWizard will require DEM/GCP evidence before claiming absolute height." : "No usable CRS detected. DepthWizard will preserve this as relative elevation and will not claim metric height."}`
                            : "Import a single-view RGB remote-sensing image. DepthWizard inspects geospatial metadata before any metric elevation claim is made."}
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="dw-workspace-status">
          <span>{demoMode ? "Reconstruction loaded · local processing" : normalStatus}</span>
          <span>
            {metadata?.crs ?? "Projection —"} · GSD {metadata?.ground_sample_distance_x?.toFixed(3) ?? "—"} m · {calibrationReady ? "DSM metres" : geometryReady ? "rDSM" : "Elevation —"}
          </span>
        </footer>
      </section>

      <Inspector
        metadata={metadata}
        geometryReady={geometryReady}
        meshReady={meshReady}
        calibrationReady={calibrationReady}
        elevationMode={calibrationReady ? "Absolute DSM (m)" : geometryReady ? "Relative DSM" : undefined}
        modelId={demoReport?.model ?? estimatorModel(projectManifest)}
        tileCount={demoReport?.tile_count ?? stageNumber(projectManifest, "geometry", "tile_count")}
        harmonizedTiles={demoReport?.harmonized_tiles ?? stageNumber(projectManifest, "geometry", "harmonized_tiles")}
        validationEvidence={validationEvidence}
      />
    </main>
  );
}
