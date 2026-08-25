import { useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { inspectRaster, type RasterMetadata } from "./api";
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

export function App() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const [activeTool, setActiveTool] = useState("Project");
  const [activeView, setActiveView] = useState<(typeof views)[number]>("3D Terrain");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [activeLayer, setActiveLayer] = useState<(typeof layers)[number]>("Texture");
  const [metadata, setMetadata] = useState<RasterMetadata | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [demoReport, setDemoReport] = useState<AbsoluteDemoReport | null>(null);
  const [validationEvidence, setValidationEvidence] = useState<ValidationEvidence | null>(null);
  const meshUrl: string | undefined = demoMode ? "/demo/terrain.glb" : undefined;
  const geometryReady = Boolean(meshUrl);

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

  const projectName = useMemo(
    () => (demoMode ? "Joshimath absolute DSM" : metadata?.path.split(/[\\/]/).pop() ?? "Untitled reconstruction"),
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
      setMetadata(await inspectRaster(selected));
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to inspect imagery");
    } finally {
      setImporting(false);
    }
  };

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
          <button className="dw-btn" onClick={importImagery} disabled={importing}><UploadIcon /> {importing ? "Inspecting…" : "Import imagery"}</button>
          <button className="dw-btn dw-btn--primary" disabled={!metadata}>Export</button>
        </div>
      </header>

      <ToolRail active={activeTool} onChange={setActiveTool} />

      <section className="dw-workspace" aria-label="Scientific workspace">
        <div className="dw-workspace-bar">
          <div className="dw-segmented" role="tablist" aria-label="Data view">
            {views.map((view) => {
              const available = view === "3D Terrain";
              return (
                <button
                  key={view}
                  data-active={activeView === view}
                  disabled={!available}
                  title={available ? undefined : "Available when the corresponding analysis raster is loaded"}
                  onClick={() => available && setActiveView(view)}
                >
                  {view}
                </button>
              );
            })}
          </div>
          <div className="dw-toolbar-group">
            {activeView === "3D Terrain" && cameraModes.map((mode) => (
              <button className="dw-chip" key={mode.id} data-active={cameraMode === mode.id} onClick={() => setCameraMode(mode.id)}>{mode.label}</button>
            ))}
            <span className="dw-toolbar-divider" aria-hidden="true" />
            {layers.map((layer) => {
              const available = layer === "Texture";
              return (
                <button
                  className="dw-chip"
                  key={layer}
                  data-active={activeLayer === layer}
                  disabled={!available}
                  title={available ? undefined : "Layer not generated in this reconstruction yet"}
                  onClick={() => available && setActiveLayer(layer)}
                >
                  {layer}
                </button>
              );
            })}
          </div>
        </div>

        <div className="dw-canvas">
          {activeView === "3D Terrain" && <TerrainViewport meshUrl={meshUrl} cameraMode={cameraMode} />}
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
                <h2>{metadata ? "Source accepted" : "Load a reconstruction project"}</h2>
                <p>
                  {importError
                    ? importError
                    : metadata
                      ? `${metadata.crs ? "Georeferenced input detected. Metric calibration requires DEM/GCP evidence before DepthWizard will claim absolute height." : "No usable CRS detected. DepthWizard will preserve this as relative elevation and will not claim metric height."}`
                      : "Import a single-view RGB remote-sensing image. DepthWizard inspects geospatial metadata before any metric elevation claim is made."}
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="dw-workspace-status">
          <span>{geometryReady ? "Reconstruction loaded · local processing" : metadata ? "Input ready · local processing" : "Ready · local processing"}</span>
          <span>
            {metadata?.crs ?? "Projection —"} · GSD {metadata?.ground_sample_distance_x?.toFixed(3) ?? "—"} m · {demoMode && geometryReady ? "DSM metres" : geometryReady ? "rDSM" : "Elevation —"}
          </span>
        </footer>
      </section>

      <Inspector
        metadata={metadata}
        geometryReady={geometryReady}
        meshReady={geometryReady}
        calibrationReady={demoMode && geometryReady}
        elevationMode={demoMode && geometryReady ? "Absolute DSM (m)" : geometryReady ? "Relative DSM" : undefined}
        modelId={demoReport?.model}
        tileCount={demoReport?.tile_count}
        harmonizedTiles={demoReport?.harmonized_tiles}
        validationEvidence={validationEvidence}
      />
    </main>
  );
}
