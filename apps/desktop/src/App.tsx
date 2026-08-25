import { useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { inspectRaster, type RasterMetadata } from "./api";
import { Inspector } from "./components/Inspector";
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

export function App() {
  const [activeTool, setActiveTool] = useState("Project");
  const [activeView, setActiveView] = useState<(typeof views)[number]>("3D Terrain");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [activeLayer, setActiveLayer] = useState<(typeof layers)[number]>("Texture");
  const [metadata, setMetadata] = useState<RasterMetadata | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const meshUrl: string | undefined = undefined;

  const projectName = useMemo(() => metadata?.path.split(/[\\/]/).pop() ?? "Untitled reconstruction", [metadata]);

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
            {views.map((view) => (
              <button key={view} data-active={activeView === view} onClick={() => setActiveView(view)}>{view}</button>
            ))}
          </div>
          <div className="dw-toolbar-group">
            {activeView === "3D Terrain" && cameraModes.map((mode) => (
              <button className="dw-chip" key={mode.id} data-active={cameraMode === mode.id} onClick={() => setCameraMode(mode.id)}>{mode.label}</button>
            ))}
            {layers.map((layer) => (
              <button className="dw-chip" key={layer} data-active={activeLayer === layer} onClick={() => setActiveLayer(layer)}>{layer}</button>
            ))}
          </div>
        </div>

        <div className="dw-canvas">
          {activeView === "3D Terrain" && <TerrainViewport meshUrl={meshUrl} cameraMode={cameraMode} />}
          {!meshUrl && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>{metadata ? "Source accepted" : "Load a reconstruction project"}</h2>
                <p>
                  {importError
                    ? importError
                    : metadata
                      ? `${metadata.crs ? "Georeferenced input detected. Absolute DSM mode is eligible for metric calibration." : "No usable CRS detected. DepthWizard will preserve this as relative elevation and will not claim metric height."}`
                      : "Import a single-view RGB remote-sensing image. DepthWizard inspects geospatial metadata before any metric elevation claim is made."}
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="dw-workspace-status">
          <span>{metadata ? "Input ready · local processing" : "Ready · local processing"}</span>
          <span>{metadata?.crs ?? "Projection —"} · GSD {metadata?.ground_sample_distance_x?.toFixed(3) ?? "—"} · {metadata ? (metadata.crs ? "Absolute DSM" : "Relative DSM") : "Elevation —"}</span>
        </footer>
      </section>

      <Inspector metadata={metadata} />
    </main>
  );
}
