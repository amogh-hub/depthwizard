import { useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  buildProjectMesh,
  getProjectJob,
  getProjectManifest,
  getProjectMesh,
  getProjectMeshUrl,
  getProjectPreviewUrl,
  getProjectValidation,
  inspectRaster,
  probeProject,
  sampleProjectProfile,
  submitProject,
  validateProjectReference,
  type NormalizedPoint,
  type ProjectJobState,
  type ProjectManifest,
  type ProjectMeshReport,
  type ProjectPreviewLayer,
  type ProjectProbeResult,
  type ProjectProfileResult,
  type RasterMetadata,
  type ReferenceValidationReport,
} from "./api";
import { Inspector, type ValidationEvidence } from "./components/Inspector";
import { ToolRail } from "./components/ToolRail";
import { UploadIcon } from "./components/icons";
import { ComparisonViewport } from "./workspace/ComparisonViewport";
import { RasterAnalysisViewport } from "./workspace/RasterAnalysisViewport";
import { TerrainViewport, type CameraMode } from "./workspace/TerrainViewport";

const views = ["Optical", "DSM", "3D Terrain", "Reference", "Residual", "Confidence"] as const;
const layers = ["Texture", "DSM", "Slope", "Confidence", "Residual"] as const;
const cameraModes: { id: CameraMode; label: string }[] = [
  { id: "orbit", label: "Orbit" },
  { id: "fly", label: "Fly" },
  { id: "firstPerson", label: "First person" },
  { id: "topDown", label: "Top down" },
];
const exaggerations = [1, 1.5, 2, 3] as const;
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

function projectPreviewLayer(
  view: (typeof views)[number],
  activeLayer: (typeof layers)[number],
  manifest: ProjectManifest | null,
): ProjectPreviewLayer | null {
  if (!manifest) return null;
  if (view === "Optical") return "optical";
  if (view === "DSM") {
    if (activeLayer === "Slope" && manifest.artifacts.slope) return "slope";
    if (manifest.artifacts.dsm) return "dsm";
    if (manifest.artifacts.rdsm) return "rdsm";
    return null;
  }
  if (view === "Reference" && manifest.artifacts.reference) return "reference";
  if (view === "Residual" && manifest.artifacts.residual) return "residual";
  if (view === "Confidence" && manifest.artifacts.confidence) return "confidence";
  return null;
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
  const [projectValidation, setProjectValidation] = useState<ReferenceValidationReport | null>(null);
  const [validatingReference, setValidatingReference] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [comparisonUrl, setComparisonUrl] = useState<string | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [probe, setProbe] = useState<ProjectProbeResult | null>(null);
  const [lineStart, setLineStart] = useState<NormalizedPoint | null>(null);
  const [lineEnd, setLineEnd] = useState<NormalizedPoint | null>(null);
  const [measurement, setMeasurement] = useState<ProjectProfileResult | null>(null);
  const [profile, setProfile] = useState<ProjectProfileResult | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [projectMesh, setProjectMesh] = useState<ProjectMeshReport | null>(null);
  const [projectMeshUrl, setProjectMeshUrl] = useState<string | null>(null);
  const [buildingMesh, setBuildingMesh] = useState(false);
  const [meshLod, setMeshLod] = useState(0);
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(1);
  const meshUrl: string | undefined = demoMode ? "/demo/terrain.glb" : projectMeshUrl ?? undefined;

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

  const projectJobId = projectJob?.job_id;
  useEffect(() => {
    if (!projectJobId || (projectJob && terminalJobStates.has(projectJob.status))) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getProjectJob(projectJobId)
        .then(async (next) => {
          if (cancelled) return;
          if (terminalJobStates.has(next.status)) {
            window.clearInterval(timer);
            const manifest = await getProjectManifest(next.project_dir);
            if (cancelled) return;
            let validation: ReferenceValidationReport | null = null;
            let mesh: ProjectMeshReport | null = null;
            if (manifest.artifacts.metrics) {
              validation = await getProjectValidation(next.project_dir).catch(() => null);
            }
            if (manifest.artifacts.mesh_manifest) {
              mesh = await getProjectMesh(next.project_dir).catch(() => null);
            }
            if (cancelled) return;
            setProjectValidation(validation);
            setProjectMesh(mesh);
            setProjectManifest(manifest);
            setProjectJob(next);
            setActiveLayer(manifest.artifacts.dsm ? "DSM" : "Texture");
            setActiveView(manifest.artifacts.dsm || manifest.artifacts.rdsm ? "DSM" : "Optical");
            return;
          }
          setProjectJob(next);
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
  }, [projectJobId]);

  const geometryReady = demoMode
    ? Boolean(meshUrl)
    : Boolean(projectManifest?.artifacts.rdsm || projectManifest?.artifacts.dsm);
  const calibrationReady = demoMode
    ? Boolean(meshUrl)
    : Boolean(projectManifest?.artifacts.dsm);
  const meshReady = Boolean(meshUrl);
  const processing = projectJob?.status === "queued" || projectJob?.status === "running";
  const waitingForCalibration = projectJob?.status === "waiting_for_calibration";
  const previewLayer = projectPreviewLayer(activeView, activeLayer, projectManifest);
  const compareActive = activeTool === "Compare" && Boolean(projectValidation) && calibrationReady;
  const analystInteractive = !demoMode && activeView !== "3D Terrain" && Boolean(projectDir) && geometryReady;
  const projectAnalystInteractive = !demoMode && Boolean(projectDir) && geometryReady;

  useEffect(() => {
    if (demoMode || !projectDir || !projectMesh) {
      setProjectMeshUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    void getProjectMeshUrl(projectDir, meshLod)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setProjectMeshUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return url;
        });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setProjectMeshUrl((current) => {
            if (current) URL.revokeObjectURL(current);
            return null;
          });
          setImportError(error instanceof Error ? error.message : "Unable to load project terrain LOD");
        }
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [demoMode, meshLod, projectDir, projectMesh?.build_config_sha256]);

  useEffect(() => {
    if (demoMode || activeView === "3D Terrain" || !projectDir || !previewLayer) {
      setPreviewUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setPreviewLoading(false);
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    setPreviewLoading(true);
    void getProjectPreviewUrl(projectDir, previewLayer)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setPreviewUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return url;
        });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setPreviewUrl((current) => {
            if (current) URL.revokeObjectURL(current);
            return null;
          });
          setImportError(error instanceof Error ? error.message : "Unable to render project layer");
        }
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [activeView, demoMode, previewLayer, projectDir, projectManifest?.updated_at_utc]);

  useEffect(() => {
    if (!compareActive || !projectDir) {
      setComparisonUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setComparisonLoading(false);
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    setComparisonLoading(true);
    void getProjectPreviewUrl(projectDir, "reference")
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setComparisonUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return url;
        });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setImportError(error instanceof Error ? error.message : "Unable to load comparison reference");
        }
      })
      .finally(() => {
        if (!cancelled) setComparisonLoading(false);
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [compareActive, projectDir, projectManifest?.updated_at_utc]);

  useEffect(() => {
    setLineStart(null);
    setLineEnd(null);
    setMeasurement(null);
    setProfile(null);
    if (activeTool === "Validation" && projectValidation) {
      setActiveView("Residual");
      setActiveLayer("Residual");
    } else if (activeTool === "Compare" && projectValidation) {
      setActiveView("DSM");
      setActiveLayer("DSM");
    } else if ((activeTool === "Measure" || activeTool === "Profiles") && geometryReady) {
      setActiveView("DSM");
      setActiveLayer("DSM");
    }
  }, [activeTool, geometryReady, projectValidation]);

  const projectName = useMemo(
    () => (
      demoMode
        ? "Joshimath absolute DSM"
        : metadata?.path.split(/[\\/]/).pop() ?? "Untitled reconstruction"
    ),
    [demoMode, metadata],
  );

  const resetAnalysis = () => {
    setProbe(null);
    setLineStart(null);
    setLineEnd(null);
    setMeasurement(null);
    setProfile(null);
    setAnalysisBusy(false);
  };

  const clearProjectMesh = () => {
    setProjectMesh(null);
    setMeshLod(0);
    setVerticalExaggeration(1);
    setProjectMeshUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
  };

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
      setProjectValidation(null);
      clearProjectMesh();
      resetAnalysis();
      setActiveLayer("Texture");
      setActiveView("3D Terrain");
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
      setProjectValidation(null);
      clearProjectMesh();
      resetAnalysis();
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
      clearProjectMesh();
      resetAnalysis();
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start metric calibration");
    } finally {
      setSubmittingProject(false);
    }
  };

  const validateReference = async () => {
    if (!projectDir || !calibrationReady || projectValidation) return;
    setImportError(null);
    const reference = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Reference DSM", extensions: ["tif", "tiff"] }],
    });
    if (!reference || Array.isArray(reference)) return;
    try {
      setValidatingReference(true);
      const report = await validateProjectReference(projectDir, reference);
      const manifest = await getProjectManifest(projectDir);
      setProjectValidation(report);
      setProjectManifest(manifest);
      resetAnalysis();
      setActiveLayer("Residual");
      setActiveView("Residual");
      setActiveTool("Validation");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to validate reference DSM");
    } finally {
      setValidatingReference(false);
    }
  };

  const buildTerrain = async () => {
    if (!projectDir || !geometryReady || demoMode) return;
    setImportError(null);
    try {
      setBuildingMesh(true);
      const report = await buildProjectMesh(projectDir);
      const manifest = await getProjectManifest(projectDir);
      setProjectMesh(report);
      setProjectManifest(manifest);
      setMeshLod(0);
      setVerticalExaggeration(1);
      setActiveLayer("Texture");
      setActiveView("3D Terrain");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to build project terrain mesh");
    } finally {
      setBuildingMesh(false);
    }
  };

  const analyzePoint = async (point: NormalizedPoint) => {
    if (!projectDir || !geometryReady) return;
    setImportError(null);
    setAnalysisBusy(true);
    try {
      setProbe(await probeProject(projectDir, point));
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to sample project products");
    } finally {
      setAnalysisBusy(false);
    }
  };

  const analyzeRasterPoint = async (point: NormalizedPoint) => {
    if (!projectDir || !geometryReady) return;
    if (activeTool !== "Measure" && activeTool !== "Profiles") {
      await analyzePoint(point);
      return;
    }

    if (!lineStart || lineEnd) {
      setLineStart(point);
      setLineEnd(null);
      setMeasurement(null);
      setProfile(null);
      await analyzePoint(point);
      return;
    }

    setLineEnd(point);
    setImportError(null);
    setAnalysisBusy(true);
    try {
      const [nextProbe, transect] = await Promise.all([
        probeProject(projectDir, point),
        sampleProjectProfile(projectDir, lineStart, point, activeTool === "Profiles" ? 160 : 2),
      ]);
      setProbe(nextProbe);
      if (activeTool === "Profiles") {
        setProfile(transect);
        setMeasurement(null);
      } else {
        setMeasurement(transect);
        setProfile(null);
      }
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to compute analyst transect");
    } finally {
      setAnalysisBusy(false);
    }
  };

  const normalStatus = projectJob?.error
    ?? (buildingMesh
      ? "Building persistent terrain LODs…"
      : analysisBusy
        ? "Sampling persisted analytical products…"
        : projectValidation
          ? `Reference validation ready · RMSE ${projectValidation.elevation.rmse_m.toFixed(3)} m`
          : projectMesh
            ? `3D terrain ready · ${projectMesh.lods.length} LODs`
            : projectJob?.status === "waiting_for_calibration"
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

  const viewAvailable = (view: (typeof views)[number]): boolean => {
    if (view === "3D Terrain") return meshReady;
    if (demoMode) return false;
    if (view === "Optical") return Boolean(projectManifest);
    if (view === "DSM") return geometryReady;
    if (view === "Reference" || view === "Residual") return Boolean(projectValidation);
    if (view === "Confidence") return Boolean(projectManifest?.artifacts.confidence);
    return false;
  };

  const chooseView = (view: (typeof views)[number]) => {
    if (!viewAvailable(view)) return;
    setActiveView(view);
    if (view === "Optical") setActiveLayer("Texture");
    if (view === "DSM") setActiveLayer("DSM");
    if (view === "Residual") setActiveLayer("Residual");
    if (view === "Confidence") setActiveLayer("Confidence");
  };

  const layerAvailable = (layer: (typeof layers)[number]): boolean => {
    if (layer === "Texture") return meshReady || Boolean(projectManifest);
    if (layer === "DSM") return geometryReady;
    if (layer === "Slope") return Boolean(projectManifest?.artifacts.slope);
    if (layer === "Confidence") return Boolean(projectManifest?.artifacts.confidence);
    if (layer === "Residual") return Boolean(projectValidation);
    return false;
  };

  const chooseLayer = (layer: (typeof layers)[number]) => {
    if (!layerAvailable(layer)) return;
    setActiveLayer(layer);
    if (activeView === "3D Terrain" && meshReady) return;
    if (layer === "Texture") setActiveView("Optical");
    if (layer === "DSM" || layer === "Slope") setActiveView("DSM");
    if (layer === "Confidence") setActiveView("Confidence");
    if (layer === "Residual") setActiveView("Residual");
  };

  const analysisHint = activeTool === "Measure"
    ? lineStart && !lineEnd ? "Select endpoint B" : "Select point A, then point B"
    : activeTool === "Profiles"
      ? lineStart && !lineEnd ? "Select transect endpoint B" : "Select transect endpoints A → B"
      : activeTool === "Compare"
        ? "Swipe reference against prediction · click to synchronize values"
        : activeView === "3D Terrain"
          ? "Click terrain to inspect synchronized project values"
          : "Click raster to inspect synchronized project values";

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
          <button className="dw-btn" onClick={importImagery} disabled={importing || processing || validatingReference || buildingMesh}>
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
          {!demoMode && geometryReady && (
            <button className="dw-btn" onClick={buildTerrain} disabled={buildingMesh || processing || Boolean(projectMesh)}>
              {buildingMesh ? "Building 3D…" : projectMesh ? "3D terrain ready" : "Build 3D terrain"}
            </button>
          )}
          {!demoMode && calibrationReady && (
            <button
              className="dw-btn"
              onClick={validateReference}
              disabled={validatingReference || Boolean(projectValidation)}
              title={projectValidation ? "This project already preserves one completed reference validation" : undefined}
            >
              {validatingReference ? "Validating…" : projectValidation ? "Reference validated" : "Validate reference"}
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
              const available = viewAvailable(view);
              return (
                <button
                  key={view}
                  data-active={activeView === view}
                  disabled={!available}
                  title={available ? undefined : "Enabled only when its real analysis artifact is available"}
                  onClick={() => chooseView(view)}
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
            {activeView === "3D Terrain" && projectMesh?.lods.map((lod) => (
              <button
                className="dw-chip"
                key={lod.level}
                data-active={meshLod === lod.level}
                onClick={() => setMeshLod(lod.level)}
                title={`${lod.vertices.toLocaleString()} vertices · ${lod.faces.toLocaleString()} faces`}
              >
                LOD {lod.level}
              </button>
            ))}
            {activeView === "3D Terrain" && meshReady && !demoMode && exaggerations.map((value) => (
              <button
                className="dw-chip"
                key={value}
                data-active={verticalExaggeration === value}
                onClick={() => setVerticalExaggeration(value)}
                title="Display-only vertical exaggeration; source elevation values are unchanged"
              >
                {value}× Z
              </button>
            ))}
            {activeView === "3D Terrain" && projectAnalystInteractive && (
              <span className="dw-analysis-hint">{analysisHint}</span>
            )}
            {activeView !== "3D Terrain" && analystInteractive && (
              <span className="dw-analysis-hint">{analysisHint}</span>
            )}
            {(lineStart || probe) && (
              <button className="dw-chip" onClick={resetAnalysis}>Clear analysis</button>
            )}
            <span className="dw-toolbar-divider" aria-hidden="true" />
            {layers.map((layer) => (
              <button
                className="dw-chip"
                key={layer}
                data-active={activeLayer === layer}
                disabled={!layerAvailable(layer)}
                title={layerAvailable(layer) ? undefined : "Layer is enabled only after its real product is loaded"}
                onClick={() => chooseLayer(layer)}
              >
                {layer}
              </button>
            ))}
          </div>
        </div>

        <div className="dw-canvas">
          {activeView === "3D Terrain" && meshReady && (
            <TerrainViewport
              meshUrl={meshUrl}
              cameraMode={cameraMode}
              verticalExaggeration={verticalExaggeration}
              cursorPoint={probe?.point}
              onSelectPoint={projectAnalystInteractive ? analyzeRasterPoint : undefined}
            />
          )}
          {activeView !== "3D Terrain" && compareActive && previewUrl && comparisonUrl && (
            <ComparisonViewport
              predictionUrl={previewUrl}
              referenceUrl={comparisonUrl}
              cursorPoint={probe?.point}
              onSelectPoint={analyzeRasterPoint}
            />
          )}
          {activeView !== "3D Terrain" && previewUrl && !(compareActive && comparisonUrl) && (
            <RasterAnalysisViewport
              src={previewUrl}
              alt={`${previewLayer ?? activeView} scientific raster`}
              interactive={analystInteractive}
              cursorPoint={probe?.point}
              lineStart={activeTool === "Measure" || activeTool === "Profiles" ? lineStart : null}
              lineEnd={activeTool === "Measure" || activeTool === "Profiles" ? lineEnd : null}
              onSelectPoint={analyzeRasterPoint}
            />
          )}
          {activeView !== "3D Terrain" && (previewLoading || comparisonLoading) && !previewUrl && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>Loading scientific layer</h2>
                <p>Rendering a bounded local preview from the persisted project raster. Source products remain unchanged.</p>
              </div>
            </div>
          )}
          {(meshUrl || previewUrl) && (
            <>
              <div className="dw-canvas-context">
                <strong>
                  {compareActive
                    ? "Prediction ↔ reference comparison"
                    : activeView === "3D Terrain"
                      ? projectMesh?.surface_product === "dsm" || demoMode ? "Absolute DSM" : "Relative DSM"
                      : previewLayer === "residual"
                        ? "Prediction − reference"
                        : previewLayer === "reference"
                          ? "Aligned reference DSM"
                          : previewLayer === "slope"
                            ? "Surface slope"
                            : previewLayer === "confidence"
                              ? "Model-native confidence"
                              : previewLayer === "optical"
                                ? "Optical RGB"
                                : calibrationReady ? "Absolute DSM" : "Relative DSM"}
                </strong>
                <span>
                  {compareActive
                    ? "evaluation-only reference · synchronized analyst cursor"
                    : activeView === "3D Terrain"
                      ? demoMode
                        ? `${demoReport?.scene ?? "India scene"} · ${demoReport?.model ?? "DA3MONO-LARGE"}`
                        : `${estimatorModel(projectManifest) ?? "DA3MONO-LARGE"} · persistent textured LOD ${meshLod} · ${verticalExaggeration}× display Z`
                      : previewLayer === "residual"
                        ? `${projectValidation?.valid_pixels.toLocaleString() ?? "—"} valid pixels · metres`
                        : previewLayer === "reference"
                          ? "evaluation-only · aligned to prediction grid"
                          : previewLayer === "confidence"
                            ? "not probability calibrated"
                            : estimatorModel(projectManifest) ?? "persisted project raster"}
                </span>
              </div>
              <div className="dw-north-indicator" aria-label="North indicator"><strong>N</strong><span>↑</span></div>
              <div className="dw-scene-badge">
                <strong>
                  {analysisBusy
                    ? "Sampling analytical products"
                    : activeView === "3D Terrain" && probe?.surface.available
                      ? `${probe.surface.value?.toFixed(2) ?? "—"} ${probe.surface.units ?? ""}`
                      : previewLayer === "residual"
                        ? `RMSE ${projectValidation?.elevation.rmse_m.toFixed(3) ?? "—"} m`
                        : demoMode || calibrationReady ? "Metric elevation" : "Relative elevation"}
                </strong>
                <span>
                  {activeView === "3D Terrain" && projectMesh
                    ? `LOD ${meshLod} · ${projectMesh.horizontal_units} XY · ${projectMesh.vertical_units} Z · ${projectMesh.relief.toFixed(2)} ${projectMesh.vertical_units} relief`
                    : previewLayer === "residual"
                      ? `MAE ${projectValidation?.elevation.mae_m.toFixed(3) ?? "—"} m · P95 ${projectValidation?.elevation.p95_abs_error_m.toFixed(3) ?? "—"} m`
                      : activeTool === "Measure" && measurement
                        ? `${measurement.horizontal_distance_m?.toFixed(2) ?? measurement.horizontal_distance_pixels.toFixed(2)} ${measurement.horizontal_distance_m === null ? "px" : "m"} · Δz ${measurement.vertical_delta?.toFixed(2) ?? "—"} ${measurement.vertical_units ?? ""}`
                        : activeTool === "Profiles" && profile
                          ? `${profile.sample_count} samples · ${profile.horizontal_distance_m?.toFixed(2) ?? profile.horizontal_distance_pixels.toFixed(2)} ${profile.horizontal_distance_m === null ? "px" : "m"}`
                          : demoMode
                            ? "DEM-calibrated · metres · engineering path"
                            : calibrationReady
                              ? "evidence-calibrated · metres"
                              : "dimensionless relative surface height · not metric height"}
                </span>
              </div>
            </>
          )}
          {!meshUrl && !previewUrl && !previewLoading && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>
                  {buildingMesh
                    ? "Building analytical terrain"
                    : processing
                      ? "Reconstructing scene"
                      : waitingForCalibration
                        ? "Relative geometry complete"
                        : geometryReady && !projectMesh
                          ? "Terrain products ready for 3D"
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
                    : buildingMesh
                      ? "Generating persistent hashed GLB LODs from the already-produced surface and source RGB. Validation reference data is not used."
                      : geometryReady && !projectMesh
                        ? "Build the persistent terrain LOD pyramid to enable Orbit, Fly, First Person, Top Down and synchronized 3D probing."
                        : waitingForCalibration
                          ? "This georeferenced project is intentionally paused before any metric-height claim. Add a DEM now; sparse GCP workflow remains an explicit calibration path."
                          : calibrationReady
                            ? "DepthWizard completed evidence-calibrated metric elevation. Load a separate reference DSM to create evaluation-only residuals and validation metrics."
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
        projectValidation={projectValidation}
        activeTool={activeTool}
        probe={probe}
        measurement={measurement}
        profile={profile}
        analysisBusy={analysisBusy}
      />
    </main>
  );
}
