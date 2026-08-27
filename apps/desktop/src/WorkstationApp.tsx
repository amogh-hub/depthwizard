import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  buildProjectExport,
  buildProjectMesh,
  estimateProjectStructureHeight,
  getProjectExport,
  getProjectExportUrl,
  getProjectJob,
  getProjectLayerLegend,
  getProjectManifest,
  getProjectMesh,
  getProjectMeshUrl,
  getProjectPreviewUrl,
  getProjectValidation,
  inspectGroundControlPoints,
  inspectRaster,
  probeProject,
  sampleProjectProfile,
  submitProject,
  validateProjectReference,
  type GroundControlPointFileReport,
  type NormalizedPoint,
  type ProjectExportReport,
  type ProjectJobState,
  type ProjectLayerLegend,
  type ProjectManifest,
  type ProjectMeshReport,
  type ProjectPreviewLayer,
  type ProjectProbeResult,
  type ProjectProfileResult,
  type ProjectStructureHeightResult,
  type RasterMetadata,
  type ReferenceValidationReport,
} from "./api";
import { Inspector, type ValidationEvidence } from "./components/Inspector";
import { ScientificLegend } from "./components/ScientificLegend";
import { ToolRail } from "./components/ToolRail";
import { UploadIcon } from "./components/icons";
import { ComparisonViewport } from "./workspace/ComparisonViewport";
import {
  RasterAnalysisViewport,
  type RasterInteractionMode,
} from "./workspace/RasterAnalysisViewport";
import {
  TerrainViewport,
  type CameraMode,
  type TerrainPerformance,
  type TerrainRenderState,
} from "./workspace/TerrainViewport";
import {
  DEFAULT_RASTER_VIEW_STATE,
  type RasterViewState,
} from "./workspace/rasterViewport";
import {
  lodPressureDelta,
  nextAutoLod,
  validTerrainTelemetry,
} from "./workspace/terrainPolicy";
import {
  activeWorkspaceStatus,
  compareToolAvailable,
  terrainControlsEnabled,
} from "./workspace/workstationPolicy";

const views = ["Optical", "DSM", "3D Terrain", "Reference", "Residual", "Confidence"] as const;
const layers = ["Texture", "DSM", "Slope", "Hillshade", "Contours", "Confidence", "Residual"] as const;
const cameraModes: { id: CameraMode; label: string }[] = [
  { id: "orbit", label: "Orbit" },
  { id: "fly", label: "Fly" },
  { id: "firstPerson", label: "First person" },
  { id: "topDown", label: "Top down" },
];
const exaggerations = [1, 1.5, 2, 3] as const;
const terminalJobStates = new Set(["waiting_for_calibration", "complete", "failed"]);
const recentProjectsKey = "depthwizard.recent-projects.v1";

const IDLE_TERRAIN_STATE: TerrainRenderState = {
  phase: "idle",
  message: "Terrain renderer idle",
  triangles: 0,
  drawCalls: 0,
};

type ViewName = (typeof views)[number];
type LayerName = (typeof layers)[number];

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
  imagery: { source: string; path: string };
};

type BenchmarkReport = {
  dataset: string;
  protocol: string;
  results: Array<{
    anchor_count: number;
    heldout_pixels: number;
    metrics: { rmse_m: number; mae_m: number; pearson_r: number | null };
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

function projectPreviewLayer(view: ViewName, activeLayer: LayerName, manifest: ProjectManifest | null): ProjectPreviewLayer | null {
  if (!manifest) return null;
  if (view === "Optical") return "optical";
  if (view === "DSM") {
    if (activeLayer === "Slope" && manifest.artifacts.slope) return "slope";
    if (activeLayer === "Hillshade") return "hillshade";
    if (activeLayer === "Contours") return "contours";
    if (manifest.artifacts.dsm) return "dsm";
    if (manifest.artifacts.rdsm) return "rdsm";
    return null;
  }
  if (view === "Reference" && manifest.artifacts.reference) return "reference";
  if (view === "Residual" && manifest.artifacts.residual) return "residual";
  if (view === "Confidence" && manifest.artifacts.confidence) return "confidence";
  return null;
}

function terrainOverlayLayer(activeLayer: LayerName, manifest: ProjectManifest | null): ProjectPreviewLayer | null {
  if (!manifest || activeLayer === "Texture") return null;
  if (activeLayer === "DSM") {
    if (manifest.artifacts.dsm) return "dsm";
    if (manifest.artifacts.rdsm) return "rdsm";
    return null;
  }
  if (activeLayer === "Slope" && manifest.artifacts.slope) return "slope";
  if (activeLayer === "Hillshade") return "hillshade";
  if (activeLayer === "Contours") return "contours";
  if (activeLayer === "Confidence" && manifest.artifacts.confidence) return "confidence";
  if (activeLayer === "Residual" && manifest.artifacts.residual) return "residual";
  return null;
}

function bundleName(report: ProjectExportReport): string {
  return report.bundle_path.split(/[\\/]/).pop() ?? `depthwizard-${report.project_id}.zip`;
}

function readRecentProjects(): string[] {
  try {
    const raw = window.localStorage.getItem(recentProjectsKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string").slice(0, 6) : [];
  } catch {
    return [];
  }
}

function interactionMode(tool: string): RasterInteractionMode {
  if (tool === "Project") return "navigate";
  if (tool === "Measure") return "measure";
  if (tool === "Profiles") return "profile";
  if (tool === "Structures") return "structure";
  return "probe";
}

function layerTitle(layer: ProjectPreviewLayer | null, calibrationReady: boolean): string {
  if (layer === "optical") return "Optical RGB";
  if (layer === "residual") return "Prediction − reference";
  if (layer === "reference") return "Aligned reference DSM";
  if (layer === "slope") return "Surface slope";
  if (layer === "hillshade") return "Derived hillshade";
  if (layer === "contours") return "Derived contour visualization";
  if (layer === "confidence") return "Model-native confidence";
  if (layer === "rdsm") return "Relative DSM";
  if (layer === "dsm") return "Absolute DSM";
  return calibrationReady ? "Absolute DSM" : "Relative DSM";
}

export function WorkstationApp() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const [activeTool, setActiveTool] = useState("Project");
  const [activeView, setActiveView] = useState<ViewName>("Optical");
  const [activeLayer, setActiveLayer] = useState<LayerName>("Texture");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [metadata, setMetadata] = useState<RasterMetadata | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [projectJob, setProjectJob] = useState<ProjectJobState | null>(null);
  const [projectManifest, setProjectManifest] = useState<ProjectManifest | null>(null);
  const [submittingProject, setSubmittingProject] = useState(false);
  const [gcpEvidence, setGcpEvidence] = useState<GroundControlPointFileReport | null>(null);
  const [demoReport, setDemoReport] = useState<AbsoluteDemoReport | null>(null);
  const [validationEvidence, setValidationEvidence] = useState<ValidationEvidence | null>(null);
  const [projectValidation, setProjectValidation] = useState<ReferenceValidationReport | null>(null);
  const [validatingReference, setValidatingReference] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [comparisonUrl, setComparisonUrl] = useState<string | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [legend, setLegend] = useState<ProjectLayerLegend | null>(null);
  const [probe, setProbe] = useState<ProjectProbeResult | null>(null);
  const [lineStart, setLineStart] = useState<NormalizedPoint | null>(null);
  const [lineEnd, setLineEnd] = useState<NormalizedPoint | null>(null);
  const [measurement, setMeasurement] = useState<ProjectProfileResult | null>(null);
  const [profile, setProfile] = useState<ProjectProfileResult | null>(null);
  const [structurePolygon, setStructurePolygon] = useState<NormalizedPoint[]>([]);
  const [structureHeight, setStructureHeight] = useState<ProjectStructureHeightResult | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [rasterViewState, setRasterViewState] = useState<RasterViewState>(DEFAULT_RASTER_VIEW_STATE);
  const [projectMesh, setProjectMesh] = useState<ProjectMeshReport | null>(null);
  const [projectMeshUrl, setProjectMeshUrl] = useState<string | null>(null);
  const [buildingMesh, setBuildingMesh] = useState(false);
  const [meshLod, setMeshLod] = useState(0);
  const [autoLod, setAutoLod] = useState(true);
  const [terrainPerformance, setTerrainPerformance] = useState<TerrainPerformance | null>(null);
  const [terrainRenderState, setTerrainRenderState] = useState<TerrainRenderState>(IDLE_TERRAIN_STATE);
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(1);
  const [terrainOverlayUrl, setTerrainOverlayUrl] = useState<string | null>(null);
  const [terrainOverlayLoading, setTerrainOverlayLoading] = useState(false);
  const [autoFlythrough, setAutoFlythrough] = useState(false);
  const [cameraResetToken, setCameraResetToken] = useState(0);
  const [projectExport, setProjectExport] = useState<ProjectExportReport | null>(null);
  const [exporting, setExporting] = useState(false);
  const [recentProjects, setRecentProjects] = useState<string[]>(readRecentProjects);
  const lodChangedAtRef = useRef(0);
  const lodPressureRef = useRef(0);

  const meshUrl: string | undefined = demoMode ? "/demo/terrain.glb" : projectMeshUrl ?? undefined;

  const rememberProject = (dir: string) => {
    setRecentProjects((current) => {
      const next = [dir, ...current.filter((item) => item !== dir)].slice(0, 6);
      try {
        window.localStorage.setItem(recentProjectsKey, JSON.stringify(next));
      } catch {
        // Local history is convenience-only; project durability remains manifest-backed.
      }
      return next;
    });
  };

  const resetAnalysis = () => {
    setProbe(null);
    setLineStart(null);
    setLineEnd(null);
    setMeasurement(null);
    setProfile(null);
    setStructurePolygon([]);
    setStructureHeight(null);
    setAnalysisBusy(false);
  };

  const clearProjectMesh = () => {
    setProjectMesh(null);
    setMeshLod(0);
    setAutoLod(true);
    lodPressureRef.current = 0;
    setTerrainPerformance(null);
    setTerrainRenderState(IDLE_TERRAIN_STATE);
    setVerticalExaggeration(1);
    setAutoFlythrough(false);
    setProjectExport(null);
    setProjectMeshUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    setTerrainOverlayUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
  };

  const hydrateExistingProject = async (dir: string) => {
    setImportError(null);
    setImporting(true);
    try {
      const manifest = await getProjectManifest(dir);
      const nextMetadata = await inspectRaster(manifest.source_path);
      const [validation, mesh, existingExport] = await Promise.all([
        manifest.artifacts.metrics ? getProjectValidation(dir).catch(() => null) : Promise.resolve(null),
        manifest.artifacts.mesh_manifest ? getProjectMesh(dir).catch(() => null) : Promise.resolve(null),
        getProjectExport(dir).catch(() => null),
      ]);
      setMetadata(nextMetadata);
      setProjectDir(dir);
      setProjectJob(null);
      setProjectManifest(manifest);
      setGcpEvidence(null);
      setProjectValidation(validation);
      setProjectMesh(mesh);
      setProjectExport(existingExport);
      setMeshLod(0);
      setAutoLod(true);
      setTerrainPerformance(null);
      setTerrainRenderState(mesh ? { ...IDLE_TERRAIN_STATE, phase: "loading", message: "Fetching persisted terrain LOD…" } : IDLE_TERRAIN_STATE);
      setVerticalExaggeration(1);
      setAutoFlythrough(false);
      resetAnalysis();
      setRasterViewState(DEFAULT_RASTER_VIEW_STATE);
      if (manifest.artifacts.dsm || manifest.artifacts.rdsm) {
        setActiveView("DSM");
        setActiveLayer("DSM");
      } else {
        setActiveView("Optical");
        setActiveLayer("Texture");
      }
      setActiveTool("Project");
      rememberProject(dir);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to open DepthWizard project");
    } finally {
      setImporting(false);
    }
  };

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
        const preferred = benchmark.results.find((item) => item.anchor_count === 64) ?? benchmark.results.at(-1);
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
        setActiveView("3D Terrain");
      })
      .catch((error: unknown) => {
        if (!cancelled) setImportError(error instanceof Error ? error.message : "Unable to load demo evidence");
      });
    return () => { cancelled = true; };
  }, [demoMode]);

  const projectJobId = projectJob?.job_id;
  useEffect(() => {
    if (!projectJobId || (projectJob && terminalJobStates.has(projectJob.status))) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getProjectJob(projectJobId)
        .then(async (next) => {
          if (cancelled) return;
          if (!terminalJobStates.has(next.status)) {
            setProjectJob(next);
            return;
          }
          window.clearInterval(timer);
          const manifest = await getProjectManifest(next.project_dir);
          if (cancelled) return;
          const [validation, mesh] = await Promise.all([
            manifest.artifacts.metrics ? getProjectValidation(next.project_dir).catch(() => null) : Promise.resolve(null),
            manifest.artifacts.mesh_manifest ? getProjectMesh(next.project_dir).catch(() => null) : Promise.resolve(null),
          ]);
          if (cancelled) return;
          setProjectValidation(validation);
          setProjectMesh(mesh);
          setProjectManifest(manifest);
          setProjectJob(next);
          setProjectExport(null);
          setMeshLod(0);
          setTerrainPerformance(null);
          setTerrainRenderState(mesh ? { ...IDLE_TERRAIN_STATE, phase: "loading", message: "Fetching persisted terrain LOD…" } : IDLE_TERRAIN_STATE);
          resetAnalysis();
          setActiveLayer(manifest.artifacts.dsm || manifest.artifacts.rdsm ? "DSM" : "Texture");
          setActiveView(manifest.artifacts.dsm || manifest.artifacts.rdsm ? "DSM" : "Optical");
          rememberProject(next.project_dir);
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
  const calibrationReady = demoMode ? Boolean(meshUrl) : Boolean(projectManifest?.artifacts.dsm);
  const meshArtifactReady = demoMode ? Boolean(meshUrl) : Boolean(projectMesh);
  const rendererReady = terrainRenderState.phase === "ready";
  const rendererControlsEnabled = terrainControlsEnabled(terrainRenderState.phase);
  const processing = projectJob?.status === "queued" || projectJob?.status === "running";
  const waitingForCalibration = projectJob?.status === "waiting_for_calibration"
    || (!projectJob && projectManifest?.status === "waiting_for_calibration");
  const previewLayer = projectPreviewLayer(activeView, activeLayer, projectManifest);
  const terrainOverlay = terrainOverlayLayer(activeLayer, projectManifest);
  const canCompare = compareToolAvailable(calibrationReady, Boolean(projectValidation));
  const compareActive = activeTool === "Compare" && canCompare;
  const analystInteractive = !demoMode && activeView !== "3D Terrain" && Boolean(projectDir) && geometryReady;
  const projectAnalystInteractive = !demoMode && Boolean(projectDir) && geometryReady;
  const currentLegendLayer = activeView === "3D Terrain" ? terrainOverlay : previewLayer;

  const disabledTools = useMemo(() => {
    const disabled = new Set<string>();
    if (!geometryReady) {
      disabled.add("Measure");
      disabled.add("Profiles");
      disabled.add("Layers");
      disabled.add("Export");
    }
    if (!calibrationReady) {
      disabled.add("Structures");
      disabled.add("Validation");
    }
    if (!canCompare) disabled.add("Compare");
    return disabled;
  }, [calibrationReady, canCompare, geometryReady]);

  const terrainAnalysisPath = useMemo<NormalizedPoint[]>(() => {
    if (activeTool === "Profiles" && profile) return profile.samples.map((sample) => sample.point);
    if (activeTool === "Measure" && measurement) return measurement.samples.map((sample) => sample.point);
    if (lineStart && lineEnd) return [lineStart, lineEnd];
    return [];
  }, [activeTool, lineEnd, lineStart, measurement, profile]);

  useEffect(() => {
    if (demoMode || !projectDir || !projectMesh) {
      setProjectMeshUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      if (!demoMode) setTerrainRenderState(IDLE_TERRAIN_STATE);
      return;
    }
    let cancelled = false;
    setTerrainPerformance(null);
    setTerrainRenderState({ phase: "loading", message: `Fetching authenticated terrain LOD ${meshLod}…`, triangles: 0, drawCalls: 0 });
    setProjectMeshUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    void getProjectMeshUrl(projectDir, meshLod)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        setProjectMeshUrl(url);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : "Unable to fetch project terrain LOD";
        setTerrainRenderState({ phase: "error", message, triangles: 0, drawCalls: 0 });
      });
    return () => { cancelled = true; };
  }, [demoMode, meshLod, projectDir, projectMesh?.build_config_sha256]);

  useEffect(() => {
    if (demoMode || activeView !== "3D Terrain" || !projectDir || !projectMesh || !terrainOverlay) {
      setTerrainOverlayUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setTerrainOverlayLoading(false);
      return;
    }
    let cancelled = false;
    setTerrainOverlayLoading(true);
    setTerrainOverlayUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    void getProjectPreviewUrl(projectDir, terrainOverlay, 1600)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        setTerrainOverlayUrl(url);
      })
      .catch((error: unknown) => {
        if (!cancelled) setImportError(error instanceof Error ? error.message : "Unable to load 3D analytical overlay");
      })
      .finally(() => {
        if (!cancelled) setTerrainOverlayLoading(false);
      });
    return () => { cancelled = true; };
  }, [activeView, demoMode, projectDir, projectManifest?.updated_at_utc, projectMesh?.build_config_sha256, terrainOverlay]);

  useEffect(() => {
    if (!autoLod || activeView !== "3D Terrain" || !projectMesh || !validTerrainTelemetry(terrainRenderState, terrainPerformance)) {
      lodPressureRef.current = 0;
      return;
    }
    const delta = lodPressureDelta(terrainPerformance);
    if (delta === 0) {
      lodPressureRef.current = 0;
      return;
    }
    if (Math.sign(lodPressureRef.current) !== Math.sign(delta)) lodPressureRef.current = 0;
    lodPressureRef.current += delta;
    const now = performance.now();
    if (now - lodChangedAtRef.current < 3500) return;
    const lastLod = Math.max(0, projectMesh.lods.length - 1);
    const next = nextAutoLod(meshLod, lastLod, lodPressureRef.current);
    if (next !== meshLod) {
      lodChangedAtRef.current = now;
      lodPressureRef.current = 0;
      setMeshLod(next);
    }
  }, [activeView, autoLod, meshLod, projectMesh, terrainPerformance, terrainRenderState]);

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
    setPreviewLoading(true);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    void getProjectPreviewUrl(projectDir, previewLayer)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        setPreviewUrl(url);
      })
      .catch((error: unknown) => {
        if (!cancelled) setImportError(error instanceof Error ? error.message : "Unable to render project layer");
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => { cancelled = true; };
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
    setComparisonLoading(true);
    setComparisonUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    void getProjectPreviewUrl(projectDir, "reference")
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        setComparisonUrl(url);
      })
      .catch((error: unknown) => {
        if (!cancelled) setImportError(error instanceof Error ? error.message : "Unable to load comparison reference");
      })
      .finally(() => {
        if (!cancelled) setComparisonLoading(false);
      });
    return () => { cancelled = true; };
  }, [compareActive, projectDir, projectManifest?.updated_at_utc]);

  useEffect(() => {
    if (demoMode || !projectDir || !currentLegendLayer || currentLegendLayer === "optical") {
      setLegend(null);
      return;
    }
    let cancelled = false;
    setLegend(null);
    void getProjectLayerLegend(projectDir, currentLegendLayer)
      .then((next) => {
        if (!cancelled) setLegend(next);
      })
      .catch(() => {
        if (!cancelled) setLegend(null);
      });
    return () => { cancelled = true; };
  }, [currentLegendLayer, demoMode, projectDir, projectManifest?.updated_at_utc]);

  useEffect(() => {
    if (activeTool !== "Structures") return;
    const keyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLElement && target.closest("input, textarea, select, [contenteditable='true']")) return;
      if ((event.key === "Backspace" || event.key === "Delete") && structurePolygon.length > 0) {
        event.preventDefault();
        setStructureHeight(null);
        setStructurePolygon((current) => current.slice(0, -1));
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" && structurePolygon.length > 0) {
        event.preventDefault();
        setStructureHeight(null);
        setStructurePolygon((current) => current.slice(0, -1));
      } else if (event.key === "Escape") {
        event.preventDefault();
        setStructureHeight(null);
        setStructurePolygon([]);
      }
    };
    window.addEventListener("keydown", keyDown);
    return () => window.removeEventListener("keydown", keyDown);
  }, [activeTool, structurePolygon.length]);

  const projectName = useMemo(() => (
    demoMode ? "Joshimath absolute DSM" : metadata?.path.split(/[\\/]/).pop() ?? "Untitled reconstruction"
  ), [demoMode, metadata]);

  const importImagery = async () => {
    setImportError(null);
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Remote-sensing imagery", extensions: ["png", "jpg", "jpeg", "tif", "tiff"] }],
      });
      if (!selected || Array.isArray(selected)) return;
      setImporting(true);
      const nextMetadata = await inspectRaster(selected);
      setMetadata(nextMetadata);
      setProjectDir(null);
      setProjectJob(null);
      setProjectManifest(null);
      setGcpEvidence(null);
      setValidationEvidence(null);
      setProjectValidation(null);
      clearProjectMesh();
      resetAnalysis();
      setRasterViewState(DEFAULT_RASTER_VIEW_STATE);
      setActiveTool("Project");
      setActiveLayer("Texture");
      setActiveView("Optical");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to inspect imagery");
    } finally {
      setImporting(false);
    }
  };

  const openExistingProject = async () => {
    setImportError(null);
    try {
      const selectedDir = await open({ multiple: false, directory: true });
      if (!selectedDir || Array.isArray(selectedDir)) return;
      await hydrateExistingProject(selectedDir);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to choose DepthWizard project directory");
    }
  };

  const reconstruct = async (resume = false) => {
    if (!metadata) return;
    setImportError(null);
    try {
      const selectedDir = resume && projectDir ? projectDir : await open({ multiple: false, directory: true });
      if (!selectedDir || Array.isArray(selectedDir)) return;
      setSubmittingProject(true);
      const next = await submitProject({
        source: metadata.path,
        output_dir: selectedDir,
        requested_output: metadata.crs ? null : "rdsm",
      });
      setProjectDir(selectedDir);
      setProjectManifest(null);
      setGcpEvidence(null);
      setProjectValidation(null);
      clearProjectMesh();
      resetAnalysis();
      setProjectJob(next);
      setActiveView("Optical");
      setActiveLayer("Texture");
      rememberProject(selectedDir);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start reconstruction");
    } finally {
      setSubmittingProject(false);
    }
  };

  const addDemEvidence = async () => {
    if (!metadata || !projectDir) return;
    setImportError(null);
    try {
      const dem = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Metric DEM", extensions: ["tif", "tiff"] }],
      });
      if (!dem || Array.isArray(dem)) return;
      setSubmittingProject(true);
      const next = await submitProject({ source: metadata.path, output_dir: projectDir, dem_path: dem, requested_output: "dsm" });
      setGcpEvidence(null);
      clearProjectMesh();
      resetAnalysis();
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start metric calibration");
    } finally {
      setSubmittingProject(false);
    }
  };

  const addGcpEvidence = async () => {
    if (!metadata || !projectDir) return;
    setImportError(null);
    try {
      const gcpPath = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Ground control points", extensions: ["csv"] }],
      });
      if (!gcpPath || Array.isArray(gcpPath)) return;
      setSubmittingProject(true);
      const report = await inspectGroundControlPoints(gcpPath);
      const next = await submitProject({
        source: metadata.path,
        output_dir: projectDir,
        gcps: report.points,
        gcp_evidence: { source_path: report.source_path, sha256: report.sha256 },
        requested_output: "dsm",
      });
      setGcpEvidence(report);
      clearProjectMesh();
      resetAnalysis();
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start GCP calibration");
    } finally {
      setSubmittingProject(false);
    }
  };

  const addDemGcpEvidence = async () => {
    if (!metadata || !projectDir) return;
    setImportError(null);
    try {
      const dem = await open({ multiple: false, directory: false, filters: [{ name: "Metric DEM", extensions: ["tif", "tiff"] }] });
      if (!dem || Array.isArray(dem)) return;
      const gcpPath = await open({ multiple: false, directory: false, filters: [{ name: "Ground control points", extensions: ["csv"] }] });
      if (!gcpPath || Array.isArray(gcpPath)) return;
      setSubmittingProject(true);
      const report = await inspectGroundControlPoints(gcpPath);
      const next = await submitProject({
        source: metadata.path,
        output_dir: projectDir,
        dem_path: dem,
        gcps: report.points,
        gcp_evidence: { source_path: report.source_path, sha256: report.sha256 },
        requested_output: "dsm",
      });
      setGcpEvidence(report);
      clearProjectMesh();
      resetAnalysis();
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start DEM + GCP calibration");
    } finally {
      setSubmittingProject(false);
    }
  };

  const validateReference = async () => {
    if (!projectDir || !calibrationReady || projectValidation) return;
    setImportError(null);
    try {
      const reference = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Reference DSM", extensions: ["tif", "tiff"] }],
      });
      if (!reference || Array.isArray(reference)) return;
      setValidatingReference(true);
      const report = await validateProjectReference(projectDir, reference);
      const manifest = await getProjectManifest(projectDir);
      setProjectValidation(report);
      setProjectManifest(manifest);
      setProjectExport(null);
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
      setProjectExport(null);
      setMeshLod(0);
      setAutoLod(true);
      lodPressureRef.current = 0;
      setTerrainPerformance(null);
      setTerrainRenderState({ phase: "loading", message: "Fetching persisted terrain LOD…", triangles: 0, drawCalls: 0 });
      setVerticalExaggeration(1);
      setActiveLayer("Texture");
      setActiveView("3D Terrain");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to build project terrain mesh");
    } finally {
      setBuildingMesh(false);
    }
  };

  const exportProject = async () => {
    if (!projectDir || !geometryReady || demoMode) return;
    setImportError(null);
    try {
      setExporting(true);
      const report = await buildProjectExport(projectDir, { includeSource: false, includeMesh: true, includeValidation: true });
      setProjectExport(report);
      const url = await getProjectExportUrl(projectDir);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = bundleName(report);
      anchor.style.display = "none";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to build project export bundle");
    } finally {
      setExporting(false);
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
    if (activeTool === "Structures") {
      if (!calibrationReady) return;
      setStructureHeight(null);
      setStructurePolygon((current) => current.length >= 64 ? current : [...current, point]);
      await analyzePoint(point);
      return;
    }
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

  const measureStructure = async () => {
    if (!projectDir || !calibrationReady || structurePolygon.length < 3) return;
    setImportError(null);
    setAnalysisBusy(true);
    try {
      setStructureHeight(await estimateProjectStructureHeight(projectDir, structurePolygon));
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to estimate structural height");
    } finally {
      setAnalysisBusy(false);
    }
  };

  const viewAvailable = (view: ViewName): boolean => {
    if (view === "3D Terrain") return meshArtifactReady;
    if (demoMode) return false;
    if (view === "Optical") return Boolean(projectManifest);
    if (view === "DSM") return geometryReady;
    if (view === "Reference" || view === "Residual") return Boolean(projectValidation);
    if (view === "Confidence") return Boolean(projectManifest?.artifacts.confidence);
    return false;
  };

  const chooseView = (view: ViewName) => {
    if (!viewAvailable(view)) return;
    setActiveView(view);
    if (view !== "3D Terrain") setAutoFlythrough(false);
    if (view === "Optical") setActiveLayer("Texture");
    if (view === "DSM" && activeLayer === "Texture") setActiveLayer("DSM");
    if (view === "Reference") setActiveLayer("DSM");
    if (view === "Residual") setActiveLayer("Residual");
    if (view === "Confidence") setActiveLayer("Confidence");
  };

  const layerAvailable = (layer: LayerName): boolean => {
    if (layer === "Texture") return Boolean(projectManifest) || meshArtifactReady;
    if (layer === "DSM") return geometryReady;
    if (layer === "Slope") return Boolean(projectManifest?.artifacts.slope);
    if (layer === "Hillshade" || layer === "Contours") return geometryReady;
    if (layer === "Confidence") return Boolean(projectManifest?.artifacts.confidence);
    if (layer === "Residual") return Boolean(projectValidation);
    return false;
  };

  const chooseLayer = (layer: LayerName) => {
    if (!layerAvailable(layer)) return;
    setActiveLayer(layer);
    if (activeView === "3D Terrain") return;
    if (layer === "Texture") setActiveView("Optical");
    if (["DSM", "Slope", "Hillshade", "Contours"].includes(layer)) setActiveView("DSM");
    if (layer === "Confidence") setActiveView("Confidence");
    if (layer === "Residual") setActiveView("Residual");
  };

  const chooseTool = (tool: string) => {
    if (disabledTools.has(tool)) return;
    resetAnalysis();
    setActiveTool(tool);
    if (tool === "Validation" && projectValidation) {
      setActiveView("Residual");
      setActiveLayer("Residual");
    } else if (tool === "Compare" && canCompare) {
      setActiveView("DSM");
      setActiveLayer("DSM");
    } else if (tool === "Structures" || tool === "Measure" || tool === "Profiles") {
      setActiveView("DSM");
      setActiveLayer("DSM");
      setAutoFlythrough(false);
    }
  };

  const analysisHint = activeTool === "Structures"
    ? calibrationReady
      ? structurePolygon.length < 3
        ? `Select footprint vertices · ${structurePolygon.length}/3 minimum · drag numbered vertices to refine`
        : `${structurePolygon.length} vertices selected · drag to refine · measure when complete`
      : "Structural height requires an absolute metric DSM"
    : activeTool === "Measure"
      ? lineStart && !lineEnd ? "Select endpoint B · Space + drag pans" : "Select point A, then B · Space + drag pans"
      : activeTool === "Profiles"
        ? lineStart && !lineEnd ? "Move for live transect preview, then select endpoint B" : "Select transect endpoints A → B · Space + drag pans"
        : activeTool === "Compare"
          ? "Drag to pan · wheel/pinch to zoom · use slider to swipe reference ↔ prediction"
          : activeTool === "Project"
            ? "Drag to pan · wheel/pinch to zoom · double-click zooms · click without dragging probes"
            : "Click raster to inspect synchronized project values · Space + drag pans";

  const recoverableProject = Boolean(
    !projectJob
      && projectManifest
      && !geometryReady
      && ["created", "queued", "running", "failed"].includes(projectManifest.status),
  );

  const validTelemetry = validTerrainTelemetry(terrainRenderState, terrainPerformance) ? terrainPerformance : null;
  const status = recoverableProject
    ? "Persisted project interrupted · resume reconstruction"
    : activeWorkspaceStatus({
      activeView,
      previewLoading: previewLoading || comparisonLoading,
      terrainPhase: terrainRenderState.phase,
      terrainMessage: terrainRenderState.message,
      processing,
      waitingForCalibration,
      calibrationReady,
      geometryReady,
      analysisBusy,
      exporting,
      buildingMesh,
      projectError: projectJob?.error,
      projectExportMiB: projectExport ? projectExport.bundle_bytes / (1024 * 1024) : null,
      validationRmseM: projectValidation?.elevation.rmse_m ?? null,
    });

  const visibleCanvas = Boolean(
    activeView === "3D Terrain"
      ? meshArtifactReady
      : previewUrl || previewLoading || comparisonLoading,
  );

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
          {!demoMode && (
            <button className="dw-btn" onClick={() => void openExistingProject()} disabled={importing || processing || validatingReference || buildingMesh || exporting}>
              Open project
            </button>
          )}
          {!demoMode && recentProjects.length > 0 && (
            <select
              className="dw-recent-select"
              aria-label="Recent DepthWizard projects"
              value=""
              disabled={importing || processing || validatingReference || buildingMesh || exporting}
              onChange={(event) => {
                const dir = event.target.value;
                if (dir) void hydrateExistingProject(dir);
              }}
            >
              <option value="">Recent projects</option>
              {recentProjects.map((dir) => <option value={dir} key={dir}>{dir.split(/[\\/]/).pop() || dir}</option>)}
            </select>
          )}
          <button className="dw-btn" onClick={() => void importImagery()} disabled={demoMode || importing || processing || validatingReference || buildingMesh || exporting}>
            <UploadIcon /> {importing ? "Opening…" : "Import imagery"}
          </button>
          {!demoMode && metadata && !projectDir && (
            <button className="dw-btn dw-btn--primary" onClick={() => void reconstruct()} disabled={submittingProject}>
              {submittingProject ? "Starting…" : "Reconstruct"}
            </button>
          )}
          {!demoMode && recoverableProject && (
            <button className="dw-btn dw-btn--primary" onClick={() => void reconstruct(true)} disabled={submittingProject}>
              {submittingProject ? "Resuming…" : "Resume reconstruction"}
            </button>
          )}
          {!demoMode && waitingForCalibration && (
            <>
              <button className="dw-btn dw-btn--primary" onClick={() => void addDemEvidence()} disabled={submittingProject}>{submittingProject ? "Starting…" : "Add DEM"}</button>
              <button className="dw-btn" onClick={() => void addGcpEvidence()} disabled={submittingProject}>Add GCP CSV</button>
              <button className="dw-btn" onClick={() => void addDemGcpEvidence()} disabled={submittingProject}>DEM + GCP</button>
            </>
          )}
          {!demoMode && geometryReady && (
            <button className="dw-btn" onClick={() => void buildTerrain()} disabled={buildingMesh || processing || meshArtifactReady}>
              {buildingMesh ? "Building 3D…" : meshArtifactReady ? "3D mesh built" : "Build 3D terrain"}
            </button>
          )}
          {!demoMode && calibrationReady && (
            <button
              className="dw-btn"
              onClick={() => void validateReference()}
              disabled={validatingReference || Boolean(projectValidation)}
              title={projectValidation ? "This project already preserves one completed reference validation" : undefined}
            >
              {validatingReference ? "Validating…" : projectValidation ? "Reference validated" : "Validate reference"}
            </button>
          )}
          <button
            className="dw-btn dw-btn--primary"
            onClick={() => void exportProject()}
            disabled={demoMode || !projectDir || !geometryReady || processing || exporting}
            title="Build and download a hash-audited ZIP. Source imagery is excluded by default."
          >
            {exporting ? "Packaging…" : projectExport ? "Export again" : "Export"}
          </button>
        </div>
      </header>

      <ToolRail active={activeTool} onChange={chooseTool} disabledTools={disabledTools} />

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
                data-active={!autoFlythrough && cameraMode === mode.id}
                disabled={!rendererControlsEnabled}
                data-render-disabled={!rendererControlsEnabled}
                onClick={() => {
                  if (!rendererControlsEnabled) return;
                  setAutoFlythrough(false);
                  setCameraMode(mode.id);
                }}
              >
                {mode.label}
              </button>
            ))}
            {activeView === "3D Terrain" && (
              <button
                className="dw-chip"
                data-active={autoFlythrough}
                disabled={!rendererControlsEnabled}
                onClick={() => setAutoFlythrough((current) => !current)}
              >
                Flythrough
              </button>
            )}
            {activeView === "3D Terrain" && (
              <button
                className="dw-chip"
                disabled={!rendererControlsEnabled}
                onClick={() => {
                  setAutoFlythrough(false);
                  setCameraMode("orbit");
                  setCameraResetToken((current) => current + 1);
                }}
              >
                Fit
              </button>
            )}
            {activeView === "3D Terrain" && projectMesh && (
              <select
                className="dw-compact-select"
                aria-label="Terrain level of detail"
                disabled={!rendererControlsEnabled}
                value={autoLod ? "auto" : String(meshLod)}
                onChange={(event) => {
                  if (event.target.value === "auto") {
                    setAutoLod(true);
                    lodPressureRef.current = 0;
                    return;
                  }
                  setAutoLod(false);
                  lodChangedAtRef.current = performance.now();
                  lodPressureRef.current = 0;
                  setMeshLod(Number(event.target.value));
                }}
              >
                <option value="auto">Auto LOD</option>
                {projectMesh.lods.map((lod) => <option value={lod.level} key={lod.level}>LOD {lod.level} · {lod.faces.toLocaleString()} faces</option>)}
              </select>
            )}
            {activeView === "3D Terrain" && !demoMode && (
              <select
                className="dw-compact-select"
                aria-label="Display vertical exaggeration"
                disabled={!rendererControlsEnabled}
                value={verticalExaggeration}
                onChange={(event) => setVerticalExaggeration(Number(event.target.value))}
              >
                {exaggerations.map((value) => <option value={value} key={value}>{value}× Z</option>)}
              </select>
            )}
            {activeView === "3D Terrain" && validTelemetry && (
              <span className="dw-render-metric" title="Measured only after a non-zero rendered terrain frame">
                {validTelemetry.fps.toFixed(0)} fps
              </span>
            )}
            {activeTool === "Structures" && activeView !== "3D Terrain" && calibrationReady && (
              <>
                <button
                  className="dw-chip"
                  disabled={structurePolygon.length === 0 || analysisBusy}
                  onClick={() => {
                    setStructureHeight(null);
                    setStructurePolygon((current) => current.slice(0, -1));
                  }}
                >
                  Undo vertex
                </button>
                <button
                  className="dw-chip"
                  data-active={Boolean(structureHeight)}
                  disabled={structurePolygon.length < 3 || analysisBusy}
                  onClick={() => void measureStructure()}
                >
                  {analysisBusy ? "Measuring…" : "Measure footprint"}
                </button>
              </>
            )}
            {(lineStart || probe || structurePolygon.length > 0 || structureHeight) && (
              <button className="dw-chip" onClick={resetAnalysis}>Clear analysis</button>
            )}
            <span className="dw-toolbar-divider" aria-hidden="true" />
            <select
              className="dw-compact-select"
              aria-label="Scientific display layer"
              value={activeLayer}
              onChange={(event) => chooseLayer(event.target.value as LayerName)}
            >
              {layers.map((layer) => <option value={layer} key={layer} disabled={!layerAvailable(layer)}>{layer}</option>)}
            </select>
          </div>
        </div>

        <div className="dw-canvas">
          {importError && (
            <div className="dw-workspace-alert" role="alert">
              <strong>DepthWizard requires attention</strong>
              <span>{importError}</span>
              <button type="button" aria-label="Dismiss error" onClick={() => setImportError(null)}>×</button>
            </div>
          )}

          {activeView === "3D Terrain" && meshUrl && (
            <TerrainViewport
              meshUrl={meshUrl}
              cameraMode={cameraMode}
              verticalExaggeration={verticalExaggeration}
              cursorPoint={probe?.point}
              analysisPath={terrainAnalysisPath}
              overlayUrl={terrainOverlayUrl}
              autoFlythrough={autoFlythrough && rendererReady}
              resetToken={cameraResetToken}
              onSelectPoint={projectAnalystInteractive && activeTool !== "Structures" ? analyzeRasterPoint : undefined}
              onPerformance={setTerrainPerformance}
              onRenderState={setTerrainRenderState}
            />
          )}

          {activeView === "3D Terrain" && meshArtifactReady && !meshUrl && terrainRenderState.phase !== "error" && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>Preparing 3D terrain</h2>
                <p>{terrainRenderState.message}</p>
              </div>
            </div>
          )}

          {activeView === "3D Terrain" && meshArtifactReady && !meshUrl && terrainRenderState.phase === "error" && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card dw-empty-card--danger">
                <h2>Terrain LOD could not be fetched</h2>
                <p>{terrainRenderState.message}</p>
              </div>
            </div>
          )}

          {activeView !== "3D Terrain" && compareActive && previewUrl && comparisonUrl && (
            <ComparisonViewport
              predictionUrl={previewUrl}
              referenceUrl={comparisonUrl}
              viewState={rasterViewState}
              onViewStateChange={setRasterViewState}
              sourceWidth={metadata?.width}
              groundSampleDistanceM={metadata?.ground_sample_distance_x}
              cursorPoint={probe?.point}
              onSelectPoint={analyzeRasterPoint}
            />
          )}

          {activeView !== "3D Terrain" && previewUrl && !(compareActive && comparisonUrl) && (
            <RasterAnalysisViewport
              src={previewUrl}
              alt={`${previewLayer ?? activeView} scientific raster`}
              interactive={analystInteractive}
              interactionMode={interactionMode(activeTool)}
              viewState={rasterViewState}
              onViewStateChange={setRasterViewState}
              sourceWidth={metadata?.width}
              groundSampleDistanceM={metadata?.ground_sample_distance_x}
              cursorPoint={probe?.point}
              lineStart={activeTool === "Measure" || activeTool === "Profiles" ? lineStart : null}
              lineEnd={activeTool === "Measure" || activeTool === "Profiles" ? lineEnd : null}
              polygonPoints={activeTool === "Structures" ? structurePolygon : []}
              polygonClosed={activeTool === "Structures" && Boolean(structureHeight)}
              onPolygonChange={(points) => {
                setStructureHeight(null);
                setStructurePolygon(points);
              }}
              onSelectPoint={analyzeRasterPoint}
            />
          )}

          {activeView !== "3D Terrain" && (previewLoading || (compareActive && comparisonLoading)) && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>{compareActive ? "Loading synchronized comparison" : "Loading scientific layer"}</h2>
                <p>{compareActive ? "Preparing the prediction and evaluation-only reference on one registered viewport." : "Rendering a bounded local preview from the persisted project raster. Source products remain unchanged."}</p>
              </div>
            </div>
          )}

          {activeView !== "3D Terrain" && analystInteractive && (previewUrl || previewLoading) && (
            <div className="dw-interaction-hint">{analysisHint}</div>
          )}

          <ScientificLegend legend={legend} />

          {visibleCanvas && activeView !== "3D Terrain" && (
            <>
              <div className="dw-canvas-context">
                <strong>{compareActive ? "Prediction ↔ reference comparison" : layerTitle(previewLayer, calibrationReady)}</strong>
                <span>
                  {compareActive
                    ? "evaluation-only reference · synchronized map transform"
                    : previewLayer === "residual"
                      ? `${projectValidation?.valid_pixels.toLocaleString() ?? "—"} valid pixels · metres`
                      : previewLayer === "reference"
                        ? "evaluation-only · aligned to prediction grid"
                        : previewLayer === "confidence"
                          ? "model-native · not probability calibrated"
                          : previewLayer === "hillshade" || previewLayer === "contours"
                            ? "display derivative · numerical surface unchanged"
                            : estimatorModel(projectManifest) ?? "persisted project raster"}
                </span>
              </div>
              <div className="dw-north-indicator" aria-label="North indicator"><strong>N</strong><span>↑</span></div>
            </>
          )}

          {activeView === "3D Terrain" && meshArtifactReady && (
            <div className="dw-canvas-context">
              <strong>{activeLayer === "Texture" ? (calibrationReady ? "Absolute DSM terrain" : "Relative DSM terrain") : `${activeLayer} analytical overlay`}</strong>
              <span>
                {rendererReady
                  ? `${estimatorModel(projectManifest) ?? demoReport?.model ?? "DA3MONO-LARGE"} · rendered LOD ${meshLod} · ${verticalExaggeration}× display Z`
                  : terrainRenderState.message}
              </span>
            </div>
          )}

          {(previewUrl || rendererReady) && (
            <div className="dw-scene-badge">
              <strong>
                {analysisBusy
                  ? activeTool === "Structures" ? "Measuring selected structure" : "Sampling analytical products"
                  : activeTool === "Structures" && structureHeight
                    ? `${structureHeight.structure_height_m.toFixed(2)} m structure height`
                    : activeView === "3D Terrain" && probe?.surface.available
                      ? `${probe.surface.value?.toFixed(2) ?? "—"} ${probe.surface.units ?? ""}`
                      : previewLayer === "residual"
                        ? `RMSE ${projectValidation?.elevation.rmse_m.toFixed(3) ?? "—"} m`
                        : demoMode || calibrationReady ? "Metric elevation" : "Relative elevation"}
              </strong>
              <span>
                {activeView === "3D Terrain" && projectMesh && rendererReady
                  ? `LOD ${meshLod} ${autoLod ? "auto" : "manual"} · ${validTelemetry ? `${validTelemetry.fps.toFixed(0)} fps · ${validTelemetry.triangles.toLocaleString()} triangles · ` : ""}${projectMesh.relief.toFixed(2)} ${projectMesh.vertical_units} relief`
                  : activeTool === "Structures" && structureHeight
                    ? `top ${structureHeight.top_elevation_m.toFixed(2)} m · local ground ${structureHeight.ground_elevation_m.toFixed(2)} m`
                    : activeTool === "Measure" && measurement
                      ? `${measurement.horizontal_distance_m?.toFixed(2) ?? measurement.horizontal_distance_pixels.toFixed(2)} ${measurement.horizontal_distance_m === null ? "px" : "m"} · Δz ${measurement.vertical_delta?.toFixed(2) ?? "—"} ${measurement.vertical_units ?? ""}`
                      : activeTool === "Profiles" && profile
                        ? `${profile.sample_count} samples · ${profile.horizontal_distance_m?.toFixed(2) ?? profile.horizontal_distance_pixels.toFixed(2)} ${profile.horizontal_distance_m === null ? "px" : "m"}`
                        : calibrationReady ? "evidence-calibrated · metres" : geometryReady ? "dimensionless relative surface height · not metric height" : "source imagery"}
              </span>
            </div>
          )}

          {!visibleCanvas && !previewLoading && !importError && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>
                  {buildingMesh
                    ? "Building analytical terrain"
                    : processing
                      ? "Reconstructing scene"
                      : waitingForCalibration
                        ? "Relative geometry complete"
                        : geometryReady && !meshArtifactReady
                          ? "Terrain products ready"
                          : metadata
                            ? "Source accepted"
                            : "Load or open a reconstruction project"}
                </h2>
                <p>
                  {buildingMesh
                    ? "Generating persistent hashed GLB LODs from the already-produced surface and source RGB. Validation reference data is not used."
                    : processing
                      ? "DepthWizard is running the local packaged scientific pipeline. The source stays on this machine."
                      : waitingForCalibration
                        ? "This georeferenced project is intentionally paused before any metric-height claim. Add DEM evidence, sparse GCP evidence, or combine DEM + GCP."
                        : geometryReady && !meshArtifactReady
                          ? "The scientific surface is ready. Build the terrain LOD pyramid when you want the interactive 3D workspace."
                          : metadata
                            ? metadata.crs
                              ? "Georeferenced input detected. Reconstruct once, then DepthWizard requires DEM/GCP evidence before claiming absolute height."
                              : "No usable CRS detected. DepthWizard will preserve a relative DSM and will not claim metric height."
                            : "Import single-view RGB remote-sensing imagery or open an existing durable DepthWizard project."}
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="dw-workspace-status">
          <span>{demoMode ? (rendererReady ? "Interactive demo terrain ready" : terrainRenderState.message) : status}</span>
          <span>
            {metadata?.crs ?? "Projection —"} · GSD {metadata?.ground_sample_distance_x?.toFixed(3) ?? "—"} m · {calibrationReady ? "DSM metres" : geometryReady ? "rDSM" : "Elevation —"}
          </span>
        </footer>
      </section>

      <Inspector
        metadata={metadata}
        geometryReady={geometryReady}
        meshArtifactReady={meshArtifactReady}
        rendererReady={rendererReady}
        terrainRenderState={terrainRenderState}
        activeView={activeView}
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
        structureHeight={structureHeight}
        structureVertexCount={structurePolygon.length}
        gcpEvidence={gcpEvidence}
        analysisBusy={analysisBusy}
        projectExport={projectExport}
        meshLod={meshLod}
        autoLod={autoLod}
        terrainPerformance={validTelemetry}
      />
    </main>
  );
}
