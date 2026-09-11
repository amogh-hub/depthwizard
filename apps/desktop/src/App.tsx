import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  buildProjectExport,
  buildProjectMesh,
  cancelProjectJob,
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
  DEFAULT_RASTER_VIEW_STATE,
  type RasterViewState,
} from "./workspace/rasterViewport";
import {
  TerrainViewport,
  type CameraMode,
  type TerrainOverlayState,
  type TerrainPerformance,
  type TerrainRenderState,
  type TerrainScreenshot,
} from "./workspace/TerrainViewport";
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
const terminalJobStates = new Set(["waiting_for_calibration", "complete", "failed", "cancelled"]);
const recentProjectStorageKey = "depthwizard.recentProjects.v1";
const emptyTerrainState: TerrainRenderState = {
  phase: "idle",
  message: "Terrain renderer idle",
  triangles: 0,
  drawCalls: 0,
};
const emptyTerrainOverlayState: TerrainOverlayState = {
  phase: "idle",
  message: "Source texture active",
};

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

function projectPreviewLayer(
  view: (typeof views)[number],
  activeLayer: (typeof layers)[number],
  manifest: ProjectManifest | null,
): ProjectPreviewLayer | null {
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

function terrainOverlayLayer(
  activeLayer: (typeof layers)[number],
  manifest: ProjectManifest | null,
): ProjectPreviewLayer | null {
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

function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

function safeFileStem(value: string): string {
  return value.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "") || "project";
}

function readRecentProjects(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(recentProjectStorageKey) ?? "[]") as unknown;
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string").slice(0, 6) : [];
  } catch {
    return [];
  }
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.matches("input, textarea, select, [contenteditable='true']")
    || Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function reopenedJobState(manifest: ProjectManifest, projectDir: string): ProjectJobState | null {
  if (!["waiting_for_calibration", "complete", "failed", "cancelled"].includes(manifest.status)) return null;
  return {
    job_id: manifest.job_id ?? `reopened-${manifest.project_id}`,
    project_dir: projectDir,
    status: manifest.status,
    manifest_path: `${projectDir.replace(/[\\/]$/, "")}/project-manifest.json`,
    submitted_at_utc: manifest.created_at_utc,
    updated_at_utc: manifest.updated_at_utc,
    error: manifest.status === "failed" ? manifest.errors.at(-1)?.message ?? "Project requires recovery" : null,
    failure_kind: manifest.status === "failed" ? "processing_error" : null,
    cancellation_requested: manifest.status === "cancelled",
  };
}

export function App() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const [activeTool, setActiveTool] = useState("Project");
  const [activeView, setActiveView] = useState<(typeof views)[number]>("Optical");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [activeLayer, setActiveLayer] = useState<(typeof layers)[number]>("Texture");
  const [metadata, setMetadata] = useState<RasterMetadata | null>(null);
  const [sourceAvailable, setSourceAvailable] = useState(true);
  const [importError, setImportError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [openingProject, setOpeningProject] = useState(false);
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
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewRetryGeneration, setPreviewRetryGeneration] = useState(0);
  const [renderedPreviewLayer, setRenderedPreviewLayer] = useState<ProjectPreviewLayer | null>(null);
  const [layerLegend, setLayerLegend] = useState<ProjectLayerLegend | null>(null);
  const [comparisonUrl, setComparisonUrl] = useState<string | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [comparisonRetryGeneration, setComparisonRetryGeneration] = useState(0);
  const [probe, setProbe] = useState<ProjectProbeResult | null>(null);
  const [lineStart, setLineStart] = useState<NormalizedPoint | null>(null);
  const [lineEnd, setLineEnd] = useState<NormalizedPoint | null>(null);
  const [measurement, setMeasurement] = useState<ProjectProfileResult | null>(null);
  const [profile, setProfile] = useState<ProjectProfileResult | null>(null);
  const [structurePolygon, setStructurePolygon] = useState<NormalizedPoint[]>([]);
  const [structureHeight, setStructureHeight] = useState<ProjectStructureHeightResult | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [projectMesh, setProjectMesh] = useState<ProjectMeshReport | null>(null);
  const [projectMeshUrl, setProjectMeshUrl] = useState<string | null>(null);
  const [meshUrlLoading, setMeshUrlLoading] = useState(false);
  const [buildingMesh, setBuildingMesh] = useState(false);
  const [meshLod, setMeshLod] = useState(0);
  const [autoLod, setAutoLod] = useState(true);
  const [terrainPerformance, setTerrainPerformance] = useState<TerrainPerformance | null>(null);
  const [terrainRenderState, setTerrainRenderState] = useState<TerrainRenderState>(emptyTerrainState);
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(1);
  const [terrainOverlayUrl, setTerrainOverlayUrl] = useState<string | null>(null);
  const [terrainOverlayLoading, setTerrainOverlayLoading] = useState(false);
  const [terrainOverlayError, setTerrainOverlayError] = useState<string | null>(null);
  const [terrainOverlayRenderState, setTerrainOverlayRenderState] = useState<TerrainOverlayState>(emptyTerrainOverlayState);
  const [terrainOverlayRetryGeneration, setTerrainOverlayRetryGeneration] = useState(0);
  const [terrainLegend, setTerrainLegend] = useState<ProjectLayerLegend | null>(null);
  const [autoFlythrough, setAutoFlythrough] = useState(false);
  const [cameraResetToken, setCameraResetToken] = useState(0);
  const [terrainScreenshotRequest, setTerrainScreenshotRequest] = useState(0);
  const [capturingTerrainScreenshot, setCapturingTerrainScreenshot] = useState(false);
  const [relativeHorizontalScaleInput, setRelativeHorizontalScaleInput] = useState("");
  const [projectExport, setProjectExport] = useState<ProjectExportReport | null>(null);
  const [exporting, setExporting] = useState(false);
  const [rasterViewState, setRasterViewState] = useState<RasterViewState>(DEFAULT_RASTER_VIEW_STATE);
  const [recentProjects, setRecentProjects] = useState<string[]>(readRecentProjects);
  const lodPressureRef = useRef(0);
  const meshUrl: string | undefined = demoMode ? "/demo/terrain.glb" : projectMeshUrl ?? undefined;

  useEffect(() => {
    setRelativeHorizontalScaleInput("");
  }, [metadata?.path]);

  const persistRecentProjects = (paths: string[]) => {
    setRecentProjects(paths);
    try {
      localStorage.setItem(recentProjectStorageKey, JSON.stringify(paths));
    } catch {
      // Recent paths are convenience only; project evidence remains on disk.
    }
  };

  const rememberProject = (path: string) => {
    persistRecentProjects([path, ...recentProjects.filter((item) => item !== path)].slice(0, 6));
  };

  const forgetProject = (path: string) => {
    persistRecentProjects(recentProjects.filter((item) => item !== path));
  };

  const revokePreview = () => {
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    setRenderedPreviewLayer(null);
    setLayerLegend(null);
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
    setTerrainRenderState(emptyTerrainState);
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
    setTerrainOverlayLoading(false);
    setTerrainOverlayError(null);
    setTerrainOverlayRenderState(emptyTerrainOverlayState);
    setTerrainLegend(null);
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
        setSourceAvailable(true);
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
          valid_data_fraction: 1,
          vertical_crs: null,
          vertical_datum: null,
          elevation_reference: "unknown",
          quality: {
            status: "not_assessed",
            flags: [],
            saturation_fraction: null,
            deep_shadow_candidate_fraction: null,
            bright_low_chroma_candidate_fraction: null,
            texture_gradient_score: null,
            off_nadir_degrees: null,
            assessment_limitations: ["Legacy static demo metadata has no source-quality report"],
          },
        });
        setActiveView("3D Terrain");
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
      })
      .catch((error: unknown) => {
        if (!cancelled) setImportError(error instanceof Error ? error.message : "Unable to load demo evidence");
      });
    return () => { cancelled = true; };
  }, [demoMode]);

  const projectJobId = projectJob?.job_id;
  const projectJobStatus = projectJob?.status;
  useEffect(() => {
    if (!projectJobId || !projectJobStatus || terminalJobStates.has(projectJobStatus)) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getProjectJob(projectJobId)
        .then(async (next) => {
          if (cancelled) return;
          if (terminalJobStates.has(next.status)) {
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
            setPreviewError(null);
            resetAnalysis();
            setActiveLayer(manifest.artifacts.dsm ? "DSM" : "Texture");
            setActiveView(manifest.artifacts.dsm || manifest.artifacts.rdsm ? "DSM" : "Optical");
            setRasterViewState(DEFAULT_RASTER_VIEW_STATE);
            rememberProject(next.project_dir);
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
  }, [projectJobId, projectJobStatus]);

  const geometryReady = demoMode
    ? Boolean(meshUrl)
    : Boolean(projectManifest?.artifacts.rdsm || projectManifest?.artifacts.dsm);
  const calibrationReady = demoMode ? Boolean(meshUrl) : Boolean(projectManifest?.artifacts.dsm);
  const meshArtifactReady = demoMode ? Boolean(meshUrl) : Boolean(projectMesh);
  const rendererReady = terrainRenderState.phase === "ready";
  const rendererControlsReady = terrainControlsEnabled(terrainRenderState.phase);
  const processing = projectJob?.status === "queued" || projectJob?.status === "running";
  const waitingForCalibration = projectJob?.status === "waiting_for_calibration"
    || (!projectJob && projectManifest?.status === "waiting_for_calibration");
  const needsRecovery = Boolean(
    !demoMode
      && projectDir
      && projectManifest
      && ["created", "queued", "running", "failed", "cancelled"].includes(projectManifest.status),
  );
  const previewLayer = projectPreviewLayer(activeView, activeLayer, projectManifest);
  const terrainOverlay = terrainOverlayLayer(activeLayer, projectManifest);
  const compareAvailable = compareToolAvailable(calibrationReady, Boolean(projectValidation));
  const compareActive = activeTool === "Compare" && compareAvailable;
  const rasterSurfaceReady = activeView !== "3D Terrain" && Boolean(previewUrl) && !previewLoading && !previewError;
  const analystInteractive = !demoMode && Boolean(projectDir) && geometryReady && rasterSurfaceReady;
  const projectAnalystInteractive = !demoMode && Boolean(projectDir) && geometryReady;
  const terrainToolInteractive = projectAnalystInteractive && (
    activeTool === "Project"
    || activeTool === "Measure"
    || activeTool === "Profiles"
    || (activeTool === "Structures" && calibrationReady)
  );
  const terrainAnalysisPath = useMemo<NormalizedPoint[]>(() => {
    if (activeTool === "Structures" && structurePolygon.length >= 2) {
      return structurePolygon.length >= 3
        ? [...structurePolygon, structurePolygon[0]]
        : structurePolygon;
    }
    if (activeTool === "Profiles" && profile) return profile.samples.map((sample) => sample.point);
    if (activeTool === "Measure" && measurement) return measurement.samples.map((sample) => sample.point);
    if (lineStart && lineEnd) return [lineStart, lineEnd];
    return [];
  }, [activeTool, lineEnd, lineStart, measurement, profile, structurePolygon]);

  useEffect(() => {
    if (demoMode || !projectDir || !projectMesh) {
      setProjectMeshUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setMeshUrlLoading(false);
      if (!demoMode) setTerrainRenderState(emptyTerrainState);
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    setMeshUrlLoading(true);
    setTerrainPerformance(null);
    setTerrainRenderState({ phase: "loading", message: `Fetching terrain LOD ${meshLod}…`, triangles: 0, drawCalls: 0 });
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
        createdUrl = url;
        setProjectMeshUrl(url);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : "Unable to load project terrain LOD";
          setTerrainRenderState({ phase: "error", message, triangles: 0, drawCalls: 0 });
          setImportError(message);
        }
      })
      .finally(() => {
        if (!cancelled) setMeshUrlLoading(false);
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [demoMode, meshLod, projectDir, projectMesh?.build_config_sha256]);

  useEffect(() => {
    setTerrainOverlayRenderState(emptyTerrainOverlayState);
    if (demoMode || activeView !== "3D Terrain" || !projectDir || !projectMesh || !terrainOverlay) {
      setTerrainOverlayUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setTerrainOverlayLoading(false);
      setTerrainOverlayError(null);
      setTerrainLegend(null);
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    setTerrainOverlayLoading(true);
    setTerrainOverlayError(null);
    setTerrainLegend(null);
    Promise.all([
      getProjectPreviewUrl(projectDir, terrainOverlay, 1600),
      getProjectLayerLegend(projectDir, terrainOverlay),
    ])
      .then(([url, legend]) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setTerrainOverlayUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return url;
        });
        setTerrainLegend(legend);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setTerrainOverlayUrl((current) => {
            if (current) URL.revokeObjectURL(current);
            return null;
          });
          setTerrainLegend(null);
          setTerrainOverlayError(error instanceof Error ? error.message : "Unable to load 3D analytical overlay");
        }
      })
      .finally(() => {
        if (!cancelled) setTerrainOverlayLoading(false);
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [activeView, demoMode, projectDir, projectManifest?.updated_at_utc, projectMesh?.build_config_sha256, terrainOverlay, terrainOverlayRetryGeneration]);

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
    const previous = lodPressureRef.current;
    lodPressureRef.current = Math.sign(previous) === delta ? previous + delta : delta;
    const next = nextAutoLod(meshLod, Math.max(0, projectMesh.lods.length - 1), lodPressureRef.current);
    if (next !== meshLod) {
      lodPressureRef.current = 0;
      setMeshLod(next);
    }
  }, [activeView, autoLod, meshLod, projectMesh, terrainPerformance, terrainRenderState]);

  useEffect(() => {
    if (demoMode || activeView === "3D Terrain" || !projectDir || !previewLayer) {
      revokePreview();
      setPreviewLoading(false);
      setPreviewError(null);
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    revokePreview();
    setPreviewLoading(true);
    setPreviewError(null);
    Promise.all([
      getProjectPreviewUrl(projectDir, previewLayer),
      getProjectLayerLegend(projectDir, previewLayer),
    ])
      .then(([url, legend]) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setPreviewUrl(url);
        setRenderedPreviewLayer(previewLayer);
        setLayerLegend(legend);
        setPreviewError(null);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          revokePreview();
          setPreviewError(error instanceof Error ? error.message : "Unable to render project layer");
        }
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [activeView, demoMode, previewLayer, projectDir, projectManifest?.updated_at_utc, previewRetryGeneration]);

  useEffect(() => {
    if (!compareActive || !projectDir) {
      setComparisonUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setComparisonLoading(false);
      setComparisonError(null);
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    setComparisonLoading(true);
    setComparisonError(null);
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
        createdUrl = url;
        setComparisonUrl(url);
      })
      .catch((error: unknown) => {
        if (!cancelled) setComparisonError(error instanceof Error ? error.message : "Unable to load comparison reference");
      })
      .finally(() => {
        if (!cancelled) setComparisonLoading(false);
      });
    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [compareActive, comparisonRetryGeneration, projectDir, projectManifest?.updated_at_utc]);

  useEffect(() => {
    resetAnalysis();
    setPreviewError(null);
    if (activeTool === "Validation" && projectValidation) {
      setActiveView("Residual");
      setActiveLayer("Residual");
      setAutoFlythrough(false);
    } else if (activeTool === "Compare" && projectValidation) {
      setActiveView("DSM");
      setActiveLayer("DSM");
      setAutoFlythrough(false);
    } else if (activeTool === "Structures" && calibrationReady && activeView !== "3D Terrain") {
      setActiveView("DSM");
      setActiveLayer("DSM");
      setAutoFlythrough(false);
    } else if ((activeTool === "Measure" || activeTool === "Profiles") && geometryReady && activeView !== "3D Terrain") {
      setActiveView("DSM");
      setActiveLayer("DSM");
      setAutoFlythrough(false);
    }
  }, [activeTool]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;
      if ((activeTool === "Measure" || activeTool === "Profiles") && event.key === "Escape") {
        event.preventDefault();
        resetAnalysis();
        return;
      }
      if (activeTool !== "Structures") return;
      if ((event.key === "Backspace" || event.key === "Delete") && structurePolygon.length > 0) {
        event.preventDefault();
        setStructureHeight(null);
        setStructurePolygon((current) => current.slice(0, -1));
      }
      if (event.key === "Escape") {
        event.preventDefault();
        setStructureHeight(null);
        setStructurePolygon([]);
      }
    };
    window.addEventListener("keydown", keyDown);
    return () => window.removeEventListener("keydown", keyDown);
  }, [activeTool, structurePolygon.length]);

  const projectName = useMemo(() => (
    demoMode ? "Joshimath absolute DSM" : metadata?.path ? fileName(metadata.path) : "Untitled reconstruction"
  ), [demoMode, metadata]);
  const analystHorizontalScaleMPerPixel = (() => {
    const value = Number(relativeHorizontalScaleInput);
    return Number.isFinite(value) && value > 0 ? value : undefined;
  })();

  const loadExistingProject = async (selectedDir: string) => {
    setImportError(null);
    setPreviewError(null);
    setOpeningProject(true);
    try {
      const manifest = await getProjectManifest(selectedDir);
      let nextMetadata: RasterMetadata;
      let nextSourceAvailable = true;
      try {
        nextMetadata = await inspectRaster(manifest.source_path);
      } catch (sourceError) {
        const persistedSurface = manifest.artifacts.dsm?.path ?? manifest.artifacts.rdsm?.path;
        if (!persistedSurface) throw sourceError;
        const surfaceMetadata = await inspectRaster(persistedSurface);
        nextMetadata = {
          ...surfaceMetadata,
          path: manifest.source_path,
          count: 0,
          dtype: "source unavailable",
        };
        nextSourceAvailable = false;
      }
      const [validation, mesh, exported] = await Promise.all([
        manifest.artifacts.metrics ? getProjectValidation(selectedDir).catch(() => null) : Promise.resolve(null),
        manifest.artifacts.mesh_manifest ? getProjectMesh(selectedDir).catch(() => null) : Promise.resolve(null),
        getProjectExport(selectedDir).catch(() => null),
      ]);
      revokePreview();
      clearProjectMesh();
      resetAnalysis();
      setMetadata(nextMetadata);
      setSourceAvailable(nextSourceAvailable);
      setProjectDir(selectedDir);
      setProjectManifest(manifest);
      setProjectJob(reopenedJobState(manifest, selectedDir));
      setProjectValidation(validation);
      setProjectMesh(mesh);
      setProjectExport(exported);
      setGcpEvidence(null);
      setValidationEvidence(null);
      setRasterViewState(DEFAULT_RASTER_VIEW_STATE);
      setActiveTool("Project");
      setActiveLayer(manifest.artifacts.dsm || manifest.artifacts.rdsm ? "DSM" : "Texture");
      setActiveView(manifest.artifacts.dsm || manifest.artifacts.rdsm ? "DSM" : "Optical");
      rememberProject(selectedDir);
    } catch (error) {
      if (recentProjects.includes(selectedDir)) forgetProject(selectedDir);
      setImportError(error instanceof Error ? error.message : "Unable to open existing DepthWizard project");
    } finally {
      setOpeningProject(false);
    }
  };

  const openProject = async () => {
    try {
      const selectedDir = await open({ multiple: false, directory: true, title: "Open DepthWizard project" });
      if (!selectedDir || Array.isArray(selectedDir)) return;
      await loadExistingProject(selectedDir);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to open project folder picker");
    }
  };

  const importImagery = async () => {
    setImportError(null);
    setPreviewError(null);
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        title: "Import remote-sensing imagery",
        filters: [{ name: "Remote-sensing imagery", extensions: ["png", "jpg", "jpeg", "tif", "tiff"] }],
      });
      if (!selected || Array.isArray(selected)) return;
      setImporting(true);
      const nextMetadata = await inspectRaster(selected);
      revokePreview();
      clearProjectMesh();
      resetAnalysis();
      setMetadata(nextMetadata);
      setSourceAvailable(true);
      setProjectDir(null);
      setProjectJob(null);
      setProjectManifest(null);
      setGcpEvidence(null);
      setValidationEvidence(null);
      setProjectValidation(null);
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

  const reconstruct = async () => {
    if (!metadata) return;
    setImportError(null);
    try {
      const selectedDir = await open({ multiple: false, directory: true, title: "Choose DepthWizard project folder" });
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
      rememberProject(selectedDir);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to start reconstruction");
    } finally {
      setSubmittingProject(false);
    }
  };

  const recoverProject = async () => {
    if (!metadata || !projectDir) return;
    setImportError(null);
    try {
      setSubmittingProject(true);
      const next = await submitProject({
        source: metadata.path,
        output_dir: projectDir,
        requested_output: metadata.crs ? null : "rdsm",
      });
      setProjectJob(next);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to recover project processing");
    } finally {
      setSubmittingProject(false);
    }
  };

  const cancelProcessing = async () => {
    if (!projectJob || !processing || projectJob.cancellation_requested) return;
    setImportError(null);
    try {
      setProjectJob(await cancelProjectJob(projectJob.job_id));
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Unable to cancel processing");
    }
  };

  const addDemEvidence = async () => {
    if (!metadata || !projectDir) return;
    setImportError(null);
    try {
      const dem = await open({
        multiple: false,
        directory: false,
        title: "Add metric DEM evidence",
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
        title: "Add sparse GCP evidence",
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
      const dem = await open({
        multiple: false,
        directory: false,
        title: "Add metric DEM evidence",
        filters: [{ name: "Metric DEM", extensions: ["tif", "tiff"] }],
      });
      if (!dem || Array.isArray(dem)) return;
      const gcpPath = await open({
        multiple: false,
        directory: false,
        title: "Add sparse GCP evidence",
        filters: [{ name: "Ground control points", extensions: ["csv"] }],
      });
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
        title: "Load independent reference DSM",
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
    if (!projectDir || !geometryReady || demoMode || !sourceAvailable) return;
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
      setTerrainRenderState({ phase: "loading", message: "Fetching terrain LOD 0…", triangles: 0, drawCalls: 0 });
      setTerrainOverlayRenderState(emptyTerrainOverlayState);
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
      setActiveTool("Export");
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

  const terrainScreenshotReady = (capture: TerrainScreenshot) => {
    const buildGitSha = window.__DEPTHWIZARD_RUNTIME__?.buildGitSha ?? "development";
    const url = URL.createObjectURL(capture.blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `depthwizard-${safeFileStem(projectName)}-${buildGitSha}-terrain.png`;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
    setCapturingTerrainScreenshot(false);
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
        sampleProjectProfile(
          projectDir,
          lineStart,
          point,
          activeTool === "Profiles" ? 160 : 2,
          metadata?.crs ? undefined : analystHorizontalScaleMPerPixel,
        ),
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

  const viewAvailable = (view: (typeof views)[number]): boolean => {
    if (view === "3D Terrain") return meshArtifactReady;
    if (demoMode) return false;
    if (view === "Optical") return Boolean(metadata) && sourceAvailable;
    if (view === "DSM") return geometryReady;
    if (view === "Reference" || view === "Residual") return Boolean(projectValidation);
    if (view === "Confidence") return Boolean(projectManifest?.artifacts.confidence);
    return false;
  };

  const chooseView = (view: (typeof views)[number]) => {
    if (!viewAvailable(view)) return;
    setPreviewError(null);
    setComparisonError(null);
    if (view !== activeView && view !== "3D Terrain") revokePreview();
    setActiveView(view);
    if (view !== "3D Terrain") {
      setAutoFlythrough(false);
      setTerrainOverlayRenderState(emptyTerrainOverlayState);
    }
    if (view === "Optical") setActiveLayer("Texture");
    if (view === "DSM") setActiveLayer("DSM");
    if (view === "Residual") setActiveLayer("Residual");
    if (view === "Confidence") setActiveLayer("Confidence");
  };

  const layerAvailable = (layer: (typeof layers)[number]): boolean => {
    if (layer === "Texture") return meshArtifactReady || Boolean(projectManifest && sourceAvailable);
    if (layer === "DSM") return geometryReady;
    if (layer === "Slope") return Boolean(projectManifest?.artifacts.slope);
    if (layer === "Hillshade" || layer === "Contours") return geometryReady;
    if (layer === "Confidence") return Boolean(projectManifest?.artifacts.confidence);
    if (layer === "Residual") return Boolean(projectValidation);
    return false;
  };

  const chooseLayer = (layer: (typeof layers)[number]) => {
    if (!layerAvailable(layer)) return;
    setPreviewError(null);
    setTerrainOverlayError(null);
    setTerrainOverlayRenderState(emptyTerrainOverlayState);
    if (activeView !== "3D Terrain") revokePreview();
    setActiveLayer(layer);
    if (activeView === "3D Terrain" && meshArtifactReady) return;
    if (layer === "Texture") setActiveView("Optical");
    if (["DSM", "Slope", "Hillshade", "Contours"].includes(layer)) setActiveView("DSM");
    if (layer === "Confidence") setActiveView("Confidence");
    if (layer === "Residual") setActiveView("Residual");
  };

  const disabledTools = useMemo(() => {
    const result = new Set<string>();
    if (!geometryReady) {
      for (const tool of ["Measure", "Profiles", "Export"]) result.add(tool);
    }
    if (!calibrationReady) {
      result.add("Structures");
      result.add("Validation");
    }
    if (previewError && activeView !== "3D Terrain") {
      result.add("Measure");
      result.add("Profiles");
      result.add("Structures");
    }
    if (!compareAvailable) result.add("Compare");
    return result;
  }, [activeView, calibrationReady, compareAvailable, geometryReady, previewError]);

  const interactionMode: RasterInteractionMode = activeTool === "Measure"
    ? "measure"
    : activeTool === "Profiles"
      ? "profile"
      : activeTool === "Structures"
        ? "structure"
        : "navigate";

  const analysisHint = activeTool === "Structures"
    ? calibrationReady
      ? structurePolygon.length < 3
        ? `Select footprint vertices · ${structurePolygon.length}/3 minimum · ${activeView === "3D Terrain" ? "use Orbit or Top down to place points" : "drag a numbered vertex to refine"}`
        : `${structurePolygon.length} vertices selected · Backspace/Undo removes last · measure when complete`
      : "Structural height requires an absolute metric DSM"
    : activeTool === "Measure"
      ? lineStart && !lineEnd ? "Select endpoint B · Space+drag pans in 2D" : "Select point A, then point B · 3D selection works in Orbit/Top down"
      : activeTool === "Profiles"
        ? lineStart && !lineEnd ? "Move to preview transect in 2D or select endpoint B in 3D" : "Select transect endpoints A → B"
        : activeTool === "Compare"
          ? "Drag to pan · wheel/pinch to zoom · slider swipes reference ↔ prediction"
          : activeView === "3D Terrain"
            ? cameraMode === "orbit" ? "Drag to orbit · Shift/right-drag to pan · wheel to dolly" : cameraMode === "topDown" ? "Drag to pan · wheel to zoom" : "Use the active navigation mode to explore the terrain"
            : "Drag to pan · wheel/pinch to zoom · click to inspect synchronized values";

  const terrainOverlayRenderError = terrainOverlay && terrainOverlayRenderState.phase === "error"
    ? terrainOverlayRenderState.message
    : null;
  const terrainOverlayFailure = terrainOverlayError ?? terrainOverlayRenderError;
  const terrainOverlayPending = Boolean(
    terrainOverlay
      && !terrainOverlayFailure
      && (terrainOverlayLoading || terrainOverlayRenderState.phase !== "ready"),
  );

  const workspaceFailure = activeView === "3D Terrain"
    ? terrainOverlayFailure
      ? `Analytical overlay unavailable · ${terrainOverlayFailure}`
      : null
    : compareActive && comparisonError
      ? `Comparison reference unavailable · ${comparisonError}`
      : previewError
        ? `${activeView} layer unavailable · ${previewError}`
        : null;

  const normalStatus = workspaceFailure
    ?? (structureHeight
      ? `Structure height ${structureHeight.structure_height_m.toFixed(3)} m`
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
        }));

  const displayedLegend = activeView === "3D Terrain"
    ? terrainOverlay && terrainOverlayRenderState.phase === "ready" && !terrainOverlayFailure ? terrainLegend : null
    : previewError ? null : layerLegend;
  const showCanvasContext = activeView === "3D Terrain" ? Boolean(meshUrl) : Boolean(previewUrl);

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
            <>
              <button className="dw-btn" onClick={() => void openProject()} disabled={openingProject || processing || exporting}>
                {openingProject ? "Opening…" : "Open project"}
              </button>
              {recentProjects.length > 0 && (
                <details className="dw-recent-projects">
                  <summary className="dw-btn">Recent</summary>
                  <div className="dw-recent-menu">
                    {recentProjects.map((path) => (
                      <button key={path} type="button" onClick={() => void loadExistingProject(path)} title={path}>
                        <span>{fileName(path)}</span>
                        <small>{path}</small>
                      </button>
                    ))}
                  </div>
                </details>
              )}
            </>
          )}
          <button className="dw-btn" onClick={() => void importImagery()} disabled={importing || processing || validatingReference || buildingMesh || exporting}>
            <UploadIcon /> {importing ? "Inspecting…" : "Import imagery"}
          </button>
          {!demoMode && metadata && !projectDir && (
            <button className="dw-btn dw-btn--primary" onClick={() => void reconstruct()} disabled={submittingProject}>
              {submittingProject ? "Starting…" : "Reconstruct"}
            </button>
          )}
          {!demoMode && processing && projectJob && (
            <button
              className="dw-btn"
              onClick={() => void cancelProcessing()}
              disabled={projectJob.cancellation_requested}
              title="Cancel at the next safe tile or processing-stage boundary"
            >
              {projectJob.cancellation_requested ? "Cancelling…" : "Cancel processing"}
            </button>
          )}
          {needsRecovery && (
            <button className="dw-btn dw-btn--primary" onClick={() => void recoverProject()} disabled={submittingProject || !sourceAvailable} title={!sourceAvailable ? "Original source imagery is required to resume processing" : undefined}>
              {submittingProject ? "Recovering…" : "Recover project"}
            </button>
          )}
          {!demoMode && waitingForCalibration && (
            <>
              <button className="dw-btn dw-btn--primary" onClick={() => void addDemEvidence()} disabled={submittingProject || !sourceAvailable}>
                {submittingProject ? "Starting…" : "Add DEM"}
              </button>
              <button className="dw-btn" onClick={() => void addGcpEvidence()} disabled={submittingProject || !sourceAvailable}>Add GCP CSV</button>
              <button className="dw-btn" onClick={() => void addDemGcpEvidence()} disabled={submittingProject || !sourceAvailable}>DEM + GCP</button>
            </>
          )}
          {!demoMode && geometryReady && (
            <button
              className="dw-btn"
              onClick={() => void buildTerrain()}
              disabled={buildingMesh || processing || meshArtifactReady || !sourceAvailable}
              title={!sourceAvailable && !meshArtifactReady ? "Original source RGB is required to build a new textured terrain mesh" : undefined}
            >
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

      <ToolRail active={activeTool} onChange={setActiveTool} disabledTools={disabledTools} />

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
                  title={view === "Optical" && !sourceAvailable ? "Original source imagery is unavailable; persisted elevation products remain usable" : available ? undefined : "Enabled only when its real project artifact is available"}
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
                disabled={!rendererControlsReady}
                onClick={() => {
                  if (!rendererControlsReady) return;
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
                disabled={!rendererControlsReady}
                onClick={() => setAutoFlythrough((current) => !current)}
                title="Deterministic display-only camera flythrough; project data is unchanged"
              >Flythrough</button>
            )}
            {activeView === "3D Terrain" && (
              <button
                className="dw-chip"
                disabled={!rendererControlsReady}
                onClick={() => {
                  setAutoFlythrough(false);
                  setCameraMode("orbit");
                  setCameraResetToken((current) => current + 1);
                }}
              >Fit</button>
            )}
            {activeView === "3D Terrain" && (
              <button
                className="dw-chip"
                disabled={!rendererControlsReady || capturingTerrainScreenshot}
                onClick={() => {
                  setCapturingTerrainScreenshot(true);
                  setTerrainScreenshotRequest((current) => current + 1);
                }}
                title="Export the rendered terrain with source-build and display-state provenance"
              >{capturingTerrainScreenshot ? "Capturing…" : "Screenshot"}</button>
            )}
            {activeView === "3D Terrain" && projectMesh && (
              <select
                className="dw-compact-select"
                aria-label="Terrain level of detail"
                value={autoLod ? "auto" : String(meshLod)}
                onChange={(event) => {
                  const value = event.target.value;
                  lodPressureRef.current = 0;
                  if (value === "auto") {
                    setAutoLod(true);
                    return;
                  }
                  setAutoLod(false);
                  setMeshLod(Number(value));
                }}
              >
                <option value="auto">LOD · Auto</option>
                {projectMesh.lods.map((lod) => (
                  <option key={lod.level} value={lod.level}>LOD {lod.level} · {lod.faces.toLocaleString()} faces</option>
                ))}
              </select>
            )}
            {activeView === "3D Terrain" && meshArtifactReady && !demoMode && (
              <select
                className="dw-compact-select"
                aria-label="Vertical exaggeration"
                value={verticalExaggeration}
                disabled={!rendererControlsReady}
                onChange={(event) => setVerticalExaggeration(Number(event.target.value))}
                title="Display-only vertical exaggeration; source elevation values are unchanged"
              >
                {exaggerations.map((value) => <option key={value} value={value}>{value}× Z</option>)}
              </select>
            )}
            {activeView === "3D Terrain" && rendererReady && validTerrainTelemetry(terrainRenderState, terrainPerformance) && (
              <span className="dw-render-metric" title="Measured WebGL renderer frame rate">
                {terrainPerformance.fps.toFixed(0)} fps
              </span>
            )}

            {(activeTool === "Measure" || activeTool === "Profiles") && geometryReady && !metadata?.crs && (
              <input
                className="dw-compact-select dw-horizontal-scale-input"
                type="number"
                min="0.000001"
                max="1000000"
                step="any"
                inputMode="decimal"
                aria-label="Optional analyst horizontal scale in metres per pixel"
                placeholder="m/px optional"
                value={relativeHorizontalScaleInput}
                onChange={(event) => {
                  setRelativeHorizontalScaleInput(event.target.value);
                  setMeasurement(null);
                  setProfile(null);
                  setLineStart(null);
                  setLineEnd(null);
                }}
                title="Optional analyst-declared horizontal scale. Vertical rDSM values remain relative."
              />
            )}

            {activeTool === "Structures" && calibrationReady && (
              <>
                <button className="dw-chip" disabled={structurePolygon.length === 0 || analysisBusy} onClick={() => {
                  setStructureHeight(null);
                  setStructurePolygon((current) => current.slice(0, -1));
                }}>Undo vertex</button>
                <button className="dw-chip" data-active={Boolean(structureHeight)} disabled={structurePolygon.length < 3 || analysisBusy || (activeView !== "3D Terrain" && Boolean(previewError))} onClick={() => void measureStructure()}>
                  {analysisBusy ? "Measuring…" : "Measure footprint"}
                </button>
              </>
            )}
            {(lineStart || probe || structurePolygon.length > 0 || structureHeight) && (
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
              >{layer}</button>
            ))}
          </div>
        </div>

        <div className="dw-canvas">
          {activeView === "3D Terrain" && meshUrl && (
            <TerrainViewport
              meshUrl={meshUrl}
              cameraMode={cameraMode}
              verticalExaggeration={verticalExaggeration}
              groundSampleDistanceM={metadata?.ground_sample_distance_x}
              cursorPoint={probe?.point}
              analysisPath={terrainAnalysisPath}
              overlayUrl={terrainOverlayUrl}
              autoFlythrough={autoFlythrough}
              resetToken={cameraResetToken}
              screenshotRequest={terrainScreenshotRequest}
              screenshotCaption={`${projectName} · ${calibrationReady ? "Metric DSM" : "Relative rDSM"} · ${activeLayer} · Z ${verticalExaggeration}×\nBuild ${window.__DEPTHWIZARD_RUNTIME__?.buildGitSha ?? "development-unversioned"}`}
              onSelectPoint={terrainToolInteractive ? analyzeRasterPoint : undefined}
              onPerformance={setTerrainPerformance}
              onRenderState={(state) => {
                setTerrainRenderState(state);
                if (state.phase !== "ready") setTerrainPerformance(null);
              }}
              onOverlayState={setTerrainOverlayRenderState}
              onScreenshot={terrainScreenshotReady}
              onScreenshotError={(message) => {
                setCapturingTerrainScreenshot(false);
                setImportError(message);
              }}
            />
          )}

          {activeView === "3D Terrain" && meshArtifactReady && !meshUrl && (
            <div className="dw-layer-loading-shade">
              <div><span className="dw-spinner" />{meshUrlLoading ? `Fetching terrain LOD ${meshLod}…` : terrainRenderState.message}</div>
            </div>
          )}

          {activeView === "3D Terrain" && terrainOverlay && terrainOverlayError && (
            <div className="dw-terrain-overlay-state dw-terrain-overlay-state--error" role="alert">
              <strong>Analytical overlay unavailable</strong>
              <span>{terrainOverlayError}</span>
              <button type="button" className="dw-overlay-retry" onClick={() => setTerrainOverlayRetryGeneration((value) => value + 1)}>Retry overlay</button>
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

          {activeView !== "3D Terrain" && previewUrl && !compareActive && (
            <RasterAnalysisViewport
              src={previewUrl}
              alt={`${renderedPreviewLayer ?? activeView} scientific raster`}
              interactive={analystInteractive}
              interactionMode={interactionMode}
              viewState={rasterViewState}
              onViewStateChange={setRasterViewState}
              sourceWidth={metadata?.width}
              groundSampleDistanceM={metadata?.ground_sample_distance_x}
              cursorPoint={probe?.point}
              lineStart={activeTool === "Measure" || activeTool === "Profiles" ? lineStart : null}
              lineEnd={activeTool === "Measure" || activeTool === "Profiles" ? lineEnd : null}
              polygonPoints={activeTool === "Structures" ? structurePolygon : []}
              polygonClosed={activeTool === "Structures" && Boolean(structureHeight)}
              onPolygonChange={activeTool === "Structures" ? (points) => {
                setStructureHeight(null);
                setStructurePolygon(points);
              } : undefined}
              onSelectPoint={analyzeRasterPoint}
            />
          )}

          {activeView !== "3D Terrain" && (previewLoading || comparisonLoading) && (
            <div className="dw-layer-loading-shade">
              <div><span className="dw-spinner" />Loading {previewLayer ?? activeView} scientific layer…</div>
            </div>
          )}

          {activeView !== "3D Terrain" && compareActive && comparisonError && !comparisonLoading && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card dw-empty-card--error">
                <h2>Comparison reference unavailable</h2>
                <p>{comparisonError}</p>
                <button className="dw-btn dw-btn--primary" type="button" onClick={() => setComparisonRetryGeneration((value) => value + 1)}>Retry comparison</button>
              </div>
            </div>
          )}

          {showCanvasContext && (
            <>
              <div className="dw-canvas-context">
                <strong>
                  {compareActive
                    ? "Prediction ↔ reference comparison"
                    : activeView === "3D Terrain"
                      ? activeLayer === "Texture"
                        ? projectMesh?.surface_product === "dsm" || demoMode ? "Absolute DSM terrain" : "Relative DSM terrain"
                        : `${activeLayer} analytical overlay`
                      : renderedPreviewLayer === "residual"
                        ? "Prediction − reference"
                        : renderedPreviewLayer === "reference"
                          ? "Aligned reference DSM"
                          : renderedPreviewLayer === "slope"
                            ? "Surface slope"
                            : renderedPreviewLayer === "hillshade"
                              ? "Derived hillshade"
                              : renderedPreviewLayer === "contours"
                                ? "Derived contour visualization"
                                : renderedPreviewLayer === "confidence"
                                  ? "Model-native confidence"
                                  : renderedPreviewLayer === "optical"
                                    ? "Optical RGB · source imagery"
                                    : calibrationReady ? "Absolute DSM" : "Relative DSM"}
                </strong>
                <span>
                  {compareActive
                    ? comparisonError
                      ? "reference preview failed · comparison disabled until recovery"
                      : "evaluation-only reference · registered viewport · synchronized cursor"
                    : activeView === "3D Terrain"
                      ? rendererReady
                        ? activeLayer === "Texture"
                          ? `${estimatorModel(projectManifest) ?? "DA3MONO-LARGE"} prior · rendered LOD ${meshLod} · ${verticalExaggeration}× display Z`
                          : terrainOverlayFailure
                            ? "analytical overlay failed · source texture restored"
                            : terrainOverlayPending
                              ? "loading and frame-validating analytical overlay · terrain geometry unchanged"
                              : terrainOverlayRenderState.phase === "ready"
                                ? `analytical overlay frame-validated · terrain geometry unchanged · LOD ${meshLod}`
                                : "source texture active · analytical overlay not yet validated"
                        : terrainRenderState.message
                      : renderedPreviewLayer === "residual"
                        ? `${projectValidation?.valid_pixels.toLocaleString() ?? "—"} valid pixels · metres`
                        : renderedPreviewLayer === "reference"
                          ? "evaluation-only · aligned to prediction grid"
                          : renderedPreviewLayer === "confidence"
                            ? "model-native · not probability calibrated"
                            : renderedPreviewLayer === "hillshade" || renderedPreviewLayer === "contours"
                              ? "display derivative · numerical surface unchanged"
                              : renderedPreviewLayer === "optical"
                                ? "original optical pixels · no elevation encoded in the RGB layer"
                                : calibrationReady
                                  ? `${estimatorModel(projectManifest) ?? "DA3MONO-LARGE"} prior + evidence calibration`
                                  : estimatorModel(projectManifest) ?? "persisted project raster"}
                </span>
              </div>
              {(activeView !== "3D Terrain" || cameraMode === "topDown") && (
                <div className="dw-north-indicator" aria-label="North indicator"><strong>N</strong><span>↑</span></div>
              )}
              {activeView === "3D Terrain" && rendererReady && (
                <div className="dw-scene-badge">
                  <strong>{probe?.surface.available ? `${probe.surface.value?.toFixed(2) ?? "—"} ${probe.surface.units ?? ""}` : calibrationReady ? "Metric elevation" : "Relative elevation"}</strong>
                  <span>
                    {projectMesh
                      ? `LOD ${meshLod} ${autoLod ? "auto" : "manual"} · ${validTerrainTelemetry(terrainRenderState, terrainPerformance) ? `${terrainPerformance.fps.toFixed(0)} fps · ${terrainPerformance.triangles.toLocaleString()} triangles · ` : ""}${projectMesh.relief.toFixed(2)} ${projectMesh.vertical_units} relief`
                      : "Renderer ready"}
                  </span>
                </div>
              )}
              {activeView !== "3D Terrain" && (
                <div className="dw-scene-badge">
                  <strong>
                    {analysisBusy
                      ? activeTool === "Structures" ? "Measuring selected structure" : "Sampling analytical products"
                      : activeTool === "Structures" && structureHeight
                        ? `${structureHeight.structure_height_m.toFixed(2)} m structure height`
                        : renderedPreviewLayer === "residual"
                          ? `RMSE ${projectValidation?.elevation.rmse_m.toFixed(3) ?? "—"} m`
                          : renderedPreviewLayer === "optical"
                            ? calibrationReady ? "Metric DSM available" : "Source imagery"
                            : calibrationReady ? "Metric elevation" : "Relative elevation"}
                  </strong>
                  <span>
                    {activeTool === "Structures" && structureHeight
                      ? `roof ${structureHeight.top_elevation_m.toFixed(2)} m · fitted local ground ${structureHeight.ground_elevation_m.toFixed(2)} m`
                      : renderedPreviewLayer === "residual"
                        ? `MAE ${projectValidation?.elevation.mae_m.toFixed(3) ?? "—"} m · P95 ${projectValidation?.elevation.p95_abs_error_m.toFixed(3) ?? "—"} m`
                        : activeTool === "Measure" && measurement
                          ? `${measurement.horizontal_distance_m?.toFixed(2) ?? measurement.horizontal_distance_pixels.toFixed(2)} ${measurement.horizontal_distance_m === null ? "px" : measurement.horizontal_distance_source === "analyst_scale" ? "m analyst scale" : "m ground"} · signed Δz ${measurement.vertical_delta?.toFixed(2) ?? "—"} ${measurement.vertical_units ?? ""}`
                          : activeTool === "Profiles" && profile
                            ? `${profile.sample_count} subpixel samples · ${profile.horizontal_distance_m?.toFixed(2) ?? profile.horizontal_distance_pixels.toFixed(2)} ${profile.horizontal_distance_m === null ? "px" : profile.horizontal_distance_source === "analyst_scale" ? "m analyst scale" : "m ground"}`
                            : renderedPreviewLayer === "optical"
                              ? calibrationReady ? "evidence-calibrated DSM available · RGB remains source imagery" : "source imagery · no elevation claim"
                              : calibrationReady ? "evidence-calibrated · metres" : geometryReady ? "dimensionless relative surface height" : "source imagery"}
                  </span>
                </div>
              )}
            </>
          )}

          <ScientificLegend legend={displayedLegend} />

          {(activeTool !== "Project" || activeView === "3D Terrain") && showCanvasContext && (
            <div className="dw-interaction-hint">{analysisHint}</div>
          )}

          {activeView !== "3D Terrain" && !previewUrl && !previewLoading && previewError && !compareActive && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card dw-empty-card--error">
                <h2>Scientific layer unavailable</h2>
                <p>{previewError}</p>
                <button className="dw-btn dw-btn--primary" type="button" onClick={() => setPreviewRetryGeneration((value) => value + 1)}>Retry layer</button>
              </div>
            </div>
          )}

          {activeView !== "3D Terrain" && !previewUrl && !previewLoading && !previewError && !compareActive && (
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
                            : "Load or open a reconstruction project"}
                </h2>
                <p>
                  {importError
                    ? importError
                    : waitingForCalibration
                      ? "This georeferenced project is intentionally paused before any metric-height claim. Add DEM evidence, sparse GCP evidence, or combine DEM + GCP."
                      : calibrationReady
                        ? "DepthWizard completed evidence-calibrated metric elevation. Navigate the registered layers, build 3D terrain, or load a separate reference DSM."
                        : geometryReady
                          ? "DepthWizard completed a truthful dimensionless relative surface model. No metric elevation has been invented."
                          : metadata
                            ? metadata.crs
                              ? "Georeferenced input detected. Reconstruct once, then DepthWizard will require DEM/GCP evidence before claiming absolute height."
                              : "No usable CRS detected. DepthWizard will preserve this as relative elevation and will not claim metric height."
                            : "Import a single-view RGB remote-sensing image or open a durable DepthWizard project. Core processing remains local."}
                </p>
              </div>
            </div>
          )}

          {activeView === "3D Terrain" && !meshArtifactReady && (
            <div className="dw-empty-canvas">
              <div className="dw-empty-card">
                <h2>{buildingMesh ? "Building analytical terrain" : "Terrain products ready for 3D"}</h2>
                <p>{buildingMesh ? "Generating persistent hashed GLB LODs from the already-produced surface and source RGB." : "Build the persistent terrain LOD pyramid to enable Orbit, Fly, First Person, Top Down and synchronized 3D analysis."}</p>
              </div>
            </div>
          )}
        </div>

        <footer className="dw-workspace-status">
          <span>{demoMode ? terrainRenderState.message : normalStatus}</span>
          <span>
            {metadata?.crs ?? "Projection —"} · Ground GSD {metadata?.ground_sample_distance_x?.toFixed(3) ?? "—"} m · {calibrationReady ? "DSM metres" : geometryReady ? "rDSM" : "Elevation —"}
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
        terrainPerformance={terrainPerformance}
      />
    </main>
  );
}
