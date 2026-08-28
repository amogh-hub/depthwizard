import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { FlyControls } from "three/examples/jsm/controls/FlyControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { NormalizedPoint } from "../api";

export type CameraMode = "orbit" | "fly" | "firstPerson" | "topDown";

export type TerrainPerformance = {
  fps: number;
  triangles: number;
  drawCalls: number;
};

export type TerrainRenderPhase = "idle" | "loading" | "ready" | "error";

export type TerrainRenderState = {
  phase: TerrainRenderPhase;
  message: string;
  triangles: number;
  drawCalls: number;
};

type TerrainOverlayState = {
  phase: "idle" | "loading" | "ready" | "error";
  message: string;
};

type TerrainViewportProps = {
  meshUrl?: string;
  cameraMode: CameraMode;
  verticalExaggeration?: number;
  groundSampleDistanceM?: number | null;
  cursorPoint?: NormalizedPoint | null;
  analysisPath?: NormalizedPoint[];
  overlayUrl?: string | null;
  autoFlythrough?: boolean;
  resetToken?: number;
  onSelectPoint?: (point: NormalizedPoint) => void;
  onPerformance?: (metrics: TerrainPerformance) => void;
  onRenderState?: (state: TerrainRenderState) => void;
};

const EMPTY_RENDER_STATE: TerrainRenderState = {
  phase: "idle",
  message: "Terrain renderer idle",
  triangles: 0,
  drawCalls: 0,
};

const EMPTY_OVERLAY_STATE: TerrainOverlayState = {
  phase: "idle",
  message: "Source texture active",
};

function navigationHelp(cameraMode: CameraMode, autoFlythrough: boolean): string {
  if (autoFlythrough) return "Flythrough active · deterministic camera tour · click Flythrough again to stop";
  if (cameraMode === "orbit") return "Orbit · drag to rotate · Shift/right-drag to pan · wheel to dolly";
  if (cameraMode === "topDown") return "Top down · drag to pan · wheel to zoom · Fit restores the scene";
  if (cameraMode === "fly") return "Fly · click terrain · WASD move · R/F rise/fall · drag to look · Shift accelerates";
  return "First person · WASD move · R/F rise/fall · drag to look · terrain clearance enforced · Shift accelerates · choose Orbit to exit";
}

function materialArray(material: THREE.Material | THREE.Material[]): THREE.Material[] {
  return Array.isArray(material) ? material : [material];
}

export function TerrainViewport({
  meshUrl,
  cameraMode,
  verticalExaggeration = 1,
  groundSampleDistanceM,
  cursorPoint,
  analysisPath = [],
  overlayUrl,
  autoFlythrough = false,
  resetToken = 0,
  onSelectPoint,
  onPerformance,
  onRenderState,
}: TerrainViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const modeRef = useRef(cameraMode);
  const exaggerationRef = useRef(verticalExaggeration);
  const cursorRef = useRef<NormalizedPoint | null | undefined>(cursorPoint);
  const analysisPathRef = useRef<NormalizedPoint[]>(analysisPath);
  const overlayRef = useRef<string | null | undefined>(overlayUrl);
  const autoFlythroughRef = useRef(autoFlythrough);
  const resetRef = useRef(resetToken);
  const selectRef = useRef(onSelectPoint);
  const performanceRef = useRef(onPerformance);
  const renderStateRef = useRef(onRenderState);
  const [retryGeneration, setRetryGeneration] = useState(0);
  const [overlayRetryGeneration, setOverlayRetryGeneration] = useState(0);
  const [renderState, setRenderState] = useState<TerrainRenderState>(EMPTY_RENDER_STATE);
  const [overlayState, setOverlayState] = useState<TerrainOverlayState>(EMPTY_OVERLAY_STATE);

  modeRef.current = cameraMode;
  exaggerationRef.current = verticalExaggeration;
  cursorRef.current = cursorPoint;
  analysisPathRef.current = analysisPath;
  overlayRef.current = overlayUrl;
  autoFlythroughRef.current = autoFlythrough;
  resetRef.current = resetToken;
  selectRef.current = onSelectPoint;
  performanceRef.current = onPerformance;
  renderStateRef.current = onRenderState;

  const publishState = (state: TerrainRenderState) => {
    setRenderState(state);
    renderStateRef.current?.(state);
  };

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !meshUrl) {
      publishState(EMPTY_RENDER_STATE);
      setOverlayState(EMPTY_OVERLAY_STATE);
      return;
    }

    let disposed = false;
    let fatal = false;
    const fail = (message: string) => {
      if (disposed || fatal) return;
      fatal = true;
      publishState({ phase: "error", message, triangles: 0, drawCalls: 0 });
    };

    publishState({
      phase: "loading",
      message: "Loading persistent terrain LOD…",
      triangles: 0,
      drawCalls: 0,
    });
    setOverlayState(EMPTY_OVERLAY_STATE);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xebeff3);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1_000_000);
    camera.position.set(0, 250, 350);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.domElement.tabIndex = 0;
      renderer.domElement.setAttribute("aria-label", "Interactive 3D terrain canvas");
      host.appendChild(renderer.domElement);
    } catch (error) {
      fail(`WebGL renderer initialization failed: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }

    const contextLost = (event: Event) => {
      event.preventDefault();
      fail("WebGL context was lost. Retry the renderer to recreate GPU resources.");
    };
    renderer.domElement.addEventListener("webglcontextlost", contextLost);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x728094, 2.15));
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(300, 700, 240);
    scene.add(sun);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.enablePan = true;
    orbit.screenSpacePanning = false;

    const freeCamera = new FlyControls(camera, renderer.domElement);
    freeCamera.movementSpeed = 80;
    freeCamera.rollSpeed = 0.30;
    freeCamera.dragToLook = true;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x1f5fae, emissiveIntensity: 1.8 }),
    );
    marker.visible = false;
    marker.renderOrder = 10;
    scene.add(marker);

    const pathLine = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x1f5fae, transparent: true, opacity: 0.96 }),
    );
    pathLine.visible = false;
    pathLine.renderOrder = 11;
    scene.add(pathLine);

    const startMarker = new THREE.Mesh(
      new THREE.SphereGeometry(1, 12, 10),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    const endMarker = new THREE.Mesh(
      new THREE.SphereGeometry(1, 12, 10),
      new THREE.MeshBasicMaterial({ color: 0x1f5fae }),
    );
    startMarker.visible = false;
    endMarker.visible = false;
    startMarker.renderOrder = 12;
    endMarker.renderOrder = 12;
    scene.add(startMarker, endMarker);

    const loader = new GLTFLoader();
    const textureLoader = new THREE.TextureLoader();
    let loaded: THREE.Object3D | undefined;
    let sceneCenter = new THREE.Vector3();
    let sceneSize = new THREE.Vector3(100, 100, 100);
    let elevationCenter = 0;
    let appliedExaggeration = 1;
    let terrainMeshes: THREE.Mesh[] = [];
    const originalMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
    let overlayTexture: THREE.Texture | null = null;
    let overlayMaterials: THREE.Material[] = [];
    let appliedOverlayUrl: string | null = null;
    let overlayLoadGeneration = 0;
    let pendingOverlayGeneration: number | null = null;
    let pendingOverlayFrames = 0;
    let pendingOverlayStartedAt = 0;
    let modelLoadedAt = 0;
    let rendererReady = false;
    let shiftBoost = false;
    let flyBaseSpeed = 80;
    let firstPersonBaseSpeed = 35;
    let canvasFocused = false;

    const refreshBounds = () => {
      if (!loaded) return;
      const bounds = new THREE.Box3().setFromObject(loaded);
      sceneCenter = bounds.getCenter(new THREE.Vector3());
      sceneSize = bounds.getSize(new THREE.Vector3());
    };

    const footprint = () => Math.max(sceneSize.x, sceneSize.z, 1);

    const firstPersonClearance = () => {
      const gsdClearance = groundSampleDistanceM && Number.isFinite(groundSampleDistanceM)
        ? groundSampleDistanceM * 2.5
        : 0;
      return THREE.MathUtils.clamp(
        Math.max(8, gsdClearance, footprint() * 0.0015),
        8,
        150,
      );
    };

    const updateNavigationSpeed = () => {
      const base = modeRef.current === "firstPerson" ? firstPersonBaseSpeed : flyBaseSpeed;
      freeCamera.movementSpeed = base * (shiftBoost ? 4 : 1);
    };

    const fitView = () => {
      if (!loaded) return;
      refreshBounds();
      const width = footprint();
      const distance = Math.max(width, sceneSize.y * 2) * 0.88;
      const viewTarget = sceneCenter.clone();
      viewTarget.y -= Math.max(sceneSize.y, width * 0.08) * 0.16;
      orbit.target.copy(viewTarget);
      camera.up.set(0, 1, 0);
      camera.position.set(
        sceneCenter.x + distance * 0.50,
        sceneCenter.y + distance * 0.52,
        sceneCenter.z + distance * 0.80,
      );
      camera.near = Math.max(distance / 10000, 0.01);
      camera.far = Math.max(distance * 24, 1000);
      camera.updateProjectionMatrix();
      camera.lookAt(viewTarget);
      orbit.update();
    };

    const surfaceAtWorld = (x: number, z: number): THREE.Vector3 | null => {
      if (!loaded || terrainMeshes.length === 0) return null;
      const bounds = new THREE.Box3().setFromObject(loaded);
      if (x < bounds.min.x || x > bounds.max.x || z < bounds.min.z || z > bounds.max.z) return null;
      const origin = new THREE.Vector3(
        x,
        bounds.max.y + Math.max(sceneSize.y, firstPersonClearance()) + 50,
        z,
      );
      raycaster.set(origin, new THREE.Vector3(0, -1, 0));
      const hit = raycaster.intersectObjects(terrainMeshes, false)[0];
      return hit?.point.clone() ?? null;
    };

    const surfacePoint = (point: NormalizedPoint): THREE.Vector3 | null => {
      if (!loaded || terrainMeshes.length === 0) return null;
      const bounds = new THREE.Box3().setFromObject(loaded);
      const size = bounds.getSize(new THREE.Vector3());
      const x = bounds.min.x + point.x * size.x;
      const z = bounds.max.z - point.y * size.z;
      return surfaceAtWorld(x, z);
    };

    const constrainFirstPerson = () => {
      if (!loaded) return;
      const bounds = new THREE.Box3().setFromObject(loaded);
      const inset = Math.max(footprint() * 0.0005, 0.01);
      camera.position.x = THREE.MathUtils.clamp(camera.position.x, bounds.min.x + inset, bounds.max.x - inset);
      camera.position.z = THREE.MathUtils.clamp(camera.position.z, bounds.min.z + inset, bounds.max.z - inset);
      const ground = surfaceAtWorld(camera.position.x, camera.position.z);
      if (ground) camera.position.y = Math.max(camera.position.y, ground.y + firstPersonClearance());
    };

    const enterFirstPerson = () => {
      if (!loaded) return;
      const eyeSurface = surfacePoint({ x: 0.5, y: 0.60 });
      const lookSurface = surfacePoint({ x: 0.5, y: 0.48 });
      if (!eyeSurface) {
        fitView();
        return;
      }
      const clearance = firstPersonClearance();
      camera.up.set(0, 1, 0);
      camera.position.copy(eyeSurface).add(new THREE.Vector3(0, clearance, 0));
      const lookTarget = (lookSurface ?? sceneCenter).clone().add(new THREE.Vector3(0, clearance * 0.25, 0));
      camera.lookAt(lookTarget);
      camera.near = THREE.MathUtils.clamp(clearance / 80, 0.1, 2.0);
      camera.updateProjectionMatrix();
      constrainFirstPerson();
    };

    const positionMarker = (point: NormalizedPoint | null | undefined) => {
      if (!point) {
        marker.visible = false;
        return;
      }
      const hit = surfacePoint(point);
      if (!hit) {
        marker.visible = false;
        return;
      }
      const offset = Math.max(footprint() * 0.0025, 0.5);
      marker.position.copy(hit);
      marker.position.y += offset;
      const radius = Math.max(footprint() * 0.004, 0.7);
      marker.scale.setScalar(radius);
      marker.visible = true;
    };

    const refreshAnalysisPath = () => {
      const points = analysisPathRef.current;
      if (!loaded || points.length < 2) {
        pathLine.visible = false;
        startMarker.visible = false;
        endMarker.visible = false;
        return;
      }
      const offset = Math.max(footprint() * 0.0012, 0.25);
      const surfacePoints = points
        .map((point) => surfacePoint(point))
        .filter((point): point is THREE.Vector3 => point !== null)
        .map((point) => point.add(new THREE.Vector3(0, offset, 0)));
      if (surfacePoints.length < 2) {
        pathLine.visible = false;
        startMarker.visible = false;
        endMarker.visible = false;
        return;
      }
      pathLine.geometry.dispose();
      pathLine.geometry = new THREE.BufferGeometry().setFromPoints(surfacePoints);
      pathLine.visible = true;
      const endpointRadius = Math.max(footprint() * 0.0032, 0.55);
      startMarker.position.copy(surfacePoints[0]);
      endMarker.position.copy(surfacePoints[surfacePoints.length - 1]);
      startMarker.scale.setScalar(endpointRadius);
      endMarker.scale.setScalar(endpointRadius);
      startMarker.visible = true;
      endMarker.visible = true;
    };

    const applyExaggeration = () => {
      if (!loaded) return false;
      const exaggeration = Math.max(exaggerationRef.current, 0.1);
      if (Math.abs(exaggeration - appliedExaggeration) < 1e-6) return false;
      loaded.scale.y = exaggeration;
      loaded.position.y = elevationCenter * (1 - exaggeration);
      appliedExaggeration = exaggeration;
      loaded.updateMatrixWorld(true);
      refreshBounds();
      positionMarker(cursorRef.current);
      refreshAnalysisPath();
      if (modeRef.current === "firstPerson") constrainFirstPerson();
      return true;
    };

    const disposeOverlayMaterials = () => {
      for (const material of overlayMaterials) material.dispose();
      overlayMaterials = [];
      overlayTexture?.dispose();
      overlayTexture = null;
    };

    const restoreOriginalMaterials = () => {
      for (const mesh of terrainMeshes) {
        const material = originalMaterials.get(mesh);
        if (material) mesh.material = material;
      }
      disposeOverlayMaterials();
      pendingOverlayGeneration = null;
      pendingOverlayFrames = 0;
    };

    const overlayMaterialFrom = (source: THREE.Material, texture: THREE.Texture): THREE.MeshBasicMaterial => {
      const material = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        transparent: source.transparent,
        opacity: source.opacity,
        depthTest: source.depthTest,
        depthWrite: source.depthWrite,
        alphaTest: source.alphaTest,
        toneMapped: false,
      });
      material.name = `DepthWizard analytical overlay · ${source.name || source.type}`;
      overlayMaterials.push(material);
      return material;
    };

    const applyOverlay = (nextUrl: string | null | undefined) => {
      const normalizedUrl = nextUrl ?? null;
      if (normalizedUrl === appliedOverlayUrl && pendingOverlayGeneration === null) return;
      appliedOverlayUrl = normalizedUrl;
      overlayLoadGeneration += 1;
      const generation = overlayLoadGeneration;
      restoreOriginalMaterials();
      if (!normalizedUrl || terrainMeshes.length === 0) {
        if (!disposed) setOverlayState(EMPTY_OVERLAY_STATE);
        return;
      }
      if (!disposed) setOverlayState({ phase: "loading", message: "Loading analytical terrain overlay…" });
      textureLoader.load(
        normalizedUrl,
        (texture) => {
          if (generation !== overlayLoadGeneration || overlayRef.current !== normalizedUrl || disposed) {
            texture.dispose();
            return;
          }
          const image = texture.image as { width?: number; height?: number } | undefined;
          if (!image?.width || !image?.height) {
            texture.dispose();
            appliedOverlayUrl = null;
            setOverlayState({ phase: "error", message: "Analytical texture decoded without valid image dimensions." });
            return;
          }
          if (terrainMeshes.some((mesh) => !mesh.geometry.getAttribute("uv"))) {
            texture.dispose();
            appliedOverlayUrl = null;
            setOverlayState({ phase: "error", message: "Terrain mesh has no UV coordinates required for analytical projection." });
            return;
          }
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.flipY = false;
          texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
          texture.minFilter = THREE.LinearMipmapLinearFilter;
          texture.magFilter = THREE.LinearFilter;
          texture.generateMipmaps = true;
          texture.needsUpdate = true;
          overlayTexture = texture;

          for (const mesh of terrainMeshes) {
            const original = originalMaterials.get(mesh);
            if (!original) continue;
            const nextMaterials = materialArray(original).map((material) => overlayMaterialFrom(material, texture));
            mesh.material = Array.isArray(original) ? nextMaterials : nextMaterials[0];
          }
          pendingOverlayGeneration = generation;
          pendingOverlayFrames = 0;
          pendingOverlayStartedAt = performance.now();
          setOverlayState({ phase: "loading", message: "Analytical material applied · validating rendered frame…" });
        },
        undefined,
        (error) => {
          if (generation !== overlayLoadGeneration || disposed) return;
          restoreOriginalMaterials();
          appliedOverlayUrl = null;
          setOverlayState({
            phase: "error",
            message: `Analytical overlay could not be rendered: ${error instanceof Error ? error.message : String(error)}`,
          });
        },
      );
    };

    loader.load(
      meshUrl,
      (gltf) => {
        if (disposed) return;
        loaded = gltf.scene;
        terrainMeshes = [];
        let geometryVertices = 0;
        loaded.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            terrainMeshes.push(object);
            originalMaterials.set(object, object.material);
            const position = object.geometry.getAttribute("position");
            geometryVertices += position?.count ?? 0;
          }
        });
        if (terrainMeshes.length === 0 || geometryVertices < 3) {
          fail("Terrain GLB parsed but contained no renderable triangle geometry.");
          return;
        }
        scene.add(loaded);
        const unscaledBounds = new THREE.Box3().setFromObject(loaded);
        elevationCenter = unscaledBounds.getCenter(new THREE.Vector3()).y;
        appliedExaggeration = 1;
        applyExaggeration();
        refreshBounds();
        flyBaseSpeed = THREE.MathUtils.clamp(footprint() * 0.04, 80, 3000);
        firstPersonBaseSpeed = THREE.MathUtils.clamp(Math.max(firstPersonClearance() * 0.8, footprint() * 0.006), 20, 900);
        updateNavigationSpeed();
        fitView();
        positionMarker(cursorRef.current);
        refreshAnalysisPath();
        applyOverlay(overlayRef.current);
        modelLoadedAt = performance.now();
        publishState({
          phase: "loading",
          message: "Preparing GPU resources and validating the first terrain frame…",
          triangles: 0,
          drawCalls: 0,
        });
      },
      (progress) => {
        if (disposed || fatal || !progress.total) return;
        const percent = Math.min(100, Math.max(0, Math.round((progress.loaded / progress.total) * 100)));
        publishState({
          phase: "loading",
          message: percent >= 100 ? "Terrain bytes received · preparing GPU resources…" : `Loading persistent terrain LOD… ${percent}%`,
          triangles: 0,
          drawCalls: 0,
        });
      },
      (error) => {
        fail(`Terrain GLB could not be loaded: ${error instanceof Error ? error.message : String(error)}`);
      },
    );

    const resize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = Math.max(width / Math.max(height, 1), 0.01);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    let pointerDown: { x: number; y: number } | null = null;
    let shiftPan = false;
    const onFocus = () => { canvasFocused = true; };
    const onBlur = () => {
      canvasFocused = false;
      shiftBoost = false;
      updateNavigationSpeed();
    };
    renderer.domElement.addEventListener("focus", onFocus);
    renderer.domElement.addEventListener("blur", onBlur);

    const onPointerDown = (event: PointerEvent) => {
      renderer.domElement.focus({ preventScroll: true });
      if (event.button !== 0) return;
      pointerDown = { x: event.clientX, y: event.clientY };
      shiftPan = event.shiftKey;
      if (shiftPan && modeRef.current === "orbit") orbit.mouseButtons.LEFT = THREE.MOUSE.PAN;
    };
    const onPointerUp = (event: PointerEvent) => {
      if (modeRef.current !== "topDown") orbit.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
      shiftPan = false;
      if (!pointerDown || event.button !== 0 || !loaded || terrainMeshes.length === 0 || !rendererReady) {
        pointerDown = null;
        return;
      }
      const movement = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
      pointerDown = null;
      if (movement > 5 || autoFlythroughRef.current || !["orbit", "topDown"].includes(modeRef.current)) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(terrainMeshes, false)[0];
      if (!hit?.uv) return;
      const point = {
        x: THREE.MathUtils.clamp(hit.uv.x, 0, 1),
        y: THREE.MathUtils.clamp(1 - hit.uv.y, 0, 1),
      };
      positionMarker(point);
      selectRef.current?.(point);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointerup", onPointerUp);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Shift" && canvasFocused && !shiftBoost) {
        shiftBoost = true;
        updateNavigationSpeed();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Shift" && shiftBoost) {
        shiftBoost = false;
        updateNavigationSpeed();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    const clock = new THREE.Clock();
    let previousMode: CameraMode | null = null;
    let previousCursorKey = "";
    let previousPathKey = "";
    let previousOverlay = "";
    let previousReset = resetRef.current;
    let flythroughStartedAt = 0;
    let wasAutoFlythrough = false;
    let performanceSeconds = 0;
    let performanceFrames = 0;
    let frame = 0;
    const animate = () => {
      if (disposed) return;
      const mode = modeRef.current;
      const dt = Math.min(clock.getDelta(), 0.05);
      const touring = autoFlythroughRef.current && Boolean(loaded) && rendererReady;
      orbit.enabled = rendererReady && !touring && (mode === "orbit" || mode === "topDown");
      freeCamera.enabled = rendererReady && !touring && canvasFocused && (mode === "fly" || mode === "firstPerson");

      if (mode === "topDown") {
        orbit.enableRotate = false;
        orbit.screenSpacePanning = true;
        orbit.mouseButtons.LEFT = THREE.MOUSE.PAN;
        orbit.mouseButtons.RIGHT = THREE.MOUSE.PAN;
      } else {
        orbit.enableRotate = true;
        orbit.screenSpacePanning = false;
        if (!shiftPan) orbit.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
        orbit.mouseButtons.RIGHT = THREE.MOUSE.PAN;
      }

      if (mode !== previousMode) {
        updateNavigationSpeed();
        if (mode === "topDown" && loaded) {
          const distance = footprint() * 1.05;
          camera.up.set(0, 0, -1);
          camera.position.set(sceneCenter.x, sceneCenter.y + distance, sceneCenter.z + 0.001);
          camera.lookAt(sceneCenter);
          orbit.target.copy(sceneCenter);
        } else if (mode === "firstPerson" && loaded) {
          enterFirstPerson();
        } else if (mode === "orbit" && loaded && previousMode && previousMode !== "orbit") {
          fitView();
        } else if (mode !== "topDown") {
          camera.up.set(0, 1, 0);
        }
      }

      applyExaggeration();
      const currentCursor = cursorRef.current;
      const cursorKey = currentCursor ? `${currentCursor.x.toFixed(7)}:${currentCursor.y.toFixed(7)}` : "";
      if (cursorKey !== previousCursorKey) {
        positionMarker(currentCursor);
        previousCursorKey = cursorKey;
      }
      const pathKey = analysisPathRef.current
        .map((point) => `${point.x.toFixed(5)}:${point.y.toFixed(5)}`)
        .join("|");
      if (pathKey !== previousPathKey) {
        refreshAnalysisPath();
        previousPathKey = pathKey;
      }
      const overlayKey = overlayRef.current ?? "";
      if (overlayKey !== previousOverlay) {
        applyOverlay(overlayRef.current);
        previousOverlay = overlayKey;
      }
      if (resetRef.current !== previousReset) {
        fitView();
        previousReset = resetRef.current;
      }

      if (touring && loaded) {
        if (!wasAutoFlythrough) flythroughStartedAt = performance.now() / 1000;
        const elapsed = performance.now() / 1000 - flythroughStartedAt;
        const phase = (elapsed / 18) * Math.PI * 2;
        const radius = footprint() * 0.92;
        const height = Math.max(sceneSize.y * 3.5, footprint() * (0.32 + 0.07 * Math.sin(phase * 2)));
        camera.up.set(0, 1, 0);
        camera.position.set(
          sceneCenter.x + Math.cos(phase) * radius,
          sceneCenter.y + height,
          sceneCenter.z + Math.sin(phase) * radius,
        );
        camera.lookAt(sceneCenter);
        orbit.target.copy(sceneCenter);
      } else if (wasAutoFlythrough && loaded) {
        fitView();
      }
      wasAutoFlythrough = touring;
      previousMode = mode;

      if (orbit.enabled) orbit.update();
      if (freeCamera.enabled) {
        freeCamera.update(dt);
        if (mode === "firstPerson") constrainFirstPerson();
      }
      renderer.render(scene, camera);

      const triangles = renderer.info.render.triangles;
      const drawCalls = renderer.info.render.calls;
      if (!rendererReady && loaded && !fatal) {
        if (triangles > 0 && drawCalls > 0) {
          rendererReady = true;
          publishState({
            phase: "ready",
            message: "Terrain renderer ready",
            triangles,
            drawCalls,
          });
        } else if (modelLoadedAt > 0 && performance.now() - modelLoadedAt > 5000) {
          fail("Terrain GLB loaded, but no renderable frame was produced within 5 seconds (0 triangles / 0 draw calls).");
        }
      }

      if (pendingOverlayGeneration !== null) {
        const validMaterialContract = Boolean(overlayTexture)
          && terrainMeshes.length > 0
          && terrainMeshes.every((mesh) => {
            if (!mesh.geometry.getAttribute("uv")) return false;
            return materialArray(mesh.material).every((material) => (
              material instanceof THREE.MeshBasicMaterial
              && material.map === overlayTexture
              && material.side === THREE.DoubleSide
            ));
          });
        if (triangles > 0 && drawCalls > 0 && validMaterialContract) {
          pendingOverlayFrames += 1;
          if (pendingOverlayFrames >= 2) {
            pendingOverlayGeneration = null;
            setOverlayState({ phase: "ready", message: "Analytical overlay rendered and frame-validated" });
          }
        } else if (performance.now() - pendingOverlayStartedAt > 3000) {
          restoreOriginalMaterials();
          appliedOverlayUrl = null;
          setOverlayState({
            phase: "error",
            message: "Analytical material was applied but did not produce a valid rendered terrain frame; source texture restored.",
          });
        }
      }

      if (rendererReady) {
        performanceSeconds += dt;
        performanceFrames += 1;
        if (performanceSeconds >= 1) {
          performanceRef.current?.({
            fps: performanceFrames / performanceSeconds,
            triangles,
            drawCalls,
          });
          performanceSeconds = 0;
          performanceFrames = 0;
        }
      }
      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      renderer.domElement.removeEventListener("focus", onFocus);
      renderer.domElement.removeEventListener("blur", onBlur);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      orbit.dispose();
      freeCamera.dispose();
      restoreOriginalMaterials();
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          materialArray(object.material).forEach((material) => material.dispose());
        }
      });
      pathLine.geometry.dispose();
      (pathLine.material as THREE.Material).dispose();
      renderer.domElement.remove();
    };
  }, [groundSampleDistanceM, meshUrl, retryGeneration, overlayRetryGeneration]);

  return (
    <div className="dw-terrain-viewport">
      <div ref={hostRef} className="dw-terrain-render-host" aria-label="3D terrain viewport" />
      {renderState.phase === "ready" && (
        <div className="dw-terrain-mode-help" role="status">{navigationHelp(cameraMode, autoFlythrough)}</div>
      )}
      {overlayState.phase === "loading" && overlayUrl && (
        <div className="dw-terrain-overlay-state" role="status">{overlayState.message}</div>
      )}
      {overlayState.phase === "error" && overlayUrl && (
        <div className="dw-terrain-overlay-state dw-terrain-overlay-state--error" role="alert">
          <strong>Analytical overlay unavailable</strong>
          <span>{overlayState.message}</span>
          <button type="button" className="dw-overlay-retry" onClick={() => setOverlayRetryGeneration((value) => value + 1)}>Retry overlay</button>
        </div>
      )}
      {renderState.phase === "loading" && (
        <div className="dw-render-state" role="status">
          <span className="dw-spinner" aria-hidden="true" />
          <strong>Preparing 3D terrain</strong>
          <p>{renderState.message}</p>
        </div>
      )}
      {renderState.phase === "error" && (
        <div className="dw-render-state dw-render-state--error" role="alert">
          <strong>Terrain renderer failed</strong>
          <p>{renderState.message}</p>
          <div className="dw-render-state-actions">
            <button type="button" className="dw-btn dw-btn--primary" onClick={() => setRetryGeneration((value) => value + 1)}>Retry renderer</button>
            <details>
              <summary>Open diagnostics</summary>
              <code>phase={renderState.phase}\ntriangles={renderState.triangles}\ndrawCalls={renderState.drawCalls}\nmesh={meshUrl?.startsWith("blob:") ? "authenticated blob-backed GLB" : meshUrl ?? "none"}</code>
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
