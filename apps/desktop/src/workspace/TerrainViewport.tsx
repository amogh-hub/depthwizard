import { useEffect, useRef } from "react";
import * as THREE from "three";
import { FlyControls } from "three/examples/jsm/controls/FlyControls.js";
import { FirstPersonControls } from "three/examples/jsm/controls/FirstPersonControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { NormalizedPoint } from "../api";

export type CameraMode = "orbit" | "fly" | "firstPerson" | "topDown";

type TerrainViewportProps = {
  meshUrl?: string;
  cameraMode: CameraMode;
  verticalExaggeration?: number;
  cursorPoint?: NormalizedPoint | null;
  onSelectPoint?: (point: NormalizedPoint) => void;
};

export function TerrainViewport({
  meshUrl,
  cameraMode,
  verticalExaggeration = 1,
  cursorPoint,
  onSelectPoint,
}: TerrainViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const modeRef = useRef(cameraMode);
  const exaggerationRef = useRef(verticalExaggeration);
  const cursorRef = useRef<NormalizedPoint | null | undefined>(cursorPoint);
  const selectRef = useRef(onSelectPoint);
  modeRef.current = cameraMode;
  exaggerationRef.current = verticalExaggeration;
  cursorRef.current = cursorPoint;
  selectRef.current = onSelectPoint;

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !meshUrl) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xebeff3);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1_000_000);
    camera.position.set(0, 250, 350);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x728094, 2.15));
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(300, 700, 240);
    scene.add(sun);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.screenSpacePanning = false;

    const fly = new FlyControls(camera, renderer.domElement);
    fly.movementSpeed = 50;
    fly.rollSpeed = 0.35;
    fly.dragToLook = true;

    const firstPerson = new FirstPersonControls(camera, renderer.domElement);
    firstPerson.movementSpeed = 35;
    firstPerson.lookSpeed = 0.08;
    firstPerson.lookVertical = true;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x1f5fae, emissiveIntensity: 1.8 }),
    );
    marker.visible = false;
    marker.renderOrder = 10;
    scene.add(marker);

    const loader = new GLTFLoader();
    let loaded: THREE.Object3D | undefined;
    let sceneCenter = new THREE.Vector3();
    let sceneSize = new THREE.Vector3(100, 100, 100);
    let elevationCenter = 0;
    let appliedExaggeration = 1;
    let terrainMeshes: THREE.Mesh[] = [];

    const refreshBounds = () => {
      if (!loaded) return;
      const bounds = new THREE.Box3().setFromObject(loaded);
      sceneCenter = bounds.getCenter(new THREE.Vector3());
      sceneSize = bounds.getSize(new THREE.Vector3());
    };

    const applyExaggeration = () => {
      if (!loaded) return;
      const exaggeration = Math.max(exaggerationRef.current, 0.1);
      if (Math.abs(exaggeration - appliedExaggeration) < 1e-6) return;
      loaded.scale.y = exaggeration;
      loaded.position.y = elevationCenter * (1 - exaggeration);
      appliedExaggeration = exaggeration;
      loaded.updateMatrixWorld(true);
      refreshBounds();
    };

    const positionMarker = (point: NormalizedPoint | null | undefined) => {
      if (!loaded || !point || terrainMeshes.length === 0) {
        marker.visible = false;
        return;
      }
      const bounds = new THREE.Box3().setFromObject(loaded);
      const size = bounds.getSize(new THREE.Vector3());
      const x = bounds.min.x + point.x * size.x;
      const z = bounds.max.z - point.y * size.z;
      const origin = new THREE.Vector3(x, bounds.max.y + Math.max(size.y, 10) + 10, z);
      raycaster.set(origin, new THREE.Vector3(0, -1, 0));
      const hit = raycaster.intersectObjects(terrainMeshes, false)[0];
      if (!hit) {
        marker.visible = false;
        return;
      }
      marker.position.copy(hit.point);
      marker.position.y += Math.max(Math.max(size.x, size.z) * 0.0025, 0.5);
      const radius = Math.max(Math.max(size.x, size.z) * 0.004, 0.7);
      marker.scale.setScalar(radius);
      marker.visible = true;
    };

    loader.load(meshUrl, (gltf) => {
      loaded = gltf.scene;
      terrainMeshes = [];
      loaded.traverse((object) => {
        if (object instanceof THREE.Mesh) terrainMeshes.push(object);
      });
      scene.add(loaded);
      const unscaledBounds = new THREE.Box3().setFromObject(loaded);
      elevationCenter = unscaledBounds.getCenter(new THREE.Vector3()).y;
      appliedExaggeration = 1;
      applyExaggeration();
      refreshBounds();

      const footprint = Math.max(sceneSize.x, sceneSize.z);
      const distance = Math.max(footprint, sceneSize.y * 2) * 0.82;
      const viewTarget = sceneCenter.clone();
      viewTarget.y -= Math.max(sceneSize.y, footprint * 0.08) * 0.20;
      orbit.target.copy(viewTarget);
      camera.position.set(
        sceneCenter.x + distance * 0.52,
        sceneCenter.y + distance * 0.48,
        sceneCenter.z + distance * 0.78,
      );
      camera.near = Math.max(distance / 10000, 0.01);
      camera.far = Math.max(distance * 20, 1000);
      camera.updateProjectionMatrix();
      orbit.update();
      positionMarker(cursorRef.current);
    });

    const resize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = Math.max(width / Math.max(height, 1), 0.01);
      camera.updateProjectionMatrix();
      firstPerson.handleResize();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    let pointerDown: { x: number; y: number } | null = null;
    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0) return;
      pointerDown = { x: event.clientX, y: event.clientY };
    };
    const onPointerUp = (event: PointerEvent) => {
      if (!pointerDown || event.button !== 0 || !loaded || terrainMeshes.length === 0) {
        pointerDown = null;
        return;
      }
      const movement = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
      pointerDown = null;
      if (movement > 5) return;
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

    const clock = new THREE.Clock();
    let previousMode: CameraMode | null = null;
    let previousCursorKey = "";
    let frame = 0;
    const animate = () => {
      const mode = modeRef.current;
      const dt = Math.min(clock.getDelta(), 0.05);
      orbit.enabled = mode === "orbit" || mode === "topDown";
      fly.enabled = mode === "fly";
      firstPerson.enabled = mode === "firstPerson";

      applyExaggeration();
      const currentCursor = cursorRef.current;
      const cursorKey = currentCursor ? `${currentCursor.x.toFixed(7)}:${currentCursor.y.toFixed(7)}` : "";
      if (cursorKey !== previousCursorKey) {
        positionMarker(currentCursor);
        previousCursorKey = cursorKey;
      }

      if (mode !== previousMode && mode === "topDown" && loaded) {
        const distance = Math.max(sceneSize.x, sceneSize.z) * 1.05;
        camera.up.set(0, 0, -1);
        camera.position.set(sceneCenter.x, sceneCenter.y + distance, sceneCenter.z + 0.001);
        camera.lookAt(sceneCenter);
        orbit.target.copy(sceneCenter);
      } else if (mode !== "topDown") {
        camera.up.set(0, 1, 0);
      }
      previousMode = mode;

      if (orbit.enabled) orbit.update();
      if (fly.enabled) fly.update(dt);
      if (firstPerson.enabled) firstPerson.update(dt);
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      orbit.dispose();
      fly.dispose();
      firstPerson.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      renderer.domElement.remove();
    };
  }, [meshUrl]);

  return (
    <div
      ref={hostRef}
      style={{ position: "absolute", inset: 0 }}
      aria-label="3D terrain viewport"
    />
  );
}
