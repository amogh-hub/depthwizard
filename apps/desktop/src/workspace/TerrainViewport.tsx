import { useEffect, useRef } from "react";
import * as THREE from "three";
import { FlyControls } from "three/examples/jsm/controls/FlyControls.js";
import { FirstPersonControls } from "three/examples/jsm/controls/FirstPersonControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export type CameraMode = "orbit" | "fly" | "firstPerson" | "topDown";

export function TerrainViewport({ meshUrl, cameraMode }: { meshUrl?: string; cameraMode: CameraMode }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const modeRef = useRef(cameraMode);
  modeRef.current = cameraMode;

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !meshUrl) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe7ebef);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1_000_000);
    camera.position.set(0, 250, 350);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x667085, 2.3));
    const sun = new THREE.DirectionalLight(0xffffff, 2.1);
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

    const loader = new GLTFLoader();
    let loaded: THREE.Object3D | undefined;
    let sceneCenter = new THREE.Vector3();
    let sceneSize = new THREE.Vector3(100, 100, 100);
    loader.load(meshUrl, (gltf) => {
      loaded = gltf.scene;
      scene.add(gltf.scene);
      const bounds = new THREE.Box3().setFromObject(gltf.scene);
      sceneCenter = bounds.getCenter(new THREE.Vector3());
      sceneSize = bounds.getSize(new THREE.Vector3());
      orbit.target.copy(sceneCenter);
      const distance = Math.max(sceneSize.x, sceneSize.y, sceneSize.z) * 1.4;
      camera.position.set(
        sceneCenter.x + distance * 0.45,
        sceneCenter.y + distance * 0.65,
        sceneCenter.z + distance * 0.75,
      );
      camera.near = Math.max(distance / 10000, 0.01);
      camera.far = Math.max(distance * 20, 1000);
      camera.updateProjectionMatrix();
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

    const clock = new THREE.Clock();
    let previousMode: CameraMode | null = null;
    let frame = 0;
    const animate = () => {
      const mode = modeRef.current;
      const dt = Math.min(clock.getDelta(), 0.05);
      orbit.enabled = mode === "orbit" || mode === "topDown";
      fly.enabled = mode === "fly";
      firstPerson.enabled = mode === "firstPerson";

      if (mode !== previousMode && mode === "topDown" && loaded) {
        const distance = Math.max(sceneSize.x, sceneSize.z) * 1.2;
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
