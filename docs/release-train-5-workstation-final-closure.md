# Release Train 5 — Final Workstation Closure Contract

This document freezes the corrective desktop-workstation scope discovered during real Apple Silicon operator testing after the original RT5 standalone scientific/runtime qualification.

It does **not** replace or invalidate already-earned RT5 scientific/runtime evidence. It closes manual workstation integration and usability defects without moving RT6 scientific-evidence gates or RT7 finale-Mac qualification gates into RT5.

## Acceptance principle

The desktop is accepted only when scientific product state, display-asset state, interaction state, and renderer state are independently truthful. A generated DSM or GLB artifact is not enough to claim that its current viewport is ready.

## Durable project restoration

- A completed project must reopen from its persisted manifest without rerunning DA3 or calibration.
- Persisted DSM/rDSM, derived previews, mesh, validation, and export reports remain usable after restart.
- Optical preview is regenerated from the original source when that source still exists.
- If the original source was moved or removed, DepthWizard must not fabricate Optical RGB. Persisted scientific elevation products remain usable and source-dependent processing actions are disabled.
- A stale Recent-project entry fails cleanly and is removed from the convenience list.
- Display-preview failures have explicit error/retry states and must never appear as ordinary scientific `ready` states.

## 2D registered scientific viewport

- Left-drag pans the registered raster canvas in Navigate mode.
- Wheel/trackpad zoom is cursor-anchored; touch pinch is supported.
- Double-click zoom, Fit, zoom percentage, +/- zoom, and 1:1 are available.
- The entire central viewport participates in navigation, not only the raster's initial fitted rectangle.
- Shared raster view state preserves geographic center/zoom across Optical, DSM, Slope, Hillshade, Contours, Reference, Residual, Confidence, and Compare.
- Loading is atomic: stale prior-layer pixels are removed before a new layer is labeled active.
- Map scale and scientific quantitative legend are rendered where the underlying semantics support them.
- Analytical clicks are separated from drag gestures by an explicit movement threshold.
- Space+drag temporarily pans in Measure/Profile/Structure modes.

## Analyst workflows

- Measure and Profiles default to the registered 2D DSM workspace when a raster workflow is required.
- Structure measurement requires a metric DSM surface.
- Measure: A/B selection, plan distance, endpoint elevation change, clear/cancel behavior.
- Profiles: live A→B transect preview, persisted-surface sampling, elevation chart, gain/loss, and optional reference line.
- Structures: explicit footprint vertices, draggable edit handles, undo/delete/escape, and robust local-ground structural-height calculation.
- Tools that require a raster surface are disabled if that active surface failed to load.
- Switching tools clears unrelated stale probe/transect/structure state.
- When the 3D terrain is already active and renderer-ready, Project, Measure, Profiles, and metric Structures may perform evidence-native 3D point selection. The selected world point is converted back to the authoritative raster/project coordinate system before scientific sampling; camera gestures must never masquerade as analytical clicks.

## Comparison and validation

- Compare is unavailable until a valid independent reference validation exists.
- Prediction/reference comparison shares the registered 2D pan/zoom state and uses a swipe control.
- Missing comparison display assets receive an explicit error/retry state.
- Calibration evidence must never be presented as independent validation evidence.

## 3D renderer truthfulness

The renderer state progression is:

`mesh artifact exists → GLB fetch → GLB parse → GPU preparation → first valid frame → nonzero triangles/draw calls → renderer ready`

- Camera controls remain disabled before renderer-ready.
- Blank/zero-geometry renderers cannot report success or meaningful FPS.
- WebGL/context/GLB failures show an explicit failure with retry/diagnostics.
- Analytical overlay readiness is bound to renderer-validated overlay state, not merely to a successful image fetch.
- Auto LOD ignores invalid telemetry and uses sustained pressure; an approximately 30 fps screen-recording environment does not by itself force the lowest LOD.

## 3D navigation

- Orbit: drag rotate; Shift/right-drag pan; wheel dolly.
- Fly: terrain canvas must be focused; WASD movement; R/F vertical movement; drag-to-look; Shift acceleration.
- First Person: terrain canvas must be focused; deterministic near-surface entry; slower WASD/R/F free-camera movement; drag-to-look; Shift acceleration; Orbit is an explicit exit/reset path.
- Top Down: map-like overhead camera with pan and zoom.
- Flythrough: deterministic display-only tour; stopping restores a sensible fitted view.
- Fit restores stable scene framing.
- Vertical exaggeration is display-only and never changes sampled scientific values.

## 3D analytical overlays

- Texture, DSM, Slope, Hillshade, Contours, Confidence, and Residual are shown only when their real project products exist.
- Analytical overlay selection must visibly replace the texture material when loaded.
- Overlay state has explicit loading/ready/error semantics.
- A failed overlay restores the source terrain material and shows an error/retry state; the UI must not continue claiming that the analytical overlay rendered.
- Scientific legends correspond to the actually displayed analytical layer.

## Workstation chrome and responsive layout

- Tool rail is 56 px, icon-only, with accessible labels/tooltips; the redundant Layers rail mode is removed.
- The command bar wraps into stable groups instead of exposing a horizontal scrollbar.
- FPS, LOD, vertical exaggeration, camera modes, and layer controls remain single-line controls.
- Scientific context, navigation help, overlay state, legend, north indicator, and scene badge occupy deterministic non-overlapping lanes.
- Geospatial metadata uses readable one-line values with ellipsis/title rather than one-character wrapping.
- 3D telemetry appears only in the 3D workspace.
- Footer status describes the current view/action; an old completed export does not override active DSM/Optical/3D status.

## Exact-head packaging and sidecar identity contract

A workstation correction build must not reuse a stale staged Python sidecar. The supported local target is:

```bash
make release-train-5-workstation-build
```

It synchronizes the exact checkout, runs the repository verification suite, rebuilds the packaged Python sidecar, installs/tests the frontend, and then packages Tauri.

At packaged desktop startup, Tauri launches the sidecar with two independent 256-bit random values: the API session token and a per-process boot identity nonce. Readiness is accepted only when the loopback sidecar returns the expected boot nonce from the packaged-only boot endpoint; a different local process that wins the ephemeral-port race cannot satisfy that identity proof. The session token is not sent during readiness probing. After identity-verified boot, the desktop probes the workstation preview and legend routes through the authenticated loopback API. An incompatible/stale sidecar fails closed instead of opening a partially functional workstation.

## Reproducibility contract

Final RT7 source qualification requires all dependency-resolution surfaces to be frozen and audited:

- exact top-level npm dependency versions;
- committed `apps/desktop/package-lock.json`;
- committed `apps/desktop/src-tauri/Cargo.lock`;
- committed Python resolver lock (`uv.lock`);
- clean-machine installation/build using those locks rather than unconstrained resolution.

The source audit is a prerequisite only. A passing lock audit does not replace clean-machine packaged inference, accelerator compatibility, FPS, or soak evidence.

## Hosted CI quota exception

GitHub-hosted Actions quota was exhausted during the original corrective campaign. A workflow rejected before job steps execute is not code evidence. The last actually executed green hosted run remains historical evidence only; any newer exact head still requires local exact-head verification and one hosted exact-head rerun when hosted execution is available.

## Final Mac operator acceptance

After the exact-head workstation build passes locally, perform one continuous real-user run:

1. Open the existing Joshimath project.
2. Confirm DSM and available Optical restoration without rerunning reconstruction.
3. Exercise 2D pan/zoom/Fit/1:1 and synchronized layer changes.
4. Exercise Measure, Profiles, Structure edit/undo, stale-state cleanup, then repeat supported point-selection tasks from an already-active renderer-ready 3D workspace.
5. Exercise 3D loading/readiness, Orbit, Fly, First Person, Top Down, Flythrough, Fit, LOD, and vertical exaggeration.
6. Verify every available 3D analytical overlay visibly changes and has matching semantics/legend.
7. Confirm Compare stays gated without a valid reference; if an independent reference is supplied, exercise Compare and validation.
8. Export through the GUI.
9. Quit, relaunch, and reopen through Open Project / Recent.
10. Record the full session for defect audit.

Only after this operator acceptance passes is the corrective RT5 workstation campaign frozen. RT6 then begins. RT7 remains afterward for finale-Mac FPS evidence, two-hour soak, clean-machine reproducibility, final documentation, and submission qualification.
