# DepthWizard standalone architecture

## Purpose

Release Train 5 turns DepthWizard from a desktop frontend that expects a separately started Python service into a true no-terminal scientific workstation. The Tauri shell owns the local scientific runtime lifecycle and exposes only a session-scoped loopback endpoint to the webview.

## Runtime lifecycle

1. Tauri reserves an ephemeral `127.0.0.1` port.
2. Tauri generates a fresh 256-bit random session token.
3. Tauri resolves the qualified scientific runtime from its bundled resource tree.
4. Tauri launches `depthwizard-core` directly as an owned child process, passing the token through the child environment rather than command-line arguments.
5. The Python sidecar validates that packaged mode is loopback-only and that a session token is present.
6. Tauri polls the minimal `/health` endpoint until the core is ready or the startup timeout expires.
7. The React webview obtains `{apiBase, sessionToken}` through a Tauri IPC command before rendering the main application.
8. Every scientific API request carries `x-depthwizard-token`; missing or incorrect tokens are rejected.
9. Tauri terminates and reaps the owned scientific child when the application exits.

The child is launched directly with Rust `std::process::Command`; a general-purpose shell is not required for the production scientific runtime.

## Frozen-runtime packaging policy

DepthWizard uses **PyInstaller onedir**, not onefile, for the scientific runtime. This is deliberate. The runtime contains PyTorch, DA3, Rasterio/GDAL, PROJ, SciPy and other native scientific dependencies. A onefile executable must extract that native environment on every launch and makes bootstrap failures opaque. The onedir runtime stays directly inspectable and is embedded inside the final Tauri `.app` as a resource tree, so the user still launches one normal desktop application.

The staged runtime lives at:

`apps/desktop/src-tauri/resources/depthwizard-core-runtime/`

The directory is generated, ignored by Git, and bundled by Tauri into:

`Contents/Resources/depthwizard-core-runtime/`

PyInstaller symbolic links are preserved when the runtime tree is staged. The build report records a deterministic tree SHA-256, regular-file count, symlink count, logical bytes, executable SHA-256 and the target triple.

## Build-time frozen-runtime qualification

`make sidecar-build` does not declare success merely because PyInstaller produced files. After staging the exact runtime tree that Tauri will bundle, the build script launches that frozen executable with `--self-check` under offline environment flags.

The self-check must successfully:

- reach the frozen Python entrypoint;
- import `rasterio.serde`;
- initialize Rasterio/GDAL;
- initialize PyProj/PROJ;
- resolve EPSG:32643 through PyProj;
- write and reopen an in-memory GeoTIFF with EPSG:32643 through Rasterio/GDAL;
- complete without model loading or network use.

A build that times out, crashes, cannot resolve bundled geospatial data, or produces malformed evidence is rejected. Only a passing runtime receives `QUALIFIED_DEPTHWIZARD_CORE_RUNTIME` in its bundled `runtime-manifest.json` and `PASS_QUALIFIED_SIDECAR_BUILD` in the external build report.

## Startup diagnostics

When `DEPTHWIZARD_STARTUP_TRACE=<path>` is explicitly set for acceptance or diagnostics, the scientific core emits machine-readable JSONL startup phases. The trace never records the session token. Normal launches do not write a trace. Acceptance failures include the last observed startup phase instead of returning an opaque connection-refused timeout.

The qualified Apple Silicon runtime has demonstrated a cold packaged-core readiness time of roughly 41 seconds. Tauri therefore uses a 90-second liveness watchdog. This watchdog is a correctness bound, not a performance claim; finale-Mac FPS and responsiveness qualification remains RT7 evidence.

## Offline-after-install contract

Packaged launches set:

- `DEPTHWIZARD_OFFLINE_CORE=1`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `PROJ_NETWORK=OFF`

When `DEPTHWIZARD_OFFLINE_CORE=1`, the sidecar also installs a process-wide Python INET connection guard. IPv4/IPv6 connections are permitted only to explicit loopback destinations (`127.0.0.0/8`, `::1`, or `localhost`). Other hostnames are not DNS-resolved and are rejected before connection. This is an application-layer egress barrier in addition to the Hugging Face/Transformers offline flags; it is not presented as an operating-system firewall.

The final core uses model assets already installed/cached on the machine. It must not silently download model weights during an offline judging run. The consolidated RT5 acceptance proves that real DA3 reconstruction succeeds while this packaged offline policy is active.

## Acceptance-only control hooks

Normal production launches do not write the session token to disk. Deterministic acceptance has three explicit opt-in hooks, all disabled unless their environment variables are supplied by the acceptance runner:

- `DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT=<path>` writes a non-secret boot report after the sidecar is healthy. It records endpoint, PID, offline state and token bit length, never the token value.
- `DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS=<500..60000>` supports the short bundle-lifecycle smoke.
- `DEPTHWIZARD_ACCEPTANCE_CONTROL_PATH=<path>` creates a mode-0600 ephemeral control file containing the loopback endpoint, token and sidecar PID so the external RT5 acceptance runner can drive the already-running Tauri-owned sidecar through the same authenticated API used by the desktop. The runner deletes this file immediately after reading it, and Tauri removes it again on shutdown if necessary.
- `DEPTHWIZARD_ACCEPTANCE_EXIT_SIGNAL=<path>` lets the RT5 runner request a normal Tauri exit after the full scientific workflow, so Rust shutdown still kills and reaps the owned sidecar.

The ephemeral control channel is test-only and is never used by normal production launches. Its token is never copied into final evidence artifacts.

## RT5 acceptance

`make release-train-5-sidecar-smoke` validates the already-qualified frozen runtime's process/security contract: loopback readiness, startup phases, missing/wrong-token rejection, authorized raster inspection, offline policy and clean process termination.

`make release-train-5-app-smoke` validates the real packaged `.app`: qualified runtime resource integrity, Tauri-owned startup, loopback readiness, non-export of the token to evidence, offline policy, no user-visible terminal and child termination on graceful desktop exit.

`make release-train-5-full-smoke` launches the real packaged `DepthWizard.app` and then, through the authenticated loopback sidecar owned by that same Tauri process, proves:

- a corrupt raster is rejected without losing the service;
- a deliberately failing runtime job reaches terminal `failed` state instead of wedging the queue;
- the service remains healthy after that runtime failure;
- a subsequent fresh OrthoLoC optical project completes through the packaged DA3 path with offline policy active;
- DSM/rDSM/slope artifacts are present and hash-consistent with the project manifest;
- downstream reference validation succeeds;
- packaged terrain mesh/LOD generation succeeds;
- packaged project export succeeds and the ZIP passes integrity verification;
- the application exits normally and its Rust-owned sidecar is terminated and reaped.

`make release-train-5-standalone-acceptance` runs all three RT5 acceptance layers in one command.

## Frozen release-train boundary

A passing consolidated RT5 acceptance closes **RT5 — Standalone Production Suite engineering scope**: packaged scientific runtime, real no-terminal Tauri application lifecycle, secure loopback/session model, offline DA3 execution, fresh process→validate→mesh→export flow, and recoverable malformed/runtime failures.

It does **not** consume or replace the already-frozen later release trains:

- **RT6 — Final Scientific Evidence Campaign** owns the four-terrain geographically disjoint benchmark, cross-sensor holdout, same-input baselines, ablations, seam diagnostics, confidence reliability, structural-height/slope/projection validation and final claim matrix.
- **RT7 — Finale Qualification & Submission** owns finale-Mac FPS evidence, the two-hour stability soak, clean-machine installation/reproducibility, final documentation/evidence audit, real screenshots, six-slide SIH PDF, demo video and judge Q&A package.

Therefore RT5 engineering acceptance must never be presented as DSM-accuracy, terrain-generalization, cross-sensor or learned-model-promotion evidence, and it must not be used to pre-claim the RT7 performance/soak/clean-machine gates.
