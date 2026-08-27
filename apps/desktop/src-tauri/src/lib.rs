use rand::rngs::OsRng;
use rand::RngCore;
use serde::Serialize;
use std::env;
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager, RunEvent, State};

const SIDECAR_RESOURCE_PATH: &str = "depthwizard-core-runtime/depthwizard-core";
// The qualified Apple Silicon ONEDIR runtime currently needs ~41 s on a true cold launch. This
// watchdog is a correctness/liveness bound, not the RT7 performance target, so keep enough margin
// for first-launch dyld/filesystem work while still failing closed on a genuinely stuck sidecar.
const SIDECAR_READY_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeConfig {
    api_base: String,
    session_token: String,
    sidecar_pid: u32,
    offline_core: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceptanceBootReport<'a> {
    schema_version: u8,
    status: &'static str,
    api_base: &'a str,
    sidecar_pid: u32,
    offline_core: bool,
    session_token_bits: u16,
    session_token_exported: bool,
    strict_python_egress_guard: bool,
    ephemeral_acceptance_control_enabled: bool,
    build_git_sha: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceptanceControl<'a> {
    schema_version: u8,
    api_base: &'a str,
    session_token: &'a str,
    sidecar_pid: u32,
}

struct SidecarState {
    child: Mutex<Option<Child>>,
    runtime: RuntimeConfig,
    acceptance_control_path: Option<PathBuf>,
}

impl SidecarState {
    fn shutdown(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                terminate_child(&mut child);
            }
        }
        if let Some(path) = self.acceptance_control_path.as_ref() {
            let _ = fs::remove_file(path);
        }
    }
}

#[tauri::command]
fn runtime_config(state: State<'_, SidecarState>) -> RuntimeConfig {
    state.runtime.clone()
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn generate_session_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    let mut token = String::with_capacity(64);
    for byte in bytes {
        write!(&mut token, "{byte:02x}").expect("writing into a String cannot fail");
    }
    token
}

fn reserve_loopback_port() -> io::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

fn health_check(port: u16) -> io::Result<bool> {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(300))?;
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    stream.set_write_timeout(Some(Duration::from_millis(500)))?;
    stream.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")?;
    let mut response = [0_u8; 512];
    let count = stream.read(&mut response)?;
    let head = String::from_utf8_lossy(&response[..count]);
    Ok(head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200"))
}

fn wait_for_health(port: u16, timeout: Duration) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    let mut last_error: Option<io::Error> = None;
    while Instant::now() < deadline {
        match health_check(port) {
            Ok(true) => return Ok(()),
            Ok(false) => {
                last_error = Some(io::Error::other(
                    "DepthWizard sidecar health endpoint returned a non-200 response",
                ));
            }
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(last_error.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::TimedOut,
            "DepthWizard sidecar did not become healthy before timeout",
        )
    }))
}

fn write_acceptance_boot_report(runtime: &RuntimeConfig) -> io::Result<()> {
    let Ok(path) = env::var("DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT") else {
        return Ok(());
    };
    let report = AcceptanceBootReport {
        schema_version: 2,
        status: "PASS_TAURI_SIDECAR_BOOT",
        api_base: &runtime.api_base,
        sidecar_pid: runtime.sidecar_pid,
        offline_core: runtime.offline_core,
        session_token_bits: (runtime.session_token.len() * 4) as u16,
        session_token_exported: false,
        strict_python_egress_guard: runtime.offline_core,
        ephemeral_acceptance_control_enabled: env::var_os("DEPTHWIZARD_ACCEPTANCE_CONTROL_PATH")
            .is_some(),
        build_git_sha: env!("DEPTHWIZARD_BUILD_GIT_SHA"),
    };
    let payload = serde_json::to_string_pretty(&report).map_err(io::Error::other)?;
    fs::write(path, format!("{payload}\n"))
}

fn write_acceptance_control(runtime: &RuntimeConfig) -> io::Result<Option<PathBuf>> {
    let Ok(raw_path) = env::var("DEPTHWIZARD_ACCEPTANCE_CONTROL_PATH") else {
        return Ok(None);
    };
    let path = PathBuf::from(raw_path);
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("acceptance control path already exists: {path:?}"),
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let control = AcceptanceControl {
        schema_version: 1,
        api_base: &runtime.api_base,
        session_token: &runtime.session_token,
        sidecar_pid: runtime.sidecar_pid,
    };
    let payload = serde_json::to_string(&control).map_err(io::Error::other)?;

    let write_result = (|| -> io::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path)?;
        file.write_all(payload.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&path);
        return Err(error);
    }
    Ok(Some(path))
}

fn schedule_acceptance_auto_exit(app_handle: AppHandle) -> io::Result<()> {
    let Ok(raw) = env::var("DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS") else {
        return Ok(());
    };
    let millis = raw.parse::<u64>().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS: {error}"),
        )
    })?;
    if !(500..=60_000).contains(&millis) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS must be between 500 and 60000",
        ));
    }
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(millis));
        app_handle.exit(0);
    });
    Ok(())
}

fn schedule_acceptance_exit_signal(app_handle: AppHandle) -> io::Result<()> {
    let Ok(raw_path) = env::var("DEPTHWIZARD_ACCEPTANCE_EXIT_SIGNAL") else {
        return Ok(());
    };
    let path = PathBuf::from(raw_path);
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("acceptance exit signal path already exists: {path:?}"),
        ));
    }
    thread::spawn(move || loop {
        if path.is_file() {
            let _ = fs::remove_file(&path);
            app_handle.exit(0);
            break;
        }
        thread::sleep(Duration::from_millis(100));
    });
    Ok(())
}

fn sidecar_executable(app: &tauri::App) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = app
        .path()
        .resolve(SIDECAR_RESOURCE_PATH, BaseDirectory::Resource)?;
    let metadata = fs::metadata(&path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("DepthWizard packaged scientific runtime is missing at {path:?}: {error}"),
        )
    })?;
    if !metadata.is_file() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DepthWizard packaged scientific runtime is not a file: {path:?}"),
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("DepthWizard packaged scientific runtime is not executable: {path:?}"),
            )));
        }
    }
    Ok(path)
}

fn forward_output<R: Read + Send + 'static>(reader: R, label: &'static str) {
    thread::spawn(move || {
        let buffered = BufReader::new(reader);
        for line in buffered.lines() {
            match line {
                Ok(message) => eprintln!("[{label}] {message}"),
                Err(error) => {
                    eprintln!("[{label}] output read error: {error}");
                    break;
                }
            }
        }
    });
}

fn launch_sidecar(
    app: &tauri::App,
    port: u16,
    token: &str,
) -> Result<Child, Box<dyn std::error::Error>> {
    let executable = sidecar_executable(app)?;
    let mut command = Command::new(&executable);
    command
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--log-level")
        .arg("warning")
        .env("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
        .env("DEPTHWIZARD_SESSION_TOKEN", token)
        .env("DEPTHWIZARD_OFFLINE_CORE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PROJ_NETWORK", "OFF")
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to launch DepthWizard scientific runtime {executable:?}: {error}"),
        )
    })?;
    if let Some(stdout) = child.stdout.take() {
        forward_output(stdout, "depthwizard-core stdout");
    }
    if let Some(stderr) = child.stderr.take() {
        forward_output(stderr, "depthwizard-core stderr");
    }
    Ok(child)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![runtime_config])
        .setup(|app| {
            let port = reserve_loopback_port()?;
            let token = generate_session_token();
            let mut child = launch_sidecar(app, port, &token)?;
            let sidecar_pid = child.id();

            if let Err(error) = wait_for_health(port, SIDECAR_READY_TIMEOUT) {
                terminate_child(&mut child);
                return Err(Box::new(error));
            }

            let runtime = RuntimeConfig {
                api_base: format!("http://127.0.0.1:{port}"),
                session_token: token,
                sidecar_pid,
                offline_core: true,
            };
            if let Err(error) = write_acceptance_boot_report(&runtime) {
                terminate_child(&mut child);
                return Err(Box::new(error));
            }
            let acceptance_control_path = match write_acceptance_control(&runtime) {
                Ok(path) => path,
                Err(error) => {
                    terminate_child(&mut child);
                    return Err(Box::new(error));
                }
            };
            if let Err(error) = schedule_acceptance_auto_exit(app.handle().clone()) {
                if let Some(path) = acceptance_control_path.as_ref() {
                    let _ = fs::remove_file(path);
                }
                terminate_child(&mut child);
                return Err(Box::new(error));
            }
            if let Err(error) = schedule_acceptance_exit_signal(app.handle().clone()) {
                if let Some(path) = acceptance_control_path.as_ref() {
                    let _ = fs::remove_file(path);
                }
                terminate_child(&mut child);
                return Err(Box::new(error));
            }

            app.manage(SidecarState {
                child: Mutex::new(Some(child)),
                runtime,
                acceptance_control_path,
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building DepthWizard desktop application");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
            if let Some(state) = app_handle.try_state::<SidecarState>() {
                state.shutdown();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{generate_session_token, reserve_loopback_port};
    use std::net::TcpListener;

    #[test]
    fn session_token_has_256_bits_of_hex_material() {
        let token = generate_session_token();
        assert_eq!(token.len(), 64);
        assert!(token.chars().all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn reserved_port_is_loopback_bindable_after_release() {
        let port = reserve_loopback_port().expect("reserve loopback port");
        let listener = TcpListener::bind(("127.0.0.1", port)).expect("rebind loopback port");
        assert_eq!(listener.local_addr().expect("local address").port(), port);
    }
}
