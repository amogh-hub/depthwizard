use rand::rngs::OsRng;
use rand::RngCore;
use serde::Serialize;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const SIDECAR_NAME: &str = "depthwizard-core";
const SIDECAR_READY_TIMEOUT: Duration = Duration::from_secs(30);

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
}

struct SidecarState {
    child: Mutex<Option<CommandChild>>,
    runtime: RuntimeConfig,
}

impl SidecarState {
    fn shutdown(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(child) = guard.take() {
                let _ = child.kill();
            }
        }
    }
}

#[tauri::command]
fn runtime_config(state: State<'_, SidecarState>) -> RuntimeConfig {
    state.runtime.clone()
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
        std::thread::sleep(Duration::from_millis(100));
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
        schema_version: 1,
        status: "PASS_TAURI_SIDECAR_BOOT",
        api_base: &runtime.api_base,
        sidecar_pid: runtime.sidecar_pid,
        offline_core: runtime.offline_core,
        session_token_bits: (runtime.session_token.len() * 4) as u16,
        session_token_exported: false,
        strict_python_egress_guard: runtime.offline_core,
    };
    let payload = serde_json::to_string_pretty(&report).map_err(io::Error::other)?;
    fs::write(path, format!("{payload}\n"))
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
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(millis));
        app_handle.exit(0);
    });
    Ok(())
}

fn launch_sidecar(
    app: &tauri::App,
    port: u16,
    token: &str,
) -> Result<CommandChild, Box<dyn std::error::Error>> {
    let args = vec![
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--log-level".to_string(),
        "warning".to_string(),
    ];
    let command = app
        .shell()
        .sidecar(SIDECAR_NAME)?
        .args(args)
        .env("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
        .env("DEPTHWIZARD_SESSION_TOKEN", token)
        .env("DEPTHWIZARD_OFFLINE_CORE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost");

    let (mut events, child) = command.spawn()?;
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    eprintln!(
                        "[depthwizard-core stdout] {}",
                        String::from_utf8_lossy(&bytes)
                    );
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!(
                        "[depthwizard-core stderr] {}",
                        String::from_utf8_lossy(&bytes)
                    );
                }
                CommandEvent::Error(error) => {
                    eprintln!("[depthwizard-core error] {error}");
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("[depthwizard-core terminated] {payload:?}");
                }
                _ => {}
            }
        }
    });
    Ok(child)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![runtime_config])
        .setup(|app| {
            let port = reserve_loopback_port()?;
            let token = generate_session_token();
            let child = launch_sidecar(app, port, &token)?;
            let sidecar_pid = child.pid();

            if let Err(error) = wait_for_health(port, SIDECAR_READY_TIMEOUT) {
                let _ = child.kill();
                return Err(Box::new(error));
            }

            let runtime = RuntimeConfig {
                api_base: format!("http://127.0.0.1:{port}"),
                session_token: token,
                sidecar_pid,
                offline_core: true,
            };
            write_acceptance_boot_report(&runtime)?;
            app.manage(SidecarState {
                child: Mutex::new(Some(child)),
                runtime,
            });
            schedule_acceptance_auto_exit(app.handle().clone())?;
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
