use std::env;
use std::process::Command;

fn git_sha() -> String {
    if let Ok(value) = env::var("DEPTHWIZARD_BUILD_GIT_SHA") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }

    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn main() {
    let sha = git_sha();
    println!("cargo:rustc-env=DEPTHWIZARD_BUILD_GIT_SHA={sha}");
    println!("cargo:rerun-if-env-changed=DEPTHWIZARD_BUILD_GIT_SHA");
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/index");
    tauri_build::build()
}
