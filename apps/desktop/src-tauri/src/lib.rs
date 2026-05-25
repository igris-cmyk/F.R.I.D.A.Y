// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

use tauri::{Manager, Emitter, WindowEvent, State};
use tauri_plugin_global_shortcut::{Code, Modifiers, ShortcutState};
use active_win_pos_rs::get_active_window;
use serde_json::json;
use tauri::tray::TrayIconBuilder;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use std::process::{Command, Child};
use std::sync::Mutex;
use std::path::PathBuf;

struct CoreState {
    process: Mutex<Option<Child>>,
}

fn spawn_core() -> Option<Child> {
    println!("[SUPERVISION] Spawning Python Core...");
    let current_dir = std::env::current_dir().ok()?;
    let repo_root = current_dir.join("../../").canonicalize().ok()?;
    let python_path: PathBuf = repo_root.join("core/.venv/bin/python");

    Command::new(python_path)
        .args(["-m", "core.main"])
        .current_dir(repo_root)
        .spawn()
        .map_err(|e| println!("[SUPERVISION] Failed to spawn core: {}", e))
        .ok()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(CoreState {
            process: Mutex::new(None),
        })
        .setup(|app| {
            // Spawn the python core immediately on boot
            let child = spawn_core();
            if let Some(child_proc) = child {
                let state: State<CoreState> = app.state();
                *state.process.lock().unwrap() = Some(child_proc);
            }

            // Setup Tray Menu
            let open_i = MenuItem::with_id(app, "open", "Open FRIDAY", true, None::<&str>)?;
            let restart_i = MenuItem::with_id(app, "restart", "Restart Core", true, None::<&str>)?;
            let health_i = MenuItem::with_id(app, "health", "View Health", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "Shutdown FRIDAY", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_i, &restart_i, &health_i, &PredefinedMenuItem::separator(app)?, &quit_i])?;
            
            TrayIconBuilder::new()
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => {
                        println!("Daemon Shutdown Request Received");
                        let state: State<CoreState> = app.state();
                        if let Some(mut child) = state.process.lock().unwrap().take() {
                            println!("[SUPERVISION] Killing Python Core before exit...");
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                        app.exit(0);
                    }
                    "open" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "restart" => {
                        println!("Core Restart Hook triggered");
                        let state: State<CoreState> = app.state();
                        let mut proc_guard = state.process.lock().unwrap();
                        if let Some(mut child) = proc_guard.take() {
                            println!("[SUPERVISION] Terminating old Python Core...");
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                        *proc_guard = spawn_core();
                    }
                    "health" => {
                        println!("View Health Hook triggered");
                        // Future implementation to open a health window
                    }
                    _ => {
                        println!("menu item {:?} not handled", event.id);
                    }
                })
                .build(app)?;

            Ok(())
        })
        .on_window_event(|window, event| match event {
            // Decouple window close from application termination
            WindowEvent::CloseRequested { api, .. } => {
                api.prevent_close();
                let _ = window.hide();
            }
            _ => {}
        })
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_shortcut("super+space")
                .unwrap()
                .with_handler(|app, shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        if shortcut.matches(Modifiers::SUPER, Code::Space) {
                            if let Some(window) = app.get_webview_window("main") {
                                if window.is_visible().unwrap_or(false) {
                                    let _ = window.hide();
                                } else {
                                    // 1. Context Pipeline: Capture Ambient OS State
                                    let mut active_app = String::from("unknown");
                                    let mut window_title = String::from("unknown");
                                    
                                    // Bounding the context extraction to preserve sub-ms latency
                                    if let Ok(active_window) = get_active_window() {
                                        active_app = active_window.app_name;
                                        window_title = active_window.title;
                                    }
                                    
                                    let context_payload = json!({
                                        "active_app": active_app,
                                        "window_title": window_title
                                    });
                                    
                                    // Emit to frontend before displaying
                                    let _ = window.emit("ambient-context", context_payload);
                                    
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
                            }
                        }
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
