//! VisionGuard360 operator console.
//!
//! Server-rendered with Topcoat; no client framework. Pages:
//! `/` overview · `/cameras/{id}` camera detail + camera guard · `/rules` MCC gates + ledger ·
//! `/twin` site plan · `/training` models · `/settings` cameras.
//! API: `/v1/models`, `/v1/chat/completions` (OpenAI-compatible), `/v1/rules`, `/v1/ledger`, `/v1/cameras`,
//! `/api/twin` (GET / POST from the Rig Twin editor), `/api/rule-studio`, `/api/evidence`,
//! live: `/api/live`, `/api/live/events` (SSE), `/cameras/{id}/live.jpg`, `/api/evidence/{incident}.jpg`, `POST /api/signals`.
mod api;
mod calib_api;
mod env;
mod live;
mod pages;
mod state;
mod ui;

use topcoat::router::{Router, RouterBuilderDirectoryExt};

pub fn router(root: &str) -> Router {
    Router::builder()
        .with(state::Site::load(root))
        .layout(ui::shell)
        .page(pages::overview)
        .page(pages::camera)
        .page(pages::live)
        .page(pages::rules)
        .page(pages::twin)
        .page(pages::training)
        .page(pages::settings)
        .route(api::models)
        .route(api::chat)
        .route(api::rules)
        .route(api::ledger)
        .route(api::cameras)
        .route(api::twin_get)
        .route(api::twin_apply)
        .route(api::rule_studio)
        .route(api::evidence)
        .route(api::live_state)
        .route(api::live_events)
        .route(api::live_jpeg)
        .route(api::live_evidence_jpeg)
        .route(api::signals)
        .route(api::calibrate)
        .public_dir(format!("{root}/public"))
        .build()
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let root = std::env::var("VG360_ROOT").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string());
    eprintln!("VisionGuard360 console on http://{}:{}", std::env::var("HOST").unwrap_or("127.0.0.1".into()),
        std::env::var("PORT").unwrap_or("3000".into()));
    topcoat::start(router(&root)).await
}
