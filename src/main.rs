use anyhow::Result;
use axum::{Router, routing::post};
use std::env;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::info;

mod adapters;
mod conversion;
mod http;
mod logging;
mod models;

use http::handle_messages;

#[derive(Clone)]
pub struct AppState {
    pub openai_base_url: String,
    pub default_haiku_model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::init();

    let openai_base_url =
        env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:10152/v1".to_string());

    let default_haiku_model =
        env::var("HAIKU_MODEL").unwrap_or_else(|_| "openai/gpt-4.1-mini".to_string());

    let state = AppState {
        openai_base_url,
        default_haiku_model,
    };

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:33332").await?;
    info!("Server running on 0.0.0.0:33332");

    axum::serve(listener, app).await?;

    Ok(())
}
