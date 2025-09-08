use axum::{
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde_json::json;
use thiserror::Error;
use tracing::error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Missing x-api-key header")]
    MissingApiKey,
    #[error("Upstream error: {0} - {1}")]
    UpstreamError(StatusCode, String),
    #[error("Internal Server Error: {0}")]
    InternalServerError(String),
    #[error("Stream Error: {0}")]
    StreamError(String),
    #[error("Request failed: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("JSON serialization/deserialization failed: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("SSE codec error: {0}")]
    SseCodec(#[from] tokio_sse_codec::SseDecodeError),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, error_message) = match self {
            AppError::MissingApiKey => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "Missing x-api-key header".to_string(),
            ),
            AppError::UpstreamError(status, message) => (status, "api_error", message),
            AppError::Reqwest(err) => {
                error!("Request Error: {err}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("Request failed: {err}"),
                )
            }
            AppError::SerdeJson(err) => {
                error!("Serde Error: {err}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("JSON serialization/deserialization failed: {err}"),
                )
            }
            AppError::SseCodec(err) => {
                error!("SSE Codec Error: {err}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("SSE codec error: {err}"),
                )
            }
            AppError::InternalServerError(message) => {
                error!("Internal Server Error: {message}");
                (StatusCode::INTERNAL_SERVER_ERROR, "api_error", message)
            }
            AppError::StreamError(message) => {
                error!("Stream Error: {message}");
                (StatusCode::INTERNAL_SERVER_ERROR, "api_error", message)
            }
        };

        let body = Json(json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": error_message
            }
        }));

        (status, body).into_response()
    }
}

impl From<AppError> for std::io::Error {
    fn from(error: AppError) -> Self {
        std::io::Error::other(error)
    }
}
