use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde_json::{Value, json};
use tracing::{debug, error, info};

use crate::AppState;
use crate::adapters::RequestAdapter;
use crate::conversion::convert_claude_to_openai;
use crate::conversion::convert_openai_stream_to_anthropic;
use crate::conversion::convert_openai_to_claude;
use crate::models::claude::ClaudeMessagesRequest;

pub async fn handle_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    JsonExtractor(request): JsonExtractor<ClaudeMessagesRequest>,
) -> impl IntoResponse {
    info!("Received request for model: {}", request.model);
    // debug!("Full request: {request:?}");

    let api_key = match extract_api_key(&headers) {
        Ok(key) => key,
        Err(err) => return err.into_response(),
    };

    let target_model = if request.model.to_lowercase().contains("haiku") {
        state.default_haiku_model.clone()
    } else {
        request.model.clone()
    };

    let is_streaming = request.stream.unwrap_or(false);
    let adapter = RequestAdapter::for_model(&target_model);
    let openai_request = convert_claude_to_openai(request, &target_model, &adapter);
    // debug!("Converted OpenAI request: {openai_request:?}");

    let client = reqwest::Client::new();
    let response = match client
        .post(format!("{}/chat/completions", state.openai_base_url))
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&openai_request)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            error!("Request failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": format!("Request failed: {e}")
                })),
            )
                .into_response();
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(json!({
                "error": error_text
            })),
        )
            .into_response();
    }

    if is_streaming {
        info!("Handling as a streaming request");
        let stream = convert_openai_stream_to_anthropic(response, target_model);
        let body = Body::from_stream(stream);
        return match Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(body)
        {
            Ok(response) => response.into_response(),
            Err(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Failed to build streaming response"
                })),
            )
                .into_response(),
        };
    }

    info!("Handling as a non-streaming request");
    let openai_response: Value = match response.json().await {
        Ok(resp) => resp,
        Err(e) => {
            error!("Failed to parse response: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": format!("Failed to parse response: {e}")
                })),
            )
                .into_response();
        }
    };

    let claude_response = convert_openai_to_claude(&openai_response, &target_model);
    info!("Sending back converted Claude response");
    debug!("Claude response: {:?}", claude_response);
    Json(claude_response).into_response()
}

fn extract_api_key(headers: &HeaderMap) -> Result<String, (StatusCode, Json<Value>)> {
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(std::string::ToString::to_string)
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "Missing x-api-key header"
                })),
            )
        })
}
