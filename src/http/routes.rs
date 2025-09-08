use crate::{
    AppState,
    adapters::RequestAdapter,
    conversion::{convert_claude_to_openai, convert_openai_to_claude},
    directives::processor::DirectiveProcessor,
    error::AppError,
    models::{claude::ClaudeMessagesRequest, openai::OpenAIRequest},
};
use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde_json::Value;
use tracing::{debug, info};

struct RequestContext {
    openai_request: OpenAIRequest,
    api_key: String,
    target_model: String,
    adapter: RequestAdapter,
    claude_request: ClaudeMessagesRequest,
    is_streaming: bool,
}

async fn handle_non_streaming_response(
    response: reqwest::Response,
    target_model: String,
    adapter: &RequestAdapter,
    request: &ClaudeMessagesRequest,
) -> Result<Response, AppError> {
    info!("Handling as a non-streaming request");
    let response_json: Value = response.json().await?;

    let normalized = adapter.normalize_non_stream_json(response_json, request);
    let adapted_json = adapter.adapt_non_stream_response(normalized, request);

    let claude_response = convert_openai_to_claude(&adapted_json, &target_model);
    info!("Sending back converted Claude response");
    debug!("Claude response: {claude_response:?}");
    Ok(Json(claude_response).into_response())
}

fn handle_streaming_response(
    response: reqwest::Response,
    target_model: &str,
    adapter: &RequestAdapter,
    request: &ClaudeMessagesRequest,
    state: &AppState,
) -> Result<Response, AppError> {
    info!("Handling as a streaming request");
    let stream = adapter.build_anthropic_sse_stream(response, target_model, request, state);
    let body = Body::from_stream(stream);
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .map_err(|e| {
            AppError::InternalServerError(format!("Failed to build streaming response: {e}"))
        })?;
    Ok(response.into_response())
}

async fn send_openai_request(
    state: &AppState,
    api_key: &str,
    openai_request: &OpenAIRequest,
    adapter: &RequestAdapter,
    claude_request: &ClaudeMessagesRequest,
) -> Result<reqwest::Response, AppError> {
    let url = format!("{}{}", state.openai_base_url, adapter.endpoint_suffix());
    let body = adapter.build_request_body(openai_request, claude_request);
    Ok(state
        .http_client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?)
}

fn prepare_request_context(
    state: &AppState,
    headers: &HeaderMap,
    mut request: ClaudeMessagesRequest,
) -> Result<RequestContext, AppError> {
    info!("Preparing request for model: {}", request.model);
    let settings = DirectiveProcessor::process(&mut request);
    let api_key = extract_api_key(headers)?;

    let target_model = if request.model.to_lowercase().contains("haiku") {
        state.default_haiku_model.clone()
    } else {
        request.model.clone()
    };
    let is_streaming = request.stream.unwrap_or(false);
    let adapter = RequestAdapter::for_model(&target_model, &settings);
    let openai_request = convert_claude_to_openai(request.clone(), &target_model, &adapter);

    Ok(RequestContext {
        openai_request,
        api_key,
        target_model,
        adapter,
        claude_request: request,
        is_streaming,
    })
}

async fn validate_upstream_response(
    response: reqwest::Response,
) -> Result<reqwest::Response, AppError> {
    if response.status().is_success() {
        Ok(response)
    } else {
        let status = response.status();
        let error_text = response.text().await.map_err(|e| {
            AppError::InternalServerError(format!("Failed to read upstream error body: {e}"))
        })?;
        Err(AppError::UpstreamError(status, error_text))
    }
}

pub async fn handle_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    JsonExtractor(request): JsonExtractor<ClaudeMessagesRequest>,
) -> Result<Response, AppError> {
    let context = prepare_request_context(&state, &headers, request)?;
    let response = send_openai_request(
        &state,
        &context.api_key,
        &context.openai_request,
        &context.adapter,
        &context.claude_request,
    )
    .await?;
    let response = validate_upstream_response(response).await?;

    if context.is_streaming {
        handle_streaming_response(
            response,
            &context.target_model,
            &context.adapter,
            &context.claude_request,
            &state,
        )
    } else {
        handle_non_streaming_response(
            response,
            context.target_model,
            &context.adapter,
            &context.claude_request,
        )
        .await
    }
}

fn extract_api_key(headers: &HeaderMap) -> Result<String, AppError> {
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(std::string::ToString::to_string)
        .ok_or(AppError::MissingApiKey)
}
