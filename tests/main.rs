mod adapters;
mod conversion;
mod directives;

pub mod helpers {
    use ant_compat::state::AppState;
    use reqwest::Client;
    use saphyr::{LoadableYamlNode, ScalarOwned, YamlOwned};
    use std::{fs, path::Path};

    pub fn load_system_prompt_fixture() -> String {
        let path = Path::new("tests/fixtures/system_prompt.yaml");
        fs::read_to_string(path)
            .ok()
            .and_then(|content| YamlOwned::load_from_str(&content).ok())
            .and_then(|mut docs| docs.drain(..).next())
            .and_then(|yaml| {
                if let YamlOwned::Mapping(map) = yaml {
                    map.get(&YamlOwned::Value(ScalarOwned::String("prompt".into())))
                        .and_then(|v| {
                            if let YamlOwned::Value(ScalarOwned::String(s)) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    pub fn mock_app_state() -> AppState {
        AppState {
            openai_base_url: "http://localhost:8080".to_string(),
            default_haiku_model: "test-model".to_string(),
            http_client: Client::new(),
            idle_connection_timeout: 60,
        }
    }

    use ant_compat::{
        error::AppError,
        models::{claude::AnthropicStreamEvent, openai::OpenAIStreamChunk},
    };
    use bytes::Bytes;
    use futures_util::stream::{Stream, StreamExt};

    pub async fn mock_response_from_chunks(chunks: Vec<OpenAIStreamChunk>) -> reqwest::Response {
        let sse_data: Vec<u8> = chunks
            .into_iter()
            .map(|chunk| {
                let json = serde_json::to_string(&chunk)
                    .expect("Serialization of a test data struct should not fail");
                format!("data: {json}\n\n")
            })
            .collect::<String>()
            .into_bytes();

        let body = reqwest::Body::from(sse_data);
        let response = http::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(body)
            .expect("Building a static HTTP response should not fail");

        reqwest::Response::from(response)
    }

    pub async fn collect_and_parse_stream(
        stream: impl Stream<Item = Result<Bytes, AppError>>,
    ) -> Vec<AnthropicStreamEvent> {
        let mut events = Vec::new();
        let mut stream = Box::pin(stream);

        while let Some(item) = stream.next().await {
            let bytes = item.expect("Test stream should not produce I/O errors");
            let lines = String::from_utf8_lossy(&bytes);
            for line in lines.split('\n') {
                if line.starts_with("data: ") {
                    let json_str = &line[6..];
                    if let Ok(event) = serde_json::from_str(json_str) {
                        events.push(event);
                    }
                }
            }
        }
        events
    }
}
