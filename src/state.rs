use reqwest::Client;

#[derive(Clone)]
pub struct AppState {
    pub openai_base_url: String,
    pub default_haiku_model: String,
    pub http_client: Client,
    pub idle_connection_timeout: u64,
}
