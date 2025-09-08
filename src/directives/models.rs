use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Condition {
    ModelContains(String),
}

#[derive(Debug, Deserialize)]
pub struct Rule {
    #[serde(rename = "if")]
    pub if_clause: Condition,
    pub apply: Settings,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct ResponsesSettings {
    #[serde(default)]
    pub enable: Option<bool>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning_summary: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct Settings {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub enable_meowsings: Option<bool>,
    #[serde(default)]
    pub responses: Option<ResponsesSettings>,
}

#[derive(Debug, Deserialize)]
pub struct ProxyDirective {
    #[serde(default)]
    pub global: Option<Settings>,
    #[serde(default)]
    pub rules: Vec<Rule>,
}
