#[macro_export]
macro_rules! lazy_regex {
    ($s:expr) => {
        std::sync::LazyLock::new(|| {
            regex::Regex::new($s).expect("Static regex pattern must be valid")
        })
    };
}

#[must_use]
pub fn map_budget_tokens_to_reasoning_effort(budget_tokens: u32) -> String {
    match budget_tokens {
        0..=1024 => "low".to_string(),
        1025..=4096 => "medium".to_string(),
        _ => "high".to_string(),
    }
}
#[must_use]
pub fn map_reasoning_effort_to_budget_tokens(reasoning_effort: &str) -> u32 {
    match reasoning_effort {
        "low" => 1024,
        "medium" => 4096,
        _ => 8192,
    }
}
