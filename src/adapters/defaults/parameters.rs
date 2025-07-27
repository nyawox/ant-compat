use crate::{adapters::traits::Adapter, conversion::request::Request};

pub struct KimiMaxTokensAdapter;

// workaround for groq
impl Adapter for KimiMaxTokensAdapter {
    fn adapt_max_tokens(&self, _max_tokens: u32, _request: &Request) -> u32 {
        16384
    }
}
