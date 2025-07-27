use tracing_subscriber::{EnvFilter, FmtSubscriber};

pub fn init() {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();
    if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("setting default subscriber failed: {e}");
    }
}
