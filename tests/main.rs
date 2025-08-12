mod adapters;
mod conversion;
mod directives;

pub mod helpers {
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
}
