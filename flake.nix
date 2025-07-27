{
  description = "Anthropic compatibility layer for openai-compatible endpoints";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
  outputs =
    inputs@{
      flake-parts,
      fenix,
      crane,
      devshell,
      treefmt-nix,
      nixpkgs,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      imports = [
        devshell.flakeModule
        treefmt-nix.flakeModule
      ];
      perSystem =
        {
          system,
          config,
          lib,
          ...
        }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ fenix.overlays.default ];
          };

          toolchain =
            with fenix.packages.${system};
            combine [
              latest.rustc
              latest.cargo
              latest.clippy
              latest.rust-analysis
              latest.rust-src
              latest.rustfmt
            ];

          craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;

          build-deps = with pkgs; [
            rustls-libssl
          ];

          src = lib.fileset.toSource {
            root = ./.;
            fileset = craneLib.fileset.commonCargoSources ./.;
          };

          build-attrs = {
            inherit src;
            buildInputs = build-deps;
            nativeBuildInputs = [ pkgs.pkg-config ];
            strictDeps = true;
            INSTA_UPDATE = "always";
          };

          ant-compat = craneLib.buildPackage build-attrs;

          packages = {
            default = ant-compat;
            ant-compat-container = pkgs.dockerTools.buildImage {
              name = "ant-compat";
              tag = "latest";
              copyToRoot = [ ant-compat ];
              config = {
                Cmd = [ "/bin/ant-compat" ];
              };
            };
          };

          deps-only = craneLib.buildDepsOnly build-attrs;

          checks = {
            clippy = craneLib.cargoClippy {
              inherit src;
              cargoArtifacts = deps-only;
              cargoClippyExtraArgs = "--all-features -- --deny warnings";
            };
            rust-fmt = craneLib.cargoFmt { inherit src; };
          };

        in
        {
          inherit checks packages;
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt-rfc-style.enable = true;
              statix.enable = true;
              rustfmt.enable = true;
            };
          };

          devshells.default = {
            packages =
              with pkgs;
              [
                config.treefmt.build.wrapper
                nixfmt-rfc-style
                statix
                toolchain
                cargo-insta
                gcc
              ]
              ++ build-deps;
          };
        };
    };
}
