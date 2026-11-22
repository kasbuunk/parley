{
  description = "parley: a Matrix bot that lets people talk to home-lab agents";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAll (
        pkgs:
        let
          parley = pkgs.rustPlatform.buildRustPackage {
            pname = "parley";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;
            nativeBuildInputs = [ pkgs.pkg-config ];
            buildInputs = [ pkgs.sqlite ];
          };
          # Voice messages (src/voice.rs): faster-whisper on the CPU.
          python = pkgs.python3.withPackages (p: [ p.faster-whisper ]);
          transcribe = pkgs.writeShellScriptBin "parley-transcribe" ''
            exec ${python}/bin/python3 -I ${./transcribe/parley-transcribe.py} "$@"
          '';
          # The `speak` tool (src/speech.rs): Piper on the CPU, without the
          # extras that would pull in 4 GB of torch. Any program with the
          # same contract can replace it through PARLEY_SPEAK.
          piper = pkgs.piper-tts.override {
            withAlignment = false;
            withHTTP = false;
            withJapanese = false;
            withTrain = false;
          };
          voiceUrl =
            ext:
            "https://huggingface.co/rhasspy/piper-voices/resolve/e9c8a39444359388adaa94255becb37a5a2b8e51/nl/nl_NL/alex/medium/nl_NL-alex-medium.${ext}";
          # Piper looks for the config next to the model, under this name.
          voice = pkgs.linkFarm "piper-voice-nl_NL-alex-medium" {
            "voice.onnx" = pkgs.fetchurl {
              url = voiceUrl "onnx";
              hash = "sha256-dY+mTsRhVFly/D5QlRG0iEnG8/XIIa1Vxx2XQdQ1K9w=";
            };
            "voice.onnx.json" = pkgs.fetchurl {
              url = voiceUrl "onnx.json";
              hash = "sha256-AesI+Al1qkZDwmer7YHmKJRjutd757WpbRdzOFWGUZs=";
            };
          };
          speak = pkgs.writeShellScriptBin "parley-speak" ''
            set -euo pipefail
            ${piper}/bin/piper -m ${voice}/voice.onnx --output-raw \
              | ${pkgs.ffmpeg-headless}/bin/ffmpeg -nostdin -v error -y \
                  -f s16le -ar 22050 -ac 1 -i pipe:0 \
                  -c:a libopus -b:a 32k -application voip -f ogg "$1"
          '';
        in
        {
          inherit parley transcribe speak;
          default =
            if pkgs.stdenv.hostPlatform.isLinux then
              pkgs.symlinkJoin {
                name = "parley-0.1.0";
                paths = [
                  parley
                  transcribe
                  speak
                ];
              }
            else
              parley;
        }
      );
    };
}
