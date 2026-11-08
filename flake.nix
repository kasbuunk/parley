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
        in
        {
          inherit parley transcribe;
          default =
            if pkgs.stdenv.hostPlatform.isLinux then
              pkgs.symlinkJoin {
                name = "parley-0.1.0";
                paths = [
                  parley
                  transcribe
                ];
              }
            else
              parley;
        }
      );
    };
}
