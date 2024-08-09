let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.glob2
      python-pkgs.numpy
      python-pkgs.opencv4
    ]))
  ];
}

