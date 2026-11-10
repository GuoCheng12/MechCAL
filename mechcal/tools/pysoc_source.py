"""Pinned numerical implementation; third-party files are not bundled."""

import hashlib
from pathlib import Path

SOURCE_COMMIT = "1a520c682f4851a8ec6e551a04ec5cc89ebc2894"
SOURCE_HASHES = {
    "pysoc/__init__.py": "da60e893568d36af9dc6210fb444ca773eb725d474f36c2226b41b2c342571f3",
    "pysoc/data/bin/molsoc": "61990c1ed95f530d15e3d31b1de1a4c3dae032013e9180f043103d0c47b31e96",
    "pysoc/data/bin/soc_td": "3f3ac36ab05cbabace611efb374bf0aa1c4b4981ae75f18f414a7d952ffb6ce8",
    "pysoc/data/parameters/mio-1-1-fit/C.basis": "26bd71a1a99b72ea6577956f76db0938bdede3038ca75b1c21fecbaa2a60bb84",
    "pysoc/data/parameters/mio-1-1-fit/H.basis": "570cb57fff7a524ddc848201cef45e0b8d0249d5f42c4e07a82b39aff1a4f053",
    "pysoc/data/parameters/mio-1-1-fit/N.basis": "3a106bf6b7151a0ab02ab3aa6ad8abd073639a6a2f13fd83dc7784ff22632e20",
    "pysoc/data/parameters/mio-1-1-fit/O.basis": "ec5a37751617e7be212bf67dcb080f4aaf32e329afd4170932a30638ff17c25e",
    "pysoc/io/SOC.py": "fbe15c4f0a7e82c23c0b66030033af8fa470883976b9404adca4132fea8ff56d",
    "pysoc/io/__init__.py": "9626c032720fe95fc6205ad6b30a436a9314295a53af2203a877e191d64d44cb",
    "pysoc/io/dftb_plus.py": "db4b0fde0e49f6f123d9f91750e13accf790d67b49d5b00be85b3983e1fc5b44",
    "pysoc/io/file.py": "1c35ba6c1305236666ace37642c55c6afbd782c9a05e7c8ff5ad2687e6bb554e",
    "pysoc/io/gaussian.py": "0526f0d2305c1a3e53786825c8c2bca7719d5f4619b75a709815e4dcd75be79b",
    "pysoc/io/molsoc.py": "14f8e6d9255acb027b35be79949b9175337c5c22a2bf8f3ac4706a745b913847",
    "pysoc/io/soc_td.py": "7125461ba9fd8a17bdfd3339f30f38edcad318f3067e4fe3f601f0ad70193ded"
}


def source_reason(source):
    if not source:
        return "PySOC source directory is not configured."
    root = Path(source)
    try:
        for name, expected in SOURCE_HASHES.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                return f"PySOC validated source differs: {name}."
    except OSError as exc:
        return f"PySOC source is unavailable: {exc}."
    return None
