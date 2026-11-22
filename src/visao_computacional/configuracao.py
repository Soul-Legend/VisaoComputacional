"""Configuracao e registro de proveniencia dos experimentos."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


def carregar_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    for key in ("data_root", "manifest", "processed_dir", "models_dir", "results_dir", "report_dir"):
        config[key] = str((root / config[key]).resolve())
    config["config_path"] = str(path)
    config["project_root"] = str(root)
    return config


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def salvar_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def proveniencia(config):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config["project_root"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=config["project_root"],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "indisponivel", None
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "opencv-python-headless",
        "scikit-image",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
    ):
        versions[name] = importlib.metadata.version(name)
    # Caminhos publicados relativos ao projeto; nao expor diretorios pessoais.
    public_config = dict(config)
    for key in (
        "data_root",
        "manifest",
        "processed_dir",
        "models_dir",
        "results_dir",
        "report_dir",
        "config_path",
    ):
        public_config[key] = (
            Path(config[key]).relative_to(config["project_root"]).as_posix()
            if Path(config[key]).is_relative_to(config["project_root"])
            else Path(config[key]).name
        )
    public_config.pop("project_root")
    return {
        "utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "tracked_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
        "config": public_config,
        "manifest_sha256": sha256_file(config["manifest"]) if Path(config["manifest"]).exists() else None,
    }
