"""Aquisicao oficial, auditoria e amostragem deterministica do CNR-EXT."""

import hashlib
import re
import urllib.request
import zipfile
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from .configuracao import proveniencia, salvar_json, sha256_file

URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip"
PARTICOES = ("treino", "validacao", "teste")


@lru_cache(maxsize=12000)
def caminho_seguro(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Caminho fora da raiz de dados: {relative}")
    return path


def baixar(config):
    root = Path(config["data_root"])
    archive = root.parent / "CNR-EXT-Patches-150x150.zip"
    root.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            caminho_seguro(root, member.filename)
        print(f"Extraindo {len(zipped.infolist())} entradas...", flush=True)
        zipped.extractall(root)
    salvar_json(
        root.parent / "aquisicao.json",
        {
            "url": URL,
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "licenca": "ODbL-1.0",
            "origem": "http://cnrpark.it/",
        },
    )


def ler_listas(config):
    records = []
    root = Path(config["data_root"])
    pattern = re.compile(r"[SOR]_(\d{4}-\d{2}-\d{2})_(\d{2}\.\d{2})_C(\d{2})_(\d+)\.jpg$")
    for split in PARTICOES:
        for line in caminho_seguro(root, config["split_files"][split]).read_text().splitlines():
            if not line.strip():
                continue
            relative, label = line.rsplit(maxsplit=1)
            relative = relative.replace("\\", "/")
            if not relative.startswith("PATCHES/"):
                relative = "PATCHES/" + relative
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or ":" in relative:
                raise ValueError(f"Caminho invalido na lista: {relative}")
            match = pattern.search(relative)
            if not match or label not in ("0", "1"):
                raise ValueError(f"Linha invalida: {line}")
            date, time, camera, slot = match.groups()
            weather = relative.split("/")[1]
            if weather not in ("SUNNY", "OVERCAST", "RAINY"):
                raise ValueError(f"Clima desconhecido: {weather}")
            records.append(
                {
                    "id": hashlib.sha256(relative.encode()).hexdigest()[:20],
                    "path": relative,
                    "label": int(label),
                    "camera": int(camera),
                    "date": date,
                    "time": time,
                    "slot": int(slot),
                    "weather": weather,
                    "capture": f"{date}_{time}_C{camera}",
                    "split": split,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Listas oficiais vazias")
    return frame


def amostrar(frame, size, seed):
    """Cotas pelo metodo dos maiores restos; nao depende da ordem do CSV."""
    if size <= 0:
        raise ValueError("Tamanho de amostra deve ser positivo")
    frame = frame.sort_values("path")
    size = min(size, len(frame))
    groups = list(frame.groupby(["label", "camera", "weather"], sort=True))
    exact = np.array([len(group) * size / len(frame) for _, group in groups])
    quotas = np.floor(exact).astype(int)
    remainder = size - quotas.sum()
    order = np.argsort(-(exact - quotas), kind="stable")
    quotas[order[:remainder]] += 1
    samples = [group.sample(n=int(quota), random_state=seed) for (_, group), quota in zip(groups, quotas)]
    return pd.concat(samples).sort_values("path").reset_index(drop=True)


def sobreposicoes(frame):
    result = {}
    for field in ("camera", "date", "capture", "slot"):
        sets = {s: set(frame.loc[frame.split == s, field]) for s in PARTICOES}
        result[field] = {
            f"{a}/{b}": len(sets[a] & sets[b])
            for a, b in (("treino", "validacao"), ("treino", "teste"), ("validacao", "teste"))
        }
    return result


def preparar(config):
    target = Path(config["manifest"])
    if target.exists():
        raise FileExistsError(
            "Manifesto ja congelado. Use outra configuracao e outro caminho para novo estudo."
        )
    frame = ler_listas(config)
    count_original = frame.groupby("split").size().to_dict()
    audit_rows = []
    archive = Path(config["data_root"]).parent / "CNR-EXT-Patches-150x150.zip"
    # Auditar diretamente os bytes oficiais; a extracao de features reconfere os arquivos locais.
    context = zipfile.ZipFile(archive) if archive.exists() else nullcontext(None)
    with context as zipped:
        for i, row in enumerate(frame.itertuples(index=False)):
            try:
                content = (
                    zipped.read(row.path)
                    if zipped
                    else caminho_seguro(config["data_root"], row.path).read_bytes()
                )
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Imagem ilegivel")
                audit_rows.append(
                    {
                        "path": row.path,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "pixel_sha256": hashlib.sha256(
                            str(image.shape).encode() + image.tobytes()
                        ).hexdigest(),
                        "height": image.shape[0],
                        "width": image.shape[1],
                        "error": "",
                    }
                )
            except (OSError, ValueError, KeyError) as exc:
                audit_rows.append({"path": row.path, "error": str(exc)})
            if (i + 1) % 20000 == 0:
                print(f"Auditadas {i + 1}/{len(frame)} imagens", flush=True)
    audited = pd.DataFrame(audit_rows).drop_duplicates("path")
    frame = frame.merge(audited, on="path", validate="many_to_one")
    bad = frame.loc[frame.error != ""].copy()
    good = frame.loc[frame.error == ""].copy()
    conflicts = good.groupby("pixel_sha256").label.nunique()
    conflict_hashes = conflicts[conflicts > 1].index
    conflicting = good[good.pixel_sha256.isin(conflict_hashes)]
    good = good[~good.pixel_sha256.isin(conflict_hashes)].copy()
    good["priority"] = good.split.map({s: i for i, s in enumerate(PARTICOES)})
    good = good.sort_values(["priority", "path"])
    duplicates = good[good.duplicated("pixel_sha256", keep="first")]
    good = good.drop_duplicates("pixel_sha256", keep="first").drop(columns=["priority", "error"])
    selected = pd.concat(
        [amostrar(good[good.split == s], config["sample_sizes"][s], config["seed"]) for s in PARTICOES],
        ignore_index=True,
    )
    validar_manifesto(selected)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(target, index=False, lineterminator="\n")
    metrics = Path(config["results_dir"]) / "metricas"
    metrics.mkdir(parents=True, exist_ok=True)
    for name, excluded in (("ilegíveis", bad), ("conflitos", conflicting), ("duplicatas", duplicates)):
        excluded[["path", "split", "label"]].to_csv(
            metrics / f"auditoria_{name.replace('í', 'i')}.csv", index=False
        )
    audit = {
        "original": count_original,
        "unreadable": len(bad),
        "conflicting_labels": len(conflicting),
        "duplicates_removed": len(duplicates),
        "eligible": good.groupby("split").size().to_dict(),
        "selected": selected.groupby("split").size().to_dict(),
        "overlap_original": sobreposicoes(frame),
        "overlap_selected": sobreposicoes(selected),
        "manifest_sha256": sha256_file(target),
        "provenance": proveniencia(config),
    }
    acquisition = Path(config["data_root"]).parent / "aquisicao.json"
    if acquisition.exists():
        import json

        audit["acquisition"] = json.loads(acquisition.read_text())
    salvar_json(metrics / "auditoria.json", audit)
    print(f"Manifesto congelado: {len(selected)} imagens", flush=True)


def validar_manifesto(frame):
    required = {"id", "path", "label", "camera", "date", "weather", "capture", "split", "pixel_sha256"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Manifesto vazio ou incompleto")
    if frame.id.duplicated().any() or frame.path.duplicated().any() or frame.pixel_sha256.duplicated().any():
        raise ValueError("Manifesto contem duplicatas")
    if not set(frame.label).issubset({0, 1}) or set(frame.split) != set(PARTICOES):
        raise ValueError("Classes ou particoes invalidas")
    for split in ("treino", "validacao", "teste"):
        if set(frame.loc[frame.split == split, "label"]) != {0, 1}:
            raise ValueError(f"A particao {split} precisa conter ambas as classes")


def carregar_manifesto(config):
    frame = pd.read_csv(config["manifest"], dtype={"id": str, "sha256": str, "pixel_sha256": str})
    validar_manifesto(frame)
    return frame
