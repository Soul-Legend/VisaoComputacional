"""Extracao e cache de caracteristicas com identidade experimental verificavel."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

from .configuracao import salvar_json, sha256_file
from .dados import carregar_manifesto, caminho_seguro
from .preprocessamento import cinza, ler_rgb

IMPLEMENTACAO = 1
PARAMETROS = ("image_size", "hist_bins", "lbp_points", "lbp_radius", "hog_orientations",
              "hog_pixels_per_cell", "hog_cells_per_block")


def assinatura(config):
    value = {key: config[key] for key in PARAMETROS}
    value["manifest_sha256"] = sha256_file(config["manifest"])
    value["implementation"] = IMPLEMENTACAO
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def vetor(rgb, name, config):
    if name == "hsv":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histograms = []
        for channel, upper in enumerate((180, 256, 256)):
            hist = np.histogram(hsv[:, :, channel], bins=config["hist_bins"], range=(0, upper))[0]
            histograms.append(hist / hist.sum())
        result = np.concatenate(histograms)
    elif name == "lbp":
        points = config["lbp_points"]
        codes = local_binary_pattern(cinza(rgb), points, config["lbp_radius"], method="uniform")
        hist = np.histogram(codes, bins=points + 2, range=(0, points + 2))[0]
        result = hist / hist.sum()
    elif name == "hog":
        result = hog(cinza(rgb), orientations=config["hog_orientations"],
                     pixels_per_cell=tuple(config["hog_pixels_per_cell"]),
                     cells_per_block=tuple(config["hog_cells_per_block"]), block_norm="L2-Hys")
    else:
        raise ValueError(f"Descritor desconhecido: {name}")
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Vetor de caracteristicas invalido")
    return result


def extrair(config):
    frame = carregar_manifesto(config)
    destination = Path(config["processed_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    signature = assinatura(config)
    for name in config["descriptors"]:
        path = destination / f"{name}.npz"
        metadata = destination / f"{name}.json"
        if path.exists() and metadata.exists():
            carregar_cache(config, name, frame)
            print(f"Cache valido: {name}", flush=True)
            continue
        vectors, elapsed = [], {s: 0.0 for s in frame.split.unique()}
        for i, row in enumerate(frame.itertuples(index=False)):
            start = perf_counter()
            content = caminho_seguro(config["data_root"], row.path)
            if sha256_file(content) != row.sha256:
                raise ValueError(f"Imagem mudou desde a auditoria: {row.path}")
            rgb = ler_rgb(content, config["image_size"])
            vector = vetor(rgb, name, config)
            elapsed[row.split] += perf_counter() - start
            vectors.append(vector)
            if (i + 1) % 2000 == 0:
                print(f"{name}: {i + 1}/{len(frame)}", flush=True)
        matrix = np.stack(vectors)
        np.savez_compressed(path, X=matrix, ids=frame.id.to_numpy(dtype=str), signature=signature)
        salvar_json(metadata, {"signature": signature, "npz_sha256": sha256_file(path),
                              "descriptor": name, "dimension": matrix.shape[1], "n": len(frame),
                              "extraction_seconds": elapsed})


def carregar_cache(config, name, frame=None):
    if frame is None:
        frame = carregar_manifesto(config)
    path = Path(config["processed_dir"]) / f"{name}.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["signature"] != assinatura(config) or metadata["npz_sha256"] != sha256_file(path):
        raise ValueError(f"Cache incompativel ou alterado: {name}")
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["signature"]) != metadata["signature"] or not np.array_equal(cached["ids"], frame.id):
            raise ValueError("Ordem ou identidade do cache incompativel")
        matrix = cached["X"].copy()
    if matrix.shape != (len(frame), metadata["dimension"]) or not np.isfinite(matrix).all():
        raise ValueError("Matriz de caracteristicas invalida")
    return matrix, metadata
