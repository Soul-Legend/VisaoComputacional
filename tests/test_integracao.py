"""Fluxo pequeno com imagens sinteticas; nao gera os resultados cientificos publicados."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from visao_computacional.avaliacao import avaliar, metricas
from visao_computacional.configuracao import carregar_config
from visao_computacional.dados import preparar
from visao_computacional.descritores import carregar_cache, extrair
from visao_computacional.treinamento import carregar_congelamento, treinar


def test_fluxo_auditoria_congelamento_e_teste_unico(tmp_path):
    config = carregar_config("configs/t1.yaml")
    config.update(
        data_root=str(tmp_path / "raw"),
        manifest=str(tmp_path / "manifest.csv"),
        processed_dir=str(tmp_path / "processed"),
        models_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
        sample_sizes={s: 6 for s in config["sample_sizes"]},
        descriptors=["hsv", "lbp", "hog", "combinado"],
        c_values=[0.01],
    )
    root = Path(config["data_root"])
    (root / "LABELS").mkdir(parents=True)
    rng = np.random.default_rng(4)
    original = None
    for split, date in zip(config["sample_sizes"], ("2015-11-01", "2015-11-02", "2015-11-03")):
        lines = []
        for i in range(8):
            label = i % 2
            relative = f"SUNNY/{date}/camera1/S_{date}_10.{i:02}_C01_{i:03}.jpg"
            image = rng.integers(0, 256, (150, 150, 3), dtype=np.uint8)
            if split == "treino" and i == 0:
                original = image.copy()
            if split == "teste" and i == 1:
                image = original.copy()  # Mesmos pixels, rotulos conflitantes: excluir ambos.
            target = root / "PATCHES" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(target), image)
            lines.append(f"{relative} {label}")
        # Uma copia sem conflito dentro do treino.
        if split == "treino":
            relative = f"SUNNY/{date}/camera1/S_{date}_10.09_C01_009.jpg"
            prior = root / "PATCHES" / lines[2].rsplit(" ", 1)[0]
            (root / "PATCHES" / relative).write_bytes(prior.read_bytes())
            lines.append(f"{relative} 0")
        (root / config["split_files"][split]).write_text("\n".join(lines))
    preparar(config)
    audit = json.loads((Path(config["results_dir"]) / "metricas/auditoria.json").read_text())
    assert audit["conflicting_labels"] == 2
    assert audit["duplicates_removed"] == 1
    frame = pd.read_csv(config["manifest"])
    assert len(frame) == 18
    assert frame.pixel_sha256.is_unique
    with pytest.raises(FileExistsError):
        preparar(config)
    extrair(config)
    matrix, _ = carregar_cache(config, "hsv")
    assert matrix.shape == (18, 48)
    treinar(config)
    freeze = carregar_congelamento(config)
    assert freeze["selected"] in config["descriptors"]
    with pytest.raises(FileExistsError):
        treinar(config)
    avaliar(config)
    with pytest.raises(FileExistsError):
        avaliar(config)
    results = Path(config["results_dir"])
    table = pd.read_csv(results / "metricas/teste.csv").set_index("descriptor")
    assert len(table) == 5
    for name in table.index:
        predicted = pd.read_csv(results / "predicoes" / f"{name}.csv")
        assert len(predicted) == 6
        expected = metricas(predicted.label, predicted.predicted)
        for key, value in expected.items():
            assert table.loc[name, key] == pytest.approx(value)
    cache = Path(config["processed_dir"]) / "hsv.json"
    metadata = json.loads(cache.read_text())
    metadata["signature"] = "alterado"
    cache.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Cache incompativel"):
        carregar_cache(config, "hsv")
    model_path = Path(config["models_dir"]) / freeze["models"][0]["model_file"]
    with model_path.open("ab") as stream:
        stream.write(b"alterado")
    with pytest.raises(ValueError, match="Modelo mudou"):
        carregar_congelamento(config)
