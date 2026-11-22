"""Consistencia das evidencias versionadas do T2, sem acesso as imagens brutas."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visao_computacional.avaliacao import metricas
from visao_computacional.comparacao import contagens, f1_macro
from visao_computacional.configuracao import sha256_file

BASE = Path("resultados/t2")
pytestmark = pytest.mark.skipif(
    not (BASE / "metricas/avaliacao.json").exists(), reason="Experimento do T2 nao publicado"
)


def test_identidade_do_congelamento_e_predicoes():
    evaluation = json.loads((BASE / "metricas/avaliacao.json").read_text(encoding="utf-8"))
    freeze = json.loads((BASE / "metricas/congelamento.json").read_text(encoding="utf-8"))
    assert freeze["provenance"]["manifest_sha256"] == sha256_file("data/manifests/t1.csv")
    assert sha256_file(BASE / "metricas/congelamento.json") == evaluation["freeze_sha256"]
    assert sha256_file(BASE / "metricas/teste.csv") == evaluation["test_table_sha256"]
    for filename, expected in evaluation["prediction_sha256"].items():
        assert sha256_file(BASE / "predicoes" / filename) == expected
    assert freeze["primary"] in freeze["seeds"]


def test_metricas_do_t2_correspondem_as_predicoes_no_mesmo_teste_do_t1():
    table = pd.read_csv(BASE / "metricas/teste.csv").set_index("model")
    manifest = pd.read_csv("data/manifests/t1.csv", dtype={"id": str})
    ids = manifest.loc[manifest.split == "teste", "id"].tolist()
    for name in table.index:
        predicted = pd.read_csv(BASE / "predicoes" / f"{name}.csv", dtype={"id": str})
        assert predicted.id.tolist() == ids
        assert ((predicted.probability > 0.5).astype(int) == predicted.predicted).all()
        for key, value in metricas(predicted.label, predicted.predicted).items():
            assert table.loc[name, key] == pytest.approx(value)
        cm = pd.read_csv(BASE / "metricas" / f"confusao_{name}.csv", index_col=0).to_numpy()
        assert np.trace(cm) / cm.sum() == pytest.approx(table.loc[name, "accuracy"])


def test_comparacao_usa_os_valores_pontuais_das_predicoes():
    summary = pd.read_csv(BASE / "metricas/comparacao.csv").set_index("model")
    for key, path in (
        ("lbp", "resultados/t1/predicoes/lbp.csv"),
        ("combinado", "resultados/t1/predicoes/combinado.csv"),
    ):
        predicted = pd.read_csv(path)
        counts = contagens(predicted.label, predicted.predicted, np.zeros(len(predicted), int), 1)[0]
        assert summary.loc[key, "f1_macro"] == pytest.approx(f1_macro(counts))
    assert (summary.f1_low <= summary.f1_macro + 1e-12).all()
    assert (summary.f1_macro <= summary.f1_high + 1e-12).all()


def test_generalizacao_cobre_todo_o_teste_sem_ver_a_camera():
    path = BASE / "generalizacao/predicoes_camera_nova.csv"
    if not path.exists():
        pytest.skip("Generalizacao nao publicada")
    pooled = pd.read_csv(path)
    assert len(pooled) == 1500 and pooled.id.is_unique
    for camera in pooled.camera.unique():
        assert (BASE / f"generalizacao/camera{camera}.csv").exists()
