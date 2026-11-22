"""Consistencia entre evidencias versionadas, sem acesso as imagens brutas."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visao_computacional.avaliacao import metricas
from visao_computacional.configuracao import sha256_file

BASE = Path("resultados/t1")
pytestmark = pytest.mark.skipif(
    not (BASE / "metricas/avaliacao.json").exists(), reason="Experimento nao publicado"
)


def test_identidade_manifesto_e_predicoes_publicadas():
    evaluation = json.loads((BASE / "metricas/avaliacao.json").read_text(encoding="utf-8"))
    freeze = json.loads((BASE / "metricas/congelamento.json").read_text(encoding="utf-8"))
    audit = json.loads((BASE / "metricas/auditoria.json").read_text(encoding="utf-8"))
    digest = sha256_file("data/manifests/t1.csv")
    assert audit["manifest_sha256"] == freeze["provenance"]["manifest_sha256"] == digest
    assert sha256_file(BASE / "metricas/congelamento.json") == evaluation["freeze_sha256"]
    assert sha256_file(BASE / "metricas/teste.csv") == evaluation["test_table_sha256"]
    for filename, expected in evaluation["prediction_sha256"].items():
        assert sha256_file(BASE / "predicoes" / filename) == expected


def test_metricas_matrizes_e_clima_correspondem_as_predicoes():
    table = pd.read_csv(BASE / "metricas/teste.csv").set_index("descriptor")
    climate = pd.read_csv(BASE / "metricas/clima.csv")
    manifest = pd.read_csv("data/manifests/t1.csv", dtype={"id": str})
    ids = manifest.loc[manifest.split == "teste", "id"].tolist()
    for name in table.index:
        predicted = pd.read_csv(BASE / "predicoes" / f"{name}.csv", dtype={"id": str})
        assert predicted.id.tolist() == ids
        for key, value in metricas(predicted.label, predicted.predicted).items():
            assert table.loc[name, key] == pytest.approx(value)
        cm = pd.read_csv(BASE / "metricas" / f"confusao_{name}.csv", index_col=0).to_numpy()
        assert cm.sum() == len(predicted)
        assert np.trace(cm) / cm.sum() == pytest.approx(table.loc[name, "accuracy"])
        for weather, subset in predicted.groupby("weather"):
            row = climate[(climate.descriptor == name) & (climate.weather == weather)].iloc[0]
            for key, value in metricas(subset.label, subset.predicted).items():
                assert row[key] == pytest.approx(value)


def test_selecao_foi_preservada_apesar_do_ranking_de_teste():
    freeze = json.loads((BASE / "metricas/congelamento.json").read_text(encoding="utf-8"))
    table = pd.read_csv(BASE / "metricas/teste.csv").set_index("descriptor")
    assert freeze["selected"] == "lbp"
    assert table.loc["combinado", "f1_macro"] > table.loc["lbp", "f1_macro"]
