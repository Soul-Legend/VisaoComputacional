import pandas as pd
import pytest

from visao_computacional.dados import amostrar, caminho_seguro, validar_manifesto


def test_amostragem_exata_reproduzivel_e_independente_da_ordem():
    frame = pd.DataFrame(
        {
            "path": [f"p{i:03}" for i in range(100)],
            "label": [i % 2 for i in range(100)],
            "camera": [1] * 100,
            "weather": ["SUNNY"] * 80 + ["RAINY"] * 20,
        }
    )
    a = amostrar(frame, 31, 42)
    b = amostrar(frame.sample(frac=1, random_state=5), 31, 42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 31
    assert abs((a.label == 0).mean() - 0.5) < 0.04
    assert abs((a.weather == "RAINY").mean() - 0.2) < 0.04


def test_nao_permite_escape_da_raiz(tmp_path):
    with pytest.raises(ValueError, match="fora da raiz"):
        caminho_seguro(tmp_path, "../fora.jpg")


def test_rejeita_duplicatas_entre_particoes():
    rows = [
        {
            "id": str(i),
            "path": str(i),
            "label": i % 2,
            "camera": 1,
            "date": "2015-11-12",
            "weather": "SUNNY",
            "capture": "x",
            "split": split,
            "pixel_sha256": "igual",
        }
        for i, split in enumerate(("treino", "validacao", "teste"))
    ]
    with pytest.raises(ValueError, match="duplicatas"):
        validar_manifesto(pd.DataFrame(rows))
