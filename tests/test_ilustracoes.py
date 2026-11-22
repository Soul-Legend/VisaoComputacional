import pandas as pd
import pytest

from visao_computacional.ilustracoes import selecionar_exemplos


def test_exemplos_didaticos_nao_usam_par_de_teste():
    frame = pd.DataFrame(
        [
            {
                "split": "teste",
                "camera": 6,
                "slot": 204,
                "date": "2015-11-12",
                "label": label,
                "path": f"teste{label}",
            }
            for label in (0, 1)
        ]
        + [
            {
                "split": "treino",
                "camera": 2,
                "slot": 10,
                "date": "2015-11-13",
                "label": label,
                "path": f"treino{label}",
            }
            for label in (0, 1)
        ]
    )
    pair = selecionar_exemplos(frame)
    assert pair.split.eq("treino").all()
    assert pair.camera.eq(2).all()
    assert pair.label.tolist() == [0, 1]
    with pytest.raises(ValueError, match="vaga de treino"):
        selecionar_exemplos(frame[frame.split == "teste"])
