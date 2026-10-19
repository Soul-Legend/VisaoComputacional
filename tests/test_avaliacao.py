import numpy as np
from sklearn.metrics import confusion_matrix

from visao_computacional.avaliacao import metricas


def test_metricas_coerentes_com_matriz_e_classe_ausente():
    y, p = [0, 0, 1, 1, 1], [0, 1, 0, 1, 1]
    result = metricas(y, p)
    cm = confusion_matrix(y, p, labels=[0, 1])
    assert cm.sum() == result["n"] == 5
    assert result["n_livre"] == 2
    assert result["n_ocupada"] == 3
    assert result["accuracy"] == np.trace(cm) / cm.sum()
    assert result["recall_ocupada"] == 2 / 3
    one = metricas([0, 0], [0, 0])
    assert one["n_ocupada"] == 0
    assert one["f1_macro"] == 0.5
