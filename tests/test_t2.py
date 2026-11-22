"""Verificacoes do T2 com dados sinteticos; nao geram resultados cientificos."""

import numpy as np
import torch
from sklearn.metrics import f1_score

from visao_computacional.comparacao import (
    acuracia,
    chaves_t2,
    contagens,
    f1_macro,
    mcnemar,
    pesos_bootstrap,
)
from visao_computacional.redes import CNN3, aumentar, chave_epoca, estatisticas_canal, grad_cam, resumo

AUMENTO = {
    "rotation_deg": 8,
    "zoom": [0.9, 1.1],
    "translate": 0.06,
    "brightness": [0.8, 1.2],
    "contrast": [0.8, 1.2],
}


def test_aumento_preserva_forma_intervalo_e_semente():
    x = torch.rand(4, 3, 32, 32)
    a = aumentar(x, AUMENTO, torch.Generator().manual_seed(1))
    b = aumentar(x, AUMENTO, torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert float(a.min()) >= 0 and float(a.max()) <= 1
    assert torch.equal(a, b)


def test_aumento_neutro_e_identidade():
    neutral = {"rotation_deg": 0, "zoom": [1, 1], "translate": 0, "brightness": [1, 1], "contrast": [1, 1]}
    x = torch.rand(2, 3, 16, 16)
    assert torch.allclose(aumentar(x, neutral, torch.Generator().manual_seed(0)), x, atol=1e-5)


def test_cnn3_produz_um_logit_e_grad_cam_normalizado():
    model = CNN3([0.5] * 3, [0.25] * 3, filters=(4, 8, 8), dense=8, size=32)
    x = torch.rand(3, 3, 32, 32)
    assert model(x).shape == (3,)
    cam, logits = grad_cam(model, x)
    assert cam.shape == (3, 32, 32) and logits.shape == (3,)
    assert np.all(cam >= 0) and np.all(cam <= 1 + 1e-6)


def test_estatisticas_usam_somente_indices_informados():
    images = torch.zeros(4, 3, 2, 2, dtype=torch.uint8)
    images[2:] = 255
    mean, std = estatisticas_canal(images, np.array([0, 1]))
    assert mean == [0.0, 0.0, 0.0] and std == [0.0, 0.0, 0.0]


def test_escolha_da_epoca_prefere_f1_depois_perda_depois_epoca():
    rows = [
        {"epoch": 1, "val_f1_macro": 0.90001, "val_loss": 0.30},
        {"epoch": 2, "val_f1_macro": 0.90004, "val_loss": 0.20},
        {"epoch": 3, "val_f1_macro": 0.89, "val_loss": 0.10},
    ]
    assert min(rows, key=chave_epoca)["epoch"] == 2


def test_resumo_usa_logit_zero_como_limiar():
    result = resumo(np.array([0, 1, 1, 0]), np.array([-1.0, 2.0, 0.5, 0.1]))
    assert result["accuracy"] == 0.75


def test_f1_por_contagens_coincide_com_sklearn():
    rng = np.random.default_rng(0)
    y, predicted = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    groups = rng.integers(0, 7, 200)
    counts = contagens(y, predicted, groups, 7)
    expected = f1_score(y, predicted, average="macro")
    assert np.isclose(f1_macro(counts.sum(0)), expected)
    assert np.isclose(acuracia(counts.sum(0)), (y == predicted).mean())


def test_pesos_bootstrap_somam_numero_de_grupos():
    weights = pesos_bootstrap(5, 10, 42)
    assert weights.shape == (10, 5) and np.all(weights.sum(1) == 5)


def test_mcnemar_conta_discordancias():
    y = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 1, 0, 1, 0])
    b = np.array([0, 1, 0, 0, 0])
    only_a, only_b, p = mcnemar(y, a, b)
    assert (only_a, only_b) == (1, 1) and p == 1.0


def test_chaves_t2_nomeiam_rodadas():
    freeze = {
        "primary": "cnn3_com_s42",
        "extension": "mobilenet_com_s42",
        "seeds": ["cnn3_com_s42", "cnn3_com_s43"],
    }
    assert chaves_t2(freeze) == {
        "cnn3_com_s42": "cnn3",
        "mobilenet_com_s42": "mobilenet",
        "cnn3_com_s43": "cnn3_s43",
    }
