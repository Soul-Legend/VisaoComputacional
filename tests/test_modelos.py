import numpy as np

from visao_computacional.configuracao import carregar_config
from visao_computacional.descritores import vetor
from visao_computacional.treinamento import chave_selecao, pipeline


def test_hog_finito_dimensao_correta():
    config = carregar_config("configs/t1.yaml")
    rgb = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    result = vetor(rgb, "hog", config)
    assert result.shape == (8100,)
    assert np.isfinite(result).all()


def test_scaler_aprende_apenas_no_treino():
    config = carregar_config("configs/t1.yaml")
    X = np.array([[0., 2.], [2., 4.], [4., 6.], [6., 8.]])
    model = pipeline(config, 1)
    model.fit(X, [0, 0, 1, 1])
    mean = model[0].mean_.copy()
    model.predict([[1000., 2000.]])
    np.testing.assert_array_equal(model[0].mean_, mean)
    np.testing.assert_allclose(mean, X.mean(axis=0))


def test_hsv_rgb_e_normalizacao():
    config = carregar_config("configs/t1.yaml")
    red = np.full((128, 128, 3), [255, 0, 0], dtype=np.uint8)
    vector = vetor(red, "hsv", config)
    assert vector.shape == (48,)
    np.testing.assert_allclose(vector.reshape(3, 16).sum(axis=1), 1)
    assert vector[0] == 1  # Vermelho RGB corresponde a H=0 no OpenCV.


def test_desempate_dimensao_e_c():
    candidates = [{"validation_f1_macro": .90001, "dimension": 100, "C": 1},
                  {"validation_f1_macro": .90002, "dimension": 10, "C": 10},
                  {"validation_f1_macro": .90002, "dimension": 10, "C": .1}]
    assert min(candidates, key=chave_selecao) is candidates[2]
