"""Comparacao pareada entre T1 e T2 nas mesmas 1.500 imagens de teste.

Os intervalos usam bootstrap agrupado por captura (camera + data + horario): imagens da mesma
foto compartilham iluminacao e clima, entao reamostrar imagens isoladas subestimaria a incerteza.
Nao ha ajuste de modelos aqui; apenas leitura de predicoes congeladas.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .configuracao import carregar_config, salvar_json, sha256_file
from .dados import carregar_manifesto
from .t2 import resultados_t2_verificados
from .visualizacao import resultados_verificados

NOMES = {
    "hsv": "HSV + SVM",
    "hog": "HOG + SVM",
    "lbp": "LBP + SVM",
    "combinado": "HSV + LBP + HOG + SVM",
    "majoritario": "Majoritário",
    "cnn3": "CNN-3",
    "cnn3_s43": "CNN-3 (semente 43)",
    "cnn3_s44": "CNN-3 (semente 44)",
    "mobilenet": "MobileNetV2",
}
PARES = [("cnn3", "lbp"), ("cnn3", "combinado"), ("mobilenet", "cnn3"), ("mobilenet", "combinado")]


def chaves_t2(freeze):
    """Associa nomes curtos as rodadas congeladas do T2."""
    keys = {freeze["primary"]: "cnn3", freeze["extension"]: "mobilenet"}
    for run in freeze["seeds"][1:]:
        keys[run] = "cnn3_s" + run.rsplit("_s", 1)[1]
    return keys


def carregar_predicoes(config):
    config_t1 = carregar_config(config["t1_config"])
    resultados_verificados(config_t1)
    resultados_t2_verificados(config)
    freeze = json.loads(
        (Path(config["results_dir"]) / "metricas/congelamento.json").read_text(encoding="utf-8")
    )
    t1_dir = Path(config_t1["results_dir"]) / "predicoes"
    reference = pd.read_csv(t1_dir / "lbp.csv", dtype={"id": str})
    predictions = {}
    for key in ("hsv", "hog", "lbp", "combinado", "majoritario"):
        predictions[key] = pd.read_csv(t1_dir / f"{key}.csv", dtype={"id": str})
    for run, key in chaves_t2(freeze).items():
        predictions[key] = pd.read_csv(
            Path(config["results_dir"]) / "predicoes" / f"{run}.csv", dtype={"id": str}
        )
    for key, frame in predictions.items():
        frame = frame.set_index("id").loc[reference.id].reset_index()
        if not (frame.label.to_numpy() == reference.label.to_numpy()).all():
            raise ValueError(f"Rotulos de {key} nao conferem com o teste do T1")
        predictions[key] = frame
    manifest = carregar_manifesto(config).set_index("id").loc[reference.id]
    return reference, manifest, predictions, freeze


def contagens(y, predicted, groups, n_groups):
    """Matriz (grupos, 4) com [tn, fp, fn, tp] de cada captura."""
    codes = 2 * np.asarray(y) + np.asarray(predicted)  # 0=tn, 1=fp, 2=fn, 3=tp
    result = np.zeros((n_groups, 4))
    np.add.at(result, (groups, codes), 1)
    return result


def f1_macro(c):
    tn, fp, fn, tp = (c[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_0 = np.where(2 * tn + fp + fn > 0, 2 * tn / (2 * tn + fp + fn), 0.0)
    return (f1_0 + f1_1) / 2


def acuracia(c):
    return (c[..., 0] + c[..., 3]) / c.sum(axis=-1)


def pesos_bootstrap(n_groups, resamples, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(resamples, n_groups))
    weights = np.zeros((resamples, n_groups))
    np.add.at(weights, (np.arange(resamples)[:, None], draws), 1)
    return weights


def intervalo(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def mcnemar(y, a, b):
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p)


def resumo_modelos(y, predictions, groups, weights, keys):
    n_groups = weights.shape[1]
    rows, samples = [], {}
    for key in keys:
        per_group = contagens(y, predictions[key].predicted.to_numpy(), groups, n_groups)
        resampled = weights @ per_group
        total = per_group.sum(axis=0)
        samples[key] = f1_macro(resampled)
        f1_low, f1_high = intervalo(samples[key])
        acc_low, acc_high = intervalo(acuracia(resampled))
        rows.append(
            {
                "model": key,
                "name": NOMES[key],
                "n": int(total.sum()),
                "accuracy": float(acuracia(total)),
                "accuracy_low": acc_low,
                "accuracy_high": acc_high,
                "f1_macro": float(f1_macro(total)),
                "f1_low": f1_low,
                "f1_high": f1_high,
            }
        )
    return pd.DataFrame(rows), samples


def comparar(config):
    reference, manifest, predictions, _ = carregar_predicoes(config)
    y = reference.label.to_numpy()
    captures = manifest.capture.to_numpy()
    _, groups = np.unique(captures, return_inverse=True)
    n_groups = groups.max() + 1
    resamples, seed = config["bootstrap_resamples"], config["seed"]
    weights = pesos_bootstrap(n_groups, resamples, seed)
    output = Path(config["results_dir"]) / "metricas"
    keys = list(predictions)
    summary, samples = resumo_modelos(y, predictions, groups, weights, keys)
    summary.to_csv(output / "comparacao.csv", index=False)

    pair_rows = []
    for a, b in PARES:
        diff = samples[a] - samples[b]
        low, high = intervalo(diff)
        only_a, only_b, p = mcnemar(
            y, predictions[a].predicted.to_numpy(), predictions[b].predicted.to_numpy()
        )
        point = summary.set_index("model")
        pair_rows.append(
            {
                "a": a,
                "b": b,
                "f1_diff": float(point.loc[a, "f1_macro"] - point.loc[b, "f1_macro"]),
                "diff_low": low,
                "diff_high": high,
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "mcnemar_p": p,
            }
        )
    pd.DataFrame(pair_rows).to_csv(output / "pares.csv", index=False)

    # Clima: as capturas pertencem a um unico clima, entao o bootstrap fica restrito ao grupo.
    climate_rows = []
    weather = manifest.weather.to_numpy()
    for condition in sorted(np.unique(weather)):
        mask = weather == condition
        _, local = np.unique(captures[mask], return_inverse=True)
        local_weights = pesos_bootstrap(local.max() + 1, resamples, seed)
        local_predictions = {k: v[mask] for k, v in predictions.items()}
        table, _ = resumo_modelos(y[mask], local_predictions, local, local_weights, keys)
        table.insert(0, "weather", condition)
        table["n_livre"] = int((y[mask] == 0).sum())
        table["n_ocupada"] = int((y[mask] == 1).sum())
        climate_rows.append(table)
    pd.concat(climate_rows).to_csv(output / "clima_comparado.csv", index=False)

    overlap_rows = []
    for other in ("lbp", "combinado"):
        cnn_wrong = predictions["cnn3"].predicted.to_numpy() != y
        other_wrong = predictions[other].predicted.to_numpy() != y
        overlap_rows.append(
            {
                "classico": other,
                "ambos_corretos": int((~cnn_wrong & ~other_wrong).sum()),
                "so_cnn_errou": int((cnn_wrong & ~other_wrong).sum()),
                "so_classico_errou": int((~cnn_wrong & other_wrong).sum()),
                "ambos_erraram": int((cnn_wrong & other_wrong).sum()),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(output / "sobreposicao_erros.csv", index=False)

    record = {
        "resamples": resamples,
        "seed": seed,
        "clusters": int(n_groups),
        "cluster": "capture (camera + data + horario)",
        "inputs_sha256": {
            "t2_avaliacao": sha256_file(output / "avaliacao.json"),
        },
    }
    loco_path = Path(config["results_dir"]) / "generalizacao/predicoes_camera_nova.csv"
    if loco_path.exists():
        record["generalizacao"] = comparar_generalizacao(config, reference, manifest, predictions, loco_path)
    salvar_json(output / "comparacao.json", record)


def comparar_generalizacao(config, reference, manifest, predictions, loco_path):
    """Mesmas imagens de teste: modelo que viu a camera (principal) x modelo sem a camera."""
    loco = pd.read_csv(loco_path, dtype={"id": str}).set_index("id").loc[reference.id].reset_index()
    y = reference.label.to_numpy()
    captures = manifest.capture.to_numpy()
    _, groups = np.unique(captures, return_inverse=True)
    weights = pesos_bootstrap(groups.max() + 1, config["bootstrap_resamples"], config["seed"])
    rows, pooled = [], []
    cameras = manifest.camera.to_numpy()
    for key in ("lbp", "combinado", "cnn3"):
        seen = predictions[key].predicted.to_numpy()
        unseen = loco[f"pred_{key}"].to_numpy()
        for camera in sorted(np.unique(cameras)):
            mask = cameras == camera
            rows.append(
                {
                    "model": key,
                    "camera": int(camera),
                    "n": int(mask.sum()),
                    "f1_vista": float(
                        f1_macro(contagens(y[mask], seen[mask], np.zeros(mask.sum(), int), 1)[0])
                    ),
                    "f1_nova": float(
                        f1_macro(contagens(y[mask], unseen[mask], np.zeros(mask.sum(), int), 1)[0])
                    ),
                }
            )
        seen_counts = contagens(y, seen, groups, groups.max() + 1)
        unseen_counts = contagens(y, unseen, groups, groups.max() + 1)
        diff = f1_macro(weights @ unseen_counts) - f1_macro(weights @ seen_counts)
        unseen_samples = f1_macro(weights @ unseen_counts)
        pooled.append(
            {
                "model": key,
                "f1_vista": float(f1_macro(seen_counts.sum(0))),
                "f1_nova": float(f1_macro(unseen_counts.sum(0))),
                "f1_nova_low": intervalo(unseen_samples)[0],
                "f1_nova_high": intervalo(unseen_samples)[1],
                "acc_vista": float(acuracia(seen_counts.sum(0))),
                "acc_nova": float(acuracia(unseen_counts.sum(0))),
                "diff": float(f1_macro(unseen_counts.sum(0)) - f1_macro(seen_counts.sum(0))),
                "diff_low": intervalo(diff)[0],
                "diff_high": intervalo(diff)[1],
            }
        )
    pair_rows = []
    unseen_samples = {
        key: f1_macro(weights @ contagens(y, loco[f"pred_{key}"].to_numpy(), groups, groups.max() + 1))
        for key in ("lbp", "combinado", "cnn3")
    }
    for other in ("lbp", "combinado"):
        a, b = loco["pred_cnn3"].to_numpy(), loco[f"pred_{other}"].to_numpy()
        only_a, only_b, p = mcnemar(y, a, b)
        low, high = intervalo(unseen_samples["cnn3"] - unseen_samples[other])
        point = {r["model"]: r["f1_nova"] for r in pooled}
        pair_rows.append(
            {
                "a": "cnn3",
                "b": other,
                "f1_diff": point["cnn3"] - point[other],
                "diff_low": low,
                "diff_high": high,
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "mcnemar_p": p,
            }
        )
    output = Path(config["results_dir"]) / "metricas"
    pd.DataFrame(pair_rows).to_csv(output / "generalizacao_pares.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "generalizacao_cameras.csv", index=False)
    pd.DataFrame(pooled).to_csv(output / "generalizacao.csv", index=False)
    return {"predictions_sha256": sha256_file(loco_path)}
