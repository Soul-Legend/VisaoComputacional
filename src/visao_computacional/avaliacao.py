"""Lote final de teste, sem alteracao do vencedor escolhido na validacao."""
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .configuracao import proveniencia, salvar_json, sha256_file
from .dados import carregar_manifesto
from .descritores import carregar_cache
from .treinamento import carregar_congelamento


def metricas(y, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    result = {"n": len(y), "accuracy": float(accuracy_score(y, predicted)),
              "f1_macro": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0))}
    for i, name in enumerate(("livre", "ocupada")):
        result.update({f"precision_{name}": float(precision[i]), f"recall_{name}": float(recall[i]),
                       f"f1_{name}": float(f1[i]), f"n_{name}": int(support[i])})
    return result


def avaliar(config):
    freeze = carregar_congelamento(config)
    frame = carregar_manifesto(config)
    base = Path(config["results_dir"])
    metrics_dir = base / "metricas"
    prediction_dir = base / "predicoes"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    marker = metrics_dir / "teste_iniciado.json"
    if marker.exists():
        raise FileExistsError("O teste ja foi aberto neste estudo; resultados nao podem ser sobrescritos.")
    # Validar todos os caches antes de abrir o teste.
    for item in freeze["models"]:
        _, metadata = carregar_cache(config, item["descriptor"], frame)
        if metadata["npz_sha256"] != item["cache_sha256"]:
            raise ValueError("Cache mudou depois do treinamento")
    with marker.open("x", encoding="utf-8") as stream:
        stream.write('{"status": "teste aberto; nao retunar"}\n')
    mask = frame.split.eq("teste").to_numpy()
    test = frame.loc[mask].copy()
    y = test.label.to_numpy()
    rows, climate_rows = [], []
    for item in freeze["models"] + [{**freeze["dummy"], "descriptor": "majoritario", "dimension": 0, "C": 0}]:
        name = item["descriptor"]
        model = joblib.load(Path(config["models_dir"]) / item["model_file"])
        if name == "majoritario":
            X, extraction = np.zeros((len(test), 1)), 0.0
        else:
            matrix, metadata = carregar_cache(config, name, frame)
            X = matrix[mask]
            extraction = metadata["extraction_seconds"]["teste"]
        start = perf_counter()
        predicted = model.predict(X)
        seconds = perf_counter() - start
        # A pontuacao do SVM nao e uma probabilidade.
        scores = model.decision_function(X) if name != "majoritario" else np.full(len(test), np.nan)
        predictions = test[["id", "path", "label", "camera", "date", "weather"]].copy()
        predictions["predicted"] = predicted
        predictions["decision_score"] = scores
        predictions.to_csv(prediction_dir / f"{name}.csv", index=False)
        row = {"descriptor": name, **metricas(y, predicted), "dimension": item["dimension"], "C": item["C"],
               "training_seconds": item["training_seconds"], "inference_seconds": seconds,
               "inference_ms_per_image": seconds * 1000 / len(test), "extraction_seconds": extraction,
               "extraction_ms_per_image": extraction * 1000 / len(test)}
        rows.append(row)
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        pd.DataFrame(matrix, index=["livre", "ocupada"], columns=["livre", "ocupada"]).to_csv(
            metrics_dir / f"confusao_{name}.csv", index_label="verdadeiro")
        for weather, group in predictions.groupby("weather", sort=True):
            climate_rows.append({"descriptor": name, "weather": weather,
                                 **metricas(group.label, group.predicted)})
        print(f"Teste {name}: F1 macro={row['f1_macro']:.4f}, acuracia={row['accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "teste.csv", index=False)
    pd.DataFrame(climate_rows).to_csv(metrics_dir / "clima.csv", index=False)
    salvar_json(metrics_dir / "avaliacao.json", {"selected": freeze["selected"], "provenance": proveniencia(config),
                "freeze_sha256": sha256_file(metrics_dir / "congelamento.json"),
                "test_table_sha256": sha256_file(metrics_dir / "teste.csv"),
                "prediction_sha256": {p.name: sha256_file(p) for p in prediction_dir.glob("*.csv")}})
