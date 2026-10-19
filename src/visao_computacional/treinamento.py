"""Treinamento somente no treino e selecao somente na validacao."""

import json
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .configuracao import proveniencia, salvar_json, sha256_file
from .dados import carregar_manifesto
from .descritores import assinatura, carregar_cache


def pipeline(config, c):
    return make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=c,
            dual="auto",
            random_state=config["seed"],
            max_iter=config["svm_max_iter"],
            tol=config["svm_tol"],
        ),
    )


def chave_selecao(row):
    return (-round(row["validation_f1_macro"], 4), row["dimension"], row["C"])


def treinar(config):
    frame = carregar_manifesto(config)
    results = Path(config["results_dir"]) / "metricas"
    freeze_path = results / "congelamento.json"
    if freeze_path.exists():
        raise FileExistsError("Modelos ja congelados. Use outros caminhos para novo estudo.")
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    train, val = frame.split.eq("treino").to_numpy(), frame.split.eq("validacao").to_numpy()
    labels = frame.label.to_numpy()
    trials, selected = [], []
    for name in config["descriptors"]:
        matrix, metadata = carregar_cache(config, name, frame)
        candidates = []
        for c in config["c_values"]:
            model = pipeline(config, c)
            start = perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(matrix[train], labels[train])
            seconds = perf_counter() - start
            score = f1_score(
                labels[val], model.predict(matrix[val]), average="macro", labels=[0, 1], zero_division=0
            )
            row = {
                "descriptor": name,
                "C": float(c),
                "dimension": int(matrix.shape[1]),
                "validation_f1_macro": float(score),
                "training_seconds": seconds,
                "n_iter": int(np.max(model[-1].n_iter_)),
            }
            trials.append(row)
            candidates.append((row, model))
            print(f"{name}, C={c}: F1 validacao={score:.4f}", flush=True)
        best, model = min(candidates, key=lambda pair: chave_selecao(pair[0]))
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        selected.append(
            {
                **best,
                "model_file": model_path.name,
                "model_sha256": sha256_file(model_path),
                "cache_sha256": metadata["npz_sha256"],
                "extraction_seconds": metadata["extraction_seconds"],
            }
        )
    dummy = DummyClassifier(strategy="most_frequent")
    start = perf_counter()
    dummy.fit(np.zeros((train.sum(), 1)), labels[train])
    dummy_seconds = perf_counter() - start
    joblib.dump(dummy, models_dir / "majoritario.joblib")
    winner = min(selected, key=chave_selecao)
    pd.DataFrame(trials).to_csv(results / "validacao.csv", index=False)
    freeze = {
        "signature": assinatura(config),
        "selected": winner["descriptor"],
        "models": selected,
        "dummy": {
            "model_file": "majoritario.joblib",
            "model_sha256": sha256_file(models_dir / "majoritario.joblib"),
            "training_seconds": dummy_seconds,
        },
        "provenance": proveniencia(config),
    }
    salvar_json(freeze_path, freeze)
    print(f"Vencedor congelado pela validacao: {winner['descriptor']}", flush=True)


def carregar_congelamento(config):
    freeze = json.loads(
        (Path(config["results_dir"]) / "metricas/congelamento.json").read_text(encoding="utf-8")
    )
    if freeze["signature"] != assinatura(config):
        raise ValueError("Configuracao ou manifesto mudou depois do congelamento")
    for item in freeze["models"] + [freeze["dummy"]]:
        if sha256_file(Path(config["models_dir"]) / item["model_file"]) != item["model_sha256"]:
            raise ValueError("Modelo mudou depois do congelamento")
    return freeze
