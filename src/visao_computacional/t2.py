"""Etapas do T2: selecao das redes, congelamento, teste unico e generalizacao para camera nova."""

import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

from .avaliacao import metricas
from .configuracao import carregar_config, proveniencia, salvar_json, sha256_file
from .dados import carregar_manifesto
from .descritores import carregar_cache
from .redes import carregar_imagens, carregar_modelo, executar, predizer_logits
from .treinamento import pipeline

VERSOES_EXTRAS = ("torch", "torchvision")
COLUNAS = ["id", "path", "label", "camera", "date", "weather"]


def proveniencia_t2(config):
    import importlib.metadata

    record = proveniencia(config)
    record["versions"].update({name: importlib.metadata.version(name) for name in VERSOES_EXTRAS})
    record["torch_threads"] = torch.get_num_threads()
    # Identifica o codigo mesmo quando a arvore git ainda tem alteracoes sem commit.
    root = Path(config["project_root"])
    record["source_sha256"] = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted((root / "src/visao_computacional").glob("*.py"))
    }
    return record


def assinatura_t2(config):
    keys = (
        "image_size",
        "batch_size",
        "weight_decay",
        "cnn",
        "mobilenet",
        "augmentation",
        "augmentation_options",
        "extra_seeds",
        "seed",
    )
    value = {key: config[key] for key in keys}
    value["manifest_sha256"] = sha256_file(config["manifest"])
    return json.dumps(value, sort_keys=True)


def nome_rodada(arquitetura, aumento, seed):
    return f"{arquitetura}_{aumento}_s{seed}"


def dados(config):
    frame = carregar_manifesto(config)
    images, metadata = carregar_imagens(config, frame)
    labels = frame.label.to_numpy()
    index = {s: np.flatnonzero(frame.split.eq(s).to_numpy()) for s in ("treino", "validacao", "teste")}
    return frame, images, labels, index, metadata


def treinar_redes(config):
    metrics_dir = Path(config["results_dir"]) / "metricas"
    freeze_path = metrics_dir / "congelamento.json"
    if freeze_path.exists():
        raise FileExistsError("Redes ja congeladas. Use outro estudo para novo treino.")
    frame, images, labels, index, _ = dados(config)
    train, val = index["treino"], index["validacao"]
    seed = config["seed"]
    rows, runs = [], {}
    for option in config["augmentation_options"]:
        name = nome_rodada("cnn3", option, seed)
        _, record = executar(config, name, "cnn3", images, labels, train, val, option == "com", seed)
        runs[name] = record
        rows.append(
            {
                "run": name,
                "augmentation": option,
                "best_epoch": record["best_epoch"],
                "epochs_run": record["epochs_run"],
                "validation_f1_macro": record["best"]["val_f1_macro"],
                "validation_loss": record["best"]["val_loss"],
                "validation_accuracy": record["best"]["val_accuracy"],
                "training_seconds": record["training_seconds"],
            }
        )
    # Maior F1 macro arredondado; empate favorece a opcao listada primeiro ("sem", mais simples).
    order = {option: i for i, option in enumerate(config["augmentation_options"])}
    chosen = min(rows, key=lambda r: (-round(r["validation_f1_macro"], 4), order[r["augmentation"]]))
    option = chosen["augmentation"]
    print(f"Escolha da validacao: aumento {option}", flush=True)
    for extra in config["extra_seeds"]:
        name = nome_rodada("cnn3", option, extra)
        _, runs[name] = executar(config, name, "cnn3", images, labels, train, val, option == "com", extra)
    name = nome_rodada("mobilenet", option, seed)
    _, runs[name] = executar(config, name, "mobilenet", images, labels, train, val, option == "com", seed)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "selecao.csv", index=False)
    histories = []
    for name, record in runs.items():
        histories.extend({"run": name, **row} for row in record["history"])
    pd.DataFrame(histories).to_csv(metrics_dir / "curvas.csv", index=False)
    salvar_json(
        freeze_path,
        {
            "signature": assinatura_t2(config),
            "augmentation": option,
            "primary": nome_rodada("cnn3", option, seed),
            "seeds": [nome_rodada("cnn3", option, s) for s in [seed, *config["extra_seeds"]]],
            "extension": nome_rodada("mobilenet", option, seed),
            "runs": {
                name: {k: v for k, v in record.items() if k != "history"} for name, record in runs.items()
            },
            "provenance": proveniencia_t2(config),
        },
    )
    print("Redes congeladas antes do teste", flush=True)


def carregar_congelamento_t2(config):
    path = Path(config["results_dir"]) / "metricas/congelamento.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze["signature"] != assinatura_t2(config):
        raise ValueError("Configuracao ou manifesto mudou depois do congelamento das redes")
    for record in freeze["runs"].values():
        if sha256_file(Path(config["models_dir"]) / record["model_file"]) != record["model_sha256"]:
            raise ValueError("Pesos mudaram depois do congelamento")
    return freeze


def avaliar_redes(config):
    freeze = carregar_congelamento_t2(config)
    frame, images, labels, index, metadata = dados(config)
    base = Path(config["results_dir"])
    metrics_dir, prediction_dir = base / "metricas", base / "predicoes"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    marker = metrics_dir / "teste_iniciado.json"
    if marker.exists():
        raise FileExistsError("O teste das redes ja foi aberto; resultados nao podem ser sobrescritos.")
    with marker.open("x", encoding="utf-8") as stream:
        stream.write('{"status": "teste das redes aberto; nao retunar"}\n')
    test_idx = index["teste"]
    test = frame.iloc[test_idx]
    y = labels[test_idx]
    preparation = metadata["read_seconds"]["teste"]
    rows, climate_rows = [], []
    for name, record in freeze["runs"].items():
        model = carregar_modelo(config, record, Path(config["models_dir"]) / record["model_file"])
        torch.set_num_threads(config["threads"])
        predizer_logits(model, images, test_idx[:256])  # aquecimento, fora da medicao
        start = perf_counter()
        logits = predizer_logits(model, images, test_idx)
        seconds = perf_counter() - start
        predicted = (logits > 0).astype(int)
        predictions = test[COLUNAS].copy()
        predictions["predicted"] = predicted
        predictions["probability"] = 1 / (1 + np.exp(-logits.astype(np.float64)))
        predictions.to_csv(prediction_dir / f"{name}.csv", index=False)
        rows.append(
            {
                "model": name,
                **metricas(y, predicted),
                "parameters": record["parameters"],
                "best_epoch": record["best_epoch"],
                "epochs_run": record["epochs_run"],
                "training_seconds": record["training_seconds"],
                "inference_seconds": seconds,
                "inference_ms_per_image": seconds * 1000 / len(test),
                "preparation_seconds": preparation,
                "preparation_ms_per_image": preparation * 1000 / len(test),
            }
        )
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        pd.DataFrame(matrix, index=["livre", "ocupada"], columns=["livre", "ocupada"]).to_csv(
            metrics_dir / f"confusao_{name}.csv", index_label="verdadeiro"
        )
        for weather, group in predictions.groupby("weather", sort=True):
            climate_rows.append({"model": name, "weather": weather, **metricas(group.label, group.predicted)})
        print(f"Teste {name}: F1 macro={rows[-1]['f1_macro']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "teste.csv", index=False)
    pd.DataFrame(climate_rows).to_csv(metrics_dir / "clima.csv", index=False)
    salvar_json(
        metrics_dir / "avaliacao.json",
        {
            "primary": freeze["primary"],
            "provenance": proveniencia_t2(config),
            "freeze_sha256": sha256_file(metrics_dir / "congelamento.json"),
            "test_table_sha256": sha256_file(metrics_dir / "teste.csv"),
            "prediction_sha256": {p.name: sha256_file(p) for p in sorted(prediction_dir.glob("*.csv"))},
        },
    )


def resultados_t2_verificados(config):
    base = Path(config["results_dir"])
    evaluation = json.loads((base / "metricas/avaliacao.json").read_text(encoding="utf-8"))
    if sha256_file(base / "metricas/teste.csv") != evaluation["test_table_sha256"]:
        raise ValueError("Tabela de teste das redes alterada")
    for name, digest in evaluation["prediction_sha256"].items():
        if sha256_file(base / "predicoes" / name) != digest:
            raise ValueError("Predicoes das redes alteradas")
    return evaluation


def generalizacao(config):
    """Deixa uma camera de fora: treino/validacao sem ela, teste somente com ela."""
    freeze = carregar_congelamento_t2(config)
    resultados_t2_verificados(config)
    config_t1 = carregar_config(config["t1_config"])
    frame, images, labels, index, _ = dados(config)
    option = freeze["augmentation"]
    output = Path(config["results_dir"]) / "generalizacao"
    output.mkdir(parents=True, exist_ok=True)
    cameras = sorted(frame.camera.unique())
    matrices = {name: carregar_cache(config_t1, name, frame)[0] for name in config["loco"]["svm_descriptors"]}
    fold_rows = []
    for camera in cameras:
        out_camera = ~frame.camera.eq(camera).to_numpy()
        train = np.intersect1d(index["treino"], np.flatnonzero(out_camera))
        val = np.intersect1d(index["validacao"], np.flatnonzero(out_camera))
        test = np.intersect1d(index["teste"], np.flatnonzero(~out_camera))
        fold_path = output / f"camera{camera}.csv"
        if fold_path.exists():
            print(f"Dobra reaproveitada: camera {camera}", flush=True)
            continue
        predictions = frame.iloc[test][COLUNAS].copy()
        for name, matrix in matrices.items():
            model = pipeline(config_t1, config["loco"]["svm_c"])
            start = perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(matrix[train], labels[train])
            fold_rows.append(
                {"camera": camera, "model": name, "training_seconds": perf_counter() - start, "n_train": len(train)}
            )
            predictions[f"pred_{name}"] = model.predict(matrix[test])
        name = f"cnn3_{option}_c{camera}"
        model, record = executar(
            config, name, "cnn3", images, labels, train, val, option == "com", config["seed"], "generalizacao"
        )
        fold_rows.append(
            {
                "camera": camera,
                "model": "cnn3",
                "training_seconds": record["training_seconds"],
                "n_train": len(train),
                "best_epoch": record["best_epoch"],
            }
        )
        predictions["pred_cnn3"] = (predizer_logits(model, images, test) > 0).astype(int)
        predictions.to_csv(fold_path, index=False)
        pd.DataFrame(fold_rows).to_csv(output / f"treino_camera{camera}.csv", index=False)
        fold_rows = []
    pooled = pd.concat([pd.read_csv(output / f"camera{c}.csv") for c in cameras], ignore_index=True)
    pooled.to_csv(output / "predicoes_camera_nova.csv", index=False)
    salvar_json(
        output / "generalizacao.json",
        {
            "augmentation": option,
            "svm_c": config["loco"]["svm_c"],
            "predictions_sha256": sha256_file(output / "predicoes_camera_nova.csv"),
            "provenance": proveniencia_t2(config),
        },
    )
