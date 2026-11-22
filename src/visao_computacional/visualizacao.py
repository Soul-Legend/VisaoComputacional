"""Figuras e tabelas derivadas de manifestos e resultados, sem ajustar modelos."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuracao import sha256_file
from .dados import caminho_seguro, carregar_manifesto
from .preprocessamento import ler_rgb

CORES = ["#2563eb", "#ea580c"]
NOMES = {0: "Livre", 1: "Ocupada"}
DESCRITORES = {
    "hog": "HOG",
    "hsv": "HSV",
    "lbp": "LBP",
    "combinado": "HSV + LBP + HOG",
    "majoritario": "Majoritário",
}


def estilo():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def salvar_figura(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def explorar(config):
    estilo()
    frame = carregar_manifesto(config)
    base = Path(config["results_dir"])
    metrics = base / "metricas"
    metrics.mkdir(parents=True, exist_ok=True)
    for field in ("label", "weather", "camera", "date"):
        frame.groupby(["split", field]).size().rename("n").reset_index().to_csv(
            metrics / f"distribuicao_{field}.csv", index=False
        )
    frame.groupby(["width", "height"]).size().rename("n").reset_index().to_csv(
        metrics / "resolucoes.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, field, title in zip(axes, ("label", "weather"), ("Classes por partição", "Clima por partição")):
        counts = (
            frame.groupby(["split", field])
            .size()
            .unstack(fill_value=0)
            .reindex(["treino", "validacao", "teste"])
        )
        if field == "label":
            counts = counts.rename(columns=NOMES)
        counts.plot.bar(ax=ax, color=CORES if field == "label" else None, rot=0)
        ax.set(title=title, xlabel="", ylabel="Número de recortes")
        ax.legend(fontsize=8)
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/distribuicoes.png")
    # Exemplos apenas do treino; nao expor o teste durante a exploracao.
    train = frame[frame.split == "treino"]
    groups = []
    for weather in sorted(train.weather.unique()):
        for label in (0, 1):
            candidates = train[(train.weather == weather) & (train.label == label)]
            groups.extend(
                candidates.sample(min(2, len(candidates)), random_state=config["seed"]).to_dict("records")
            )
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, groups):
        ax.imshow(ler_rgb(caminho_seguro(config["data_root"], row["path"])))
        ax.set_title(f"{NOMES[row['label']]} | {row['weather']}\nCâmera {row['camera']}", fontsize=9)
    fig.suptitle("Exemplos de treino — CNRPark+EXT (CNR-EXT)")
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/exemplos_treino.png")


def resultados_verificados(config):
    base = Path(config["results_dir"])
    evaluation = json.loads((base / "metricas/avaliacao.json").read_text(encoding="utf-8"))
    if sha256_file(base / "metricas/teste.csv") != evaluation["test_table_sha256"]:
        raise ValueError("Tabela de teste alterada depois da avaliacao")
    for name, digest in evaluation["prediction_sha256"].items():
        if sha256_file(base / "predicoes" / name) != digest:
            raise ValueError("Predicoes alteradas depois da avaliacao")
    return evaluation


def figuras(config):
    estilo()
    evaluation = resultados_verificados(config)
    base = Path(config["results_dir"])
    metrics = pd.read_csv(base / "metricas/teste.csv")
    climate = pd.read_csv(base / "metricas/clima.csv")
    names = metrics.descriptor.tolist()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, metrics.accuracy, 0.36, label="Acurácia", color=CORES[0])
    ax.bar(x + 0.18, metrics.f1_macro, 0.36, label="F1 macro", color=CORES[1])
    labels = ["HSV + LBP\n+ HOG" if n == "combinado" else DESCRITORES[n] for n in names]
    ax.set(
        xticks=x, xticklabels=labels, ylim=(0, 1.1), ylabel="Escore", title="Comparação no teste congelado"
    )
    ax.legend(loc="lower right")
    for i, value in enumerate(metrics.f1_macro):
        ax.text(i + 0.18, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/comparacao.png")
    pivot = climate.pivot(index="weather", columns="descriptor", values="f1_macro").reindex(columns=names)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    pivot.rename(columns=DESCRITORES).plot.bar(ax=ax, rot=0, ylim=(0, 1.15))
    ax.set(xlabel="Clima", ylabel="F1 macro", title="Desagregação por clima (sem ajuste de modelos)")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/clima.png")
    winner = evaluation["selected"]
    prediction = pd.read_csv(base / "predicoes" / f"{winner}.csv", dtype={"id": str})
    matrix = (
        pd.crosstab(prediction.label, prediction.predicted)
        .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        .to_numpy()
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, values, title in zip(axes, (matrix, normalized), ("Contagens", "Por classe verdadeira")):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=np.max(values))
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Livre", "Ocupada"],
            yticklabels=["Livre", "Ocupada"],
            xlabel="Previsão",
            ylabel="Verdadeiro",
            title=title,
        )
        for i in range(2):
            for j in range(2):
                text = f"{values[i, j]:.1%}" if values.dtype.kind == "f" else str(values[i, j])
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > values.max() / 2 else "black",
                )
    fig.suptitle(f"Matriz de confusão — {winner} (selecionado na validação)")
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/confusao_vencedor.png")
    errors = prediction[prediction.label != prediction.predicted]
    chosen = (
        pd.concat(
            [g.sample(min(6, len(g)), random_state=config["seed"]) for _, g in errors.groupby("label")],
            ignore_index=True,
        )
        if len(errors)
        else errors.copy()
    )
    chosen.to_csv(base / "metricas/erros_selecionados.csv", index=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, chosen.itertuples(index=False)):
        ax.imshow(ler_rgb(caminho_seguro(config["data_root"], row.path)))
        ax.set_title(
            f"Real: {NOMES[row.label]} | Prev.: {NOMES[row.predicted]}\n{row.weather}, câmera {row.camera}",
            fontsize=8,
        )
    if chosen.empty:
        axes.flat[0].text(0.1, 0.5, "Nenhum erro neste teste.")
    fig.suptitle(f"Erros ilustrativos — {winner}; origem: CNRPark+EXT")
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/erros_vencedor.png")
