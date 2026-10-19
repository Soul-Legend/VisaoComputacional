"""Figuras e tabelas derivadas de manifestos e resultados, sem ajustar modelos."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dados import carregar_manifesto, caminho_seguro
from .preprocessamento import ler_rgb

CORES = ["#2563eb", "#ea580c"]
NOMES = {0: "Livre", 1: "Ocupada"}


def estilo():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


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
            metrics / f"distribuicao_{field}.csv", index=False)
    frame.groupby(["width", "height"]).size().rename("n").reset_index().to_csv(
        metrics / "resolucoes.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, field, title in zip(axes, ("label", "weather"), ("Classes por partição", "Clima por partição")):
        counts = frame.groupby(["split", field]).size().unstack(fill_value=0).reindex(
            ["treino", "validacao", "teste"])
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
            groups.extend(candidates.sample(min(2, len(candidates)), random_state=config["seed"]).to_dict("records"))
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, groups):
        ax.imshow(ler_rgb(caminho_seguro(config["data_root"], row["path"])))
        ax.set_title(f"{NOMES[row['label']]} | {row['weather']}\nCâmera {row['camera']}", fontsize=9)
    fig.suptitle("Exemplos de treino — CNRPark+EXT (CNR-EXT)")
    fig.tight_layout()
    salvar_figura(fig, base / "figuras/exemplos_treino.png")
