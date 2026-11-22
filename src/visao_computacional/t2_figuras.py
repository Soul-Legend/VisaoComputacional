"""Figuras do T2 e da comparacao final, geradas de pesos e predicoes congelados.

Nenhuma funcao daqui treina ou ajusta modelos. As ilustracoes de metodo usam somente imagens
de treino; as figuras de erro e Grad-CAM usam o teste ja avaliado, sem retroalimentar escolhas.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter

from .comparacao import NOMES, carregar_predicoes
from .configuracao import salvar_json
from .dados import caminho_seguro, carregar_manifesto
from .ilustracoes import selecionar_exemplos
from .preprocessamento import ler_rgb
from .redes import aumentar, carregar_modelo, grad_cam
from .visualizacao import NOMES as CLASSES
from .visualizacao import estilo, salvar_figura

AZUL, LARANJA, VIOLETA, VERDE, CINZA = "#2563eb", "#ea580c", "#7c3aed", "#059669", "#64748b"
CORES_MODELOS = {
    "lbp": AZUL,
    "combinado": VIOLETA,
    "cnn3": LARANJA,
    "mobilenet": VERDE,
    "hog": CINZA,
    "hsv": "#94a3b8",
}
HACHURAS = {"lbp": "", "combinado": "//", "cnn3": "..", "mobilenet": "xx"}


def _virgula(ax, axis="y", digits=2):
    """Marcas do eixo com virgula decimal, como no texto do relatorio."""
    formatter = FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def _limpar(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255


def _modelo_principal(config):
    freeze = json.loads(
        (Path(config["results_dir"]) / "metricas/congelamento.json").read_text(encoding="utf-8")
    )
    record = freeze["runs"][freeze["primary"]]
    model = carregar_modelo(config, record, Path(config["models_dir"]) / record["model_file"])
    return model, freeze


def redes_figuras(config):
    estilo()
    plt.rcParams.update({"font.size": 10})
    output = Path(config["results_dir"]) / "figuras"
    output.mkdir(parents=True, exist_ok=True)
    frame = carregar_manifesto(config)
    pair = selecionar_exemplos(frame)
    examples = [ler_rgb(caminho_seguro(config["data_root"], p), config["image_size"]) for p in pair.path]
    model, freeze = _modelo_principal(config)
    arquitetura(config, output)
    aumento(config, examples[1], output)
    curvas(config, freeze, output)
    mapas(model, examples, output)
    reference, _, predictions, _ = carregar_predicoes(config)
    confusoes(reference, predictions, output)
    comparacao_ic(config, output)
    clima(config, output)
    erros(config, predictions["cnn3"], output)
    discordancias(config, predictions, output)
    gradcam_teste(config, model, predictions["cnn3"], output)
    if (Path(config["results_dir"]) / "metricas/generalizacao_cameras.csv").exists():
        generalizacao(config, output)
    montagens_anotacao(config, reference, predictions)


# ---------------------------------------------------------------- metodo


def arquitetura(config, output):
    """Blocos com altura proporcional a resolucao espacial e largura ao numero de canais."""
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.set(xlim=(0, 16.9), ylim=(-2.3, 2.6))
    ax.axis("off")
    filters = config["cnn"]["filters"]
    side = config["image_size"][0]
    stages = [("Entrada", f"{side}×{side}×3", side, 3, "#e7eef7")]
    for i, f in enumerate(filters, 1):
        side //= 2
        stages.append(
            (f"Bloco {i}", f"conv 3×3, {f}\nBN, ReLU\npool → {side}×{side}", side * 2, f, "#dbeafe")
        )
    flat = filters[-1] * side * side
    stages += [
        ("Flatten", f"{flat:,}".replace(",", "."), None, None, "#fde7d8"),
        ("Densa", f"dropout 0,5\n{config['cnn']['dense']} + ReLU", None, None, "#fde7d8"),
        ("Saída", "1 logit\nsigmoide", None, None, "#fcd9bd"),
    ]
    x = 0.4
    for title, text, size, channels, color in stages:
        if size:
            height = 0.6 + 2.4 * size / config["image_size"][0]
            width = 0.45 + 0.35 * np.log2(channels + 1)
        else:
            height, width = 1.0, 1.4
        ax.add_patch(Rectangle((x, -height / 2), width, height, facecolor=color, edgecolor="#173c69", lw=1.2))
        ax.text(x + width / 2, height / 2 + 0.22, title, ha="center", va="bottom", fontsize=10, weight="bold")
        ax.text(x + width / 2, -1.55, text, ha="center", va="top", fontsize=9)
        next_x = x + width + 0.55
        if title != "Saída":
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.05, 0),
                    (next_x - 0.05, 0),
                    arrowstyle="->",
                    mutation_scale=12,
                    color="#173c69",
                )
            )
        x = next_x
    ax.text(8, 2.45, "Extração de características aprendida", ha="center", fontsize=9, color="#173c69")
    ax.text(13.6, 2.45, "Classificação", ha="center", fontsize=9, color="#9a3412")
    salvar_figura(fig, output / "arquitetura_cnn.png")


def aumento(config, rgb, output):
    generator = torch.Generator().manual_seed(config["seed"])
    batch = _tensor(rgb).repeat(5, 1, 1, 1)
    augmented = aumentar(batch, config["augmentation"], generator)
    fig, axes = plt.subplots(1, 6, figsize=(8, 1.75))
    axes[0].imshow(rgb)
    axes[0].set_title("Original", fontsize=9)
    for i, ax in enumerate(axes[1:], 1):
        ax.imshow(augmented[i - 1].permute(1, 2, 0).numpy())
        ax.set_title(f"Variação {i}", fontsize=9)
    for ax in axes:
        _limpar(ax)
    fig.tight_layout()
    salvar_figura(fig, output / "aumento.png")


def curvas(config, freeze, output):
    history = pd.read_csv(Path(config["results_dir"]) / "metricas/curvas.csv")
    runs = [f"cnn3_{o}_s{config['seed']}" for o in config["augmentation_options"]] + [freeze["extension"]]
    titles = {runs[0]: "CNN-3 sem aumento", runs[1]: "CNN-3 com aumento", runs[2]: "MobileNetV2"}
    fig, axes = plt.subplots(len(runs), 2, figsize=(8, 2.35 * len(runs)), sharex="row")
    for row, run in zip(axes, runs):
        data = history[history.run == run]
        best = freeze["runs"][run]["best_epoch"]
        for ax, metric, label in zip(row, ("loss", "f1_macro"), ("Perda (entropia cruzada)", "F1 macro")):
            ax.plot(data.epoch, data[f"train_{metric}"], "-o", ms=3, color=AZUL, label="Treino")
            ax.plot(data.epoch, data[f"val_{metric}"], "-s", ms=3, color=LARANJA, label="Validação")
            ax.axvline(best, color="#334155", ls="--", lw=1)
            short = "perda" if metric == "loss" else "F1 macro"
            ax.set(ylabel=label, title=f"{titles[run]}: {short}")
            ax.title.set_fontsize(10)
            if metric == "loss":
                ax.set_yscale("log")
        _virgula(row[1])
        row[1].text(best, row[1].get_ylim()[0], f" época salva: {best}", fontsize=8, va="bottom")
        row[0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Época")
    fig.tight_layout()
    salvar_figura(fig, output / "curvas_treino.png")


def _ativacoes(model, x):
    store = []
    hooks = [
        model.features[i].register_forward_hook(lambda m, i, o: store.append(o.detach())) for i in (2, 6, 10)
    ]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return store


def mapas(model, examples, output):
    """Mapas internos da CNN-3 para o mesmo par de treino usado nas ilustracoes do T1."""
    x = torch.cat([_tensor(rgb) for rgb in examples])
    acts = _ativacoes(model, x)
    cams, logits = grad_cam(model, x)
    # Canais escolhidos pela maior ativacao media no par, sem olhar a classe.
    top = acts[0].mean(dim=(0, 2, 3)).argsort(descending=True)[:2].tolist()
    shared = [
        None,
        acts[0][:, top[0]],
        acts[0][:, top[1]],
        acts[1].mean(1),
        acts[2].mean(1),
    ]
    limits = [None] + [float(a.max()) for a in shared[1:]]
    columns = [
        "Recorte",
        f"Bloco 1, canal {top[0]}",
        f"Bloco 1, canal {top[1]}",
        "Bloco 2 (média)",
        "Bloco 3 (média)",
        "Grad-CAM",
    ]
    fig, axes = plt.subplots(2, 6, figsize=(8.4, 3.4))
    for row, (rgb, label) in enumerate(zip(examples, (0, 1))):
        prob = 1 / (1 + np.exp(-logits[row]))
        panels = [
            rgb,
            acts[0][row, top[0]].numpy(),
            acts[0][row, top[1]].numpy(),
            acts[1][row].mean(0).numpy(),
            acts[2][row].mean(0).numpy(),
        ]
        for col, (ax, panel) in enumerate(zip(axes[row], panels)):
            # Escala comum por coluna: livre e ocupada ficam comparaveis.
            ax.imshow(
                panel,
                cmap=None if col == 0 else "magma",
                vmin=None if col == 0 else 0,
                vmax=None if col == 0 else limits[col],
            )
        axes[row, 5].imshow(rgb)
        axes[row, 5].imshow(cams[row], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{CLASSES[label]}\nP(ocup.)={prob:.2f}", fontsize=9)
        for col, ax in enumerate(axes[row]):
            _limpar(ax)
            if row == 0:
                ax.set_title(columns[col], fontsize=8.5)
    fig.tight_layout()
    salvar_figura(fig, output / "mapas_cnn.png")


# ---------------------------------------------------------------- resultados


def confusoes(reference, predictions, output):
    keys = ["lbp", "combinado", "cnn3"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    y = reference.label.to_numpy()
    for ax, key in zip(axes, keys):
        predicted = predictions[key].predicted.to_numpy()
        matrix = np.array([[((y == i) & (predicted == j)).sum() for j in (0, 1)] for i in (0, 1)])
        share = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]}\n{share[i, j]:.1%}".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if share[i, j] > 0.5 else "black",
                )
        ax.set(
            xticks=[0, 1], yticks=[0, 1], xticklabels=["Livre", "Ocupada"], yticklabels=["Livre", "Ocupada"]
        )
        ax.set_xlabel("Previsão")
        ax.set_title(NOMES[key], fontsize=10)
    axes[0].set_ylabel("Verdadeiro")
    fig.tight_layout()
    salvar_figura(fig, output / "confusao_comparada.png")


def comparacao_ic(config, output):
    table = pd.read_csv(Path(config["results_dir"]) / "metricas/comparacao.csv")
    table = table[table.model != "majoritario"].sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(table))
    colors = [CORES_MODELOS.get(m.split("_s")[0], LARANJA) for m in table.model]
    ax.errorbar(
        table.f1_macro,
        y,
        xerr=[table.f1_macro - table.f1_low, table.f1_high - table.f1_macro],
        fmt="none",
        ecolor="#334155",
        capsize=3,
        lw=1.2,
    )
    markers = ["s" if m in ("hsv", "hog", "lbp", "combinado") else "o" for m in table.model]
    for xi, yi, color, marker in zip(table.f1_macro, y, colors, markers):
        ax.scatter(xi, yi, color=color, marker=marker, s=46, zorder=3)
    for xi, yi, value in zip(table.f1_high, y, table.f1_macro):
        ax.text(xi + 0.004, yi, f"{value:.3f}".replace(".", ","), va="center", fontsize=8.5)
    ax.set(yticks=y, yticklabels=table.name, xlabel="F1 macro no teste (IC 95%, bootstrap por captura)")
    ax.set_xlim(table.f1_low.min() - 0.02, 1.0)
    ax.grid(axis="x", alpha=0.3)
    _virgula(ax, "x")
    ax.text(
        0.01,
        -0.22,
        "Quadrados: métodos clássicos (T1). Círculos: redes neurais (T2).",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    salvar_figura(fig, output / "comparacao_ic.png")


def clima(config, output):
    table = pd.read_csv(Path(config["results_dir"]) / "metricas/clima_comparado.csv")
    keys = ["lbp", "combinado", "cnn3", "mobilenet"]
    weathers = ["SUNNY", "OVERCAST", "RAINY"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.2
    for k, key in enumerate(keys):
        rows = table[table.model == key].set_index("weather").loc[weathers]
        x = np.arange(3) + (k - 1.5) * width
        ax.bar(
            x,
            rows.f1_macro,
            width,
            color=CORES_MODELOS[key],
            hatch=HACHURAS[key],
            edgecolor="white",
            label=NOMES[key],
            yerr=[rows.f1_macro - rows.f1_low, rows.f1_high - rows.f1_macro],
            capsize=2,
            error_kw={"lw": 0.9},
        )
    counts = table.drop_duplicates("weather").set_index("weather").loc[weathers]
    labels = [f"{w}\nn={r.n} ({r.n_livre} livres)" for w, r in counts.iterrows()]
    ax.set(xticks=np.arange(3), xticklabels=labels, ylim=(0.7, 1.02), ylabel="F1 macro (IC 95%)")
    _virgula(ax)
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    salvar_figura(fig, output / "clima_comparado.png")


def _painel(config, rows, path, title_fn, ncols=4, figsize=(8, 6.6)):
    nrows = int(np.ceil(len(rows) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, row in zip(np.atleast_1d(axes).flat, rows):
        ax.imshow(ler_rgb(caminho_seguro(config["data_root"], row["path"])))
        ax.set_title(title_fn(row), fontsize=8)
    fig.tight_layout()
    salvar_figura(fig, path)


def erros(config, prediction, output):
    wrong = prediction[prediction.label != prediction.predicted]
    chosen = pd.concat(
        [g.sample(min(6, len(g)), random_state=config["seed"]) for _, g in wrong.groupby("label")],
        ignore_index=True,
    )
    chosen.to_csv(Path(config["results_dir"]) / "metricas/erros_selecionados_cnn.csv", index=False)
    _painel(
        config,
        chosen.to_dict("records"),
        output / "erros_cnn.png",
        lambda r: (
            f"Real: {CLASSES[r['label']]} | Prev.: {CLASSES[r['predicted']]}\n"
            f"{r['weather']}, câmera {r['camera']}, p={r['probability']:.2f}"
        ),
    )


def discordancias(config, predictions, output):
    cnn, lbp = predictions["cnn3"], predictions["lbp"]
    y = cnn.label
    groups = {
        "CNN-3 acerta, LBP erra": cnn[(cnn.predicted == y) & (lbp.predicted != y)],
        "CNN-3 erra, LBP acerta": cnn[(cnn.predicted != y) & (lbp.predicted == y)],
    }
    rows, selected = [], []
    for title, group in groups.items():
        sample = group.sample(min(6, len(group)), random_state=config["seed"])
        sample = sample.assign(grupo=title)
        selected.append(sample)
        rows.extend(sample.to_dict("records") + [None] * (6 - len(sample)))
    pd.concat(selected).to_csv(
        Path(config["results_dir"]) / "metricas/discordancias_selecionadas.csv", index=False
    )
    fig, axes = plt.subplots(2, 6, figsize=(8.6, 3.7))
    for ax, row in zip(axes.flat, rows):
        ax.axis("off")
        if row is None:
            continue
        ax.imshow(ler_rgb(caminho_seguro(config["data_root"], row["path"])))
        ax.set_title(f"Real: {CLASSES[row['label']]}\n{row['weather']}, câm. {row['camera']}", fontsize=7.5)
    for ax, title in zip(axes[:, 0], groups):
        ax.text(
            -0.12,
            0.5,
            title.replace(", ", ",\n"),
            transform=ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=8.5,
        )
    fig.tight_layout()
    salvar_figura(fig, output / "discordancias.png")


def gradcam_teste(config, model, prediction, output):
    """Dois exemplos por celula da matriz de confusao, sorteados com semente fixa."""
    cells = [(0, 0), (1, 1), (0, 1), (1, 0)]
    rows = []
    for label, predicted in cells:
        group = prediction[(prediction.label == label) & (prediction.predicted == predicted)]
        rows.extend(group.sample(min(2, len(group)), random_state=config["seed"]).to_dict("records"))
    images = [ler_rgb(caminho_seguro(config["data_root"], r["path"]), config["image_size"]) for r in rows]
    cams, _ = grad_cam(model, torch.cat([_tensor(i) for i in images]))
    fig, axes = plt.subplots(2, 8, figsize=(8.6, 2.9))
    for k, (row, rgb) in enumerate(zip(rows, images)):
        top, bottom = axes[0, k], axes[1, k]
        top.imshow(rgb)
        bottom.imshow(rgb)
        bottom.imshow(cams[k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        status = "acerto" if row["label"] == row["predicted"] else "erro"
        top.set_title(f"Real: {CLASSES[row['label']]}\n{status}, p={row['probability']:.2f}", fontsize=7.5)
        _limpar(top)
        _limpar(bottom)
    axes[0, 0].set_ylabel("Recorte", fontsize=8.5)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8.5)
    fig.tight_layout()
    salvar_figura(fig, output / "gradcam.png")
    pd.DataFrame(rows).to_csv(Path(config["results_dir"]) / "metricas/gradcam_exemplos.csv", index=False)


def generalizacao(config, output):
    table = pd.read_csv(Path(config["results_dir"]) / "metricas/generalizacao_cameras.csv")
    keys = ["lbp", "combinado", "cnn3"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2), sharey=True)
    for ax, key in zip(axes, keys):
        rows = table[table.model == key].sort_values("camera")
        x = rows.camera.to_numpy()
        ax.vlines(x, rows.f1_nova, rows.f1_vista, color="#cbd5e1", lw=2)
        ax.scatter(
            x, rows.f1_vista, color=CORES_MODELOS[key], marker="o", s=26, label="câmera vista", zorder=3
        )
        ax.scatter(
            x,
            rows.f1_nova,
            facecolor="white",
            edgecolor=CORES_MODELOS[key],
            marker="D",
            s=26,
            label="câmera nova",
            zorder=3,
        )
        ax.set(xticks=x, title=NOMES[key], xlabel="Câmera do teste")
        ax.title.set_fontsize(10)
        ax.grid(axis="y", alpha=0.3)
        _virgula(ax)
    axes[0].set_ylabel("F1 macro")
    # Legenda neutra abaixo dos paineis para nao cobrir os pontos da camera 1.
    handles = [
        plt.Line2D([], [], color="#475569", marker="o", ls="", label="câmera vista no treino"),
        plt.Line2D(
            [], [], color="#475569", marker="D", mfc="white", ls="", label="câmera retirada do treino"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    salvar_figura(fig, output / "generalizacao.png")


# ---------------------------------------------------------------- anotacao cega


def montagens_anotacao(config, reference, predictions):
    """Mostra a uniao dos erros (LBP ou CNN-3) sem indicar qual modelo errou."""
    y = reference.label.to_numpy()
    wrong = (predictions["lbp"].predicted.to_numpy() != y) | (predictions["cnn3"].predicted.to_numpy() != y)
    union = reference[wrong][["id", "path", "label", "weather", "camera"]].sample(
        frac=1, random_state=config["seed"]
    )
    union.insert(0, "item", np.arange(1, len(union) + 1))
    output = Path(config["results_dir"]) / "anotacao"
    output.mkdir(parents=True, exist_ok=True)
    union.to_csv(output / "itens.csv", index=False)
    per_page = 24
    for start in range(0, len(union), per_page):
        page = union.iloc[start : start + per_page]
        fig, axes = plt.subplots(4, 6, figsize=(12, 9))
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, page.itertuples(index=False)):
            ax.imshow(ler_rgb(caminho_seguro(config["data_root"], row.path)))
            ax.set_title(
                f"#{row.item} real: {CLASSES[row.label]}\n{row.weather}, câm. {row.camera}", fontsize=9
            )
        fig.tight_layout()
        salvar_figura(fig, output / f"lote_{start // per_page + 1:02d}.png")
    salvar_json(
        output / "protocolo.json",
        {"itens": len(union), "cego_quanto_ao_modelo": True, "seed": config["seed"]},
    )
