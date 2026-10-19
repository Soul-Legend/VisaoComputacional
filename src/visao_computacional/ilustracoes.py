"""Exemplos didaticos dos descritores reais, sem reabrir o experimento de teste."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog, local_binary_pattern

from .configuracao import salvar_json, sha256_file
from .dados import caminho_seguro, carregar_manifesto
from .descritores import vetor
from .preprocessamento import cinza, ler_rgb
from .visualizacao import CORES, NOMES, estilo, salvar_figura


def selecionar_exemplos(frame):
    """Prefere o par inspecionado; em outro estudo procura uma vaga com ambas as classes."""
    train = frame[frame.split == "treino"].copy()
    preferred = train[(train.camera == 6) & (train.slot == 204) & (train.date == "2015-11-12")]
    if set(preferred.label) == {0, 1}:
        pair = preferred
    else:
        groups = train.groupby(["camera", "slot", "date"], sort=True)
        eligible = groups.label.nunique()
        if not (eligible == 2).any():
            raise ValueError(
                "Ilustracoes exigem uma vaga de treino com exemplos das duas classes no mesmo dia"
            )
        pair = groups.get_group(eligible[eligible == 2].index[0])
    return pair.sort_values("path").groupby("label", sort=True).head(1).sort_values("label")


def mostrar(ax, array, title, **kwargs):
    handle = ax.imshow(array, **kwargs)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    return handle


def ilustracoes(config):
    estilo()
    frame = carregar_manifesto(config)
    pair = selecionar_exemplos(frame)
    images = []
    for row in pair.itertuples(index=False):
        path = caminho_seguro(config["data_root"], row.path)
        if sha256_file(path) != row.sha256:
            raise ValueError("Imagem ilustrativa mudou desde a auditoria")
        images.append(ler_rgb(path, config["image_size"]))
    output = Path(config["project_root"]) / "docs/imagens"
    output.mkdir(parents=True, exist_ok=True)
    salvar_json(
        output / "exemplos_metodos.json",
        {
            "source": "CNRPark+EXT / CNR-EXT; http://cnrpark.it/; ODbL-1.0",
            "purpose": "Ilustracao dos descritores, sem avaliacao ou ajuste de modelos",
            "manifest_sha256": sha256_file(config["manifest"]),
            "examples": pair[["id", "path", "label", "split", "camera", "slot", "date", "sha256"]].to_dict(
                "records"
            ),
            "image_size": config["image_size"],
        },
    )
    _hsv(images, config, output)
    _lbp(images, config, output)
    _hog(images, config, output)
    _fluxo(output)


def _hsv(images, config, output):
    fig = plt.figure(figsize=(10, 6.6), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.85])
    for label, rgb in enumerate(images):
        axes = grid[label].subgridspec(1, 4).subplots()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mostrar(axes[0], rgb, f"{NOMES[label]}: recorte RGB")
        mostrar(axes[1], hsv[:, :, 0], "H: matiz (cor)", cmap="hsv", vmin=0, vmax=180)
        mostrar(axes[2], hsv[:, :, 1], "S: saturação", cmap="gray", vmin=0, vmax=255)
        mostrar(axes[3], hsv[:, :, 2], "V: intensidade", cmap="gray", vmin=0, vmax=255)
    axes = grid[2].subgridspec(1, 3).subplots()
    for channel, (ax, title) in enumerate(zip(axes, ("H: matiz", "S: saturação", "V: intensidade"))):
        for label, rgb in enumerate(images):
            bins = config["hist_bins"]
            hist = vetor(rgb, "hsv", config).reshape(3, bins)[channel]
            ax.stairs(hist, np.arange(bins + 1), color=CORES[label], linewidth=1.8, label=NOMES[label])
        ax.set(title=title, xlabel="Bin do histograma", ylabel="Fração de pixels", xlim=(0, bins))
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.tick_params(labelsize=9)
    axes[0].legend(fontsize=9)
    fig.suptitle("HSV: dos canais de cor aos histogramas de 16 bins", fontsize=14)
    salvar_figura(fig, output / "metodo_hsv.png")


def _lbp(images, config, output):
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), layout="constrained")
    for label, rgb in enumerate(images):
        gray = cinza(rgb)
        codes = local_binary_pattern(gray, config["lbp_points"], config["lbp_radius"], method="uniform")
        mostrar(axes[label, 0], rgb, f"{NOMES[label]}: RGB")
        mostrar(axes[label, 1], gray, "Escala de cinza", cmap="gray", vmin=0, vmax=255)
        handle = mostrar(
            axes[label, 2],
            codes,
            "Códigos LBP uniformes",
            cmap="viridis",
            vmin=0,
            vmax=9,
            interpolation="nearest",
        )
        histogram = vetor(rgb, "lbp", config)
        axes[label, 3].bar(np.arange(len(histogram)), histogram, color=CORES[label], width=0.8)
        axes[label, 3].set(title="Histograma de textura", xlabel="Código LBP", ylabel="Fração de pixels")
        axes[label, 3].set_xticks([0, 2, 4, 6, 8, 9])
        axes[label, 3].tick_params(labelsize=9)
    fig.colorbar(handle, ax=axes[:, 2], shrink=0.6, ticks=[0, 4, 8, 9], label="Código; 9 = não uniforme")
    fig.suptitle("LBP: vizinhanças locais resumidas em dez frequências", fontsize=14)
    salvar_figura(fig, output / "metodo_lbp.png")


def _hog(images, config, output):
    gradients, glyphs = [], []
    for rgb in images:
        gray = cinza(rgb)
        row, col = np.zeros_like(gray, dtype=float), np.zeros_like(gray, dtype=float)
        gray_float = gray.astype(float)
        row[1:-1, :] = gray_float[2:, :] - gray_float[:-2, :]
        col[:, 1:-1] = gray_float[:, 2:] - gray_float[:, :-2]
        gradients.append(np.hypot(row, col))
        features, drawing = hog(
            gray,
            orientations=config["hog_orientations"],
            pixels_per_cell=tuple(config["hog_pixels_per_cell"]),
            cells_per_block=tuple(config["hog_cells_per_block"]),
            block_norm="L2-Hys",
            visualize=True,
        )
        np.testing.assert_allclose(features, vetor(rgb, "hog", config), rtol=1e-6, atol=1e-7)
        glyphs.append(drawing)
    gradient_max = np.percentile(np.stack(gradients), 99)
    glyph_max = np.percentile(np.stack(glyphs), 99)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), layout="constrained")
    for label, rgb in enumerate(images):
        mostrar(axes[label, 0], rgb, f"{NOMES[label]}: recorte RGB")
        mostrar(
            axes[label, 1],
            gradients[label],
            "Intensidade dos gradientes",
            cmap="gray",
            vmin=0,
            vmax=max(gradient_max, 1),
        )
        mostrar(
            axes[label, 2],
            glyphs[label],
            "Orientações por célula (HOG)",
            cmap="gray",
            vmin=0,
            vmax=max(glyph_max, 1),
        )
    fig.suptitle("HOG: variações de intensidade e orientações locais", fontsize=14)
    salvar_figura(fig, output / "metodo_hog.png")


def _fluxo(output):
    fig, ax = plt.subplots(figsize=(10, 1.25))
    ax.set(xlim=(0, 10), ylim=(0, 1))
    ax.axis("off")
    labels = [
        "Recorte RGB\n128 × 128",
        "HSV / LBP / HOG\nou concatenação",
        "Padronização\nStandardScaler",
        "SVM linear\nfronteira aprendida",
        "Livre (0)\nou ocupada (1)",
    ]
    for i, label in enumerate(labels):
        x = 0.95 + 2 * i
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=.5", "facecolor": "#e7eef7", "edgecolor": "#173c69"},
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(x + 1.2, 0.5),
                xytext=(x + 0.78, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#173c69"},
            )
    fig.tight_layout()
    salvar_figura(fig, output / "fluxo_classificacao.png")
