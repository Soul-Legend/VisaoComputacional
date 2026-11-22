"""Redes neurais do T2: cache de pixels, arquiteturas, aumento de dados e laco de treino.

As imagens passam pela mesma leitura do T1 (RGB, 128 x 128, interpolacao de area). A
normalizacao fica dentro do modelo para que predicao, Grad-CAM e tempos usem o mesmo caminho.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .configuracao import carregar_config, salvar_json, sha256_file
from .dados import caminho_seguro, carregar_manifesto
from .preprocessamento import ler_rgb

IMPLEMENTACAO_IMAGENS = 1
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)


def carregar_config_t2(path):
    config = carregar_config(path)
    config["t1_config"] = str((Path(config["project_root"]) / config["t1_config"]).resolve())
    return config


def configurar_torch(config, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    # Mesma semente e mesmas threads devem reproduzir os pesos na mesma maquina.
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------- cache de pixels


def assinatura_imagens(config):
    value = {
        "manifest_sha256": sha256_file(config["manifest"]),
        "image_size": list(config["image_size"]),
        "implementation": IMPLEMENTACAO_IMAGENS,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preparar_imagens(config):
    """Le uma vez cada recorte do manifesto e mede o tempo de preparo por particao."""
    frame = carregar_manifesto(config)
    destination = Path(config["processed_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    path, metadata_path = destination / "imagens.npz", destination / "imagens.json"
    if path.exists() and metadata_path.exists():
        carregar_imagens(config, frame)
        print("Cache de imagens valido", flush=True)
        return
    height, width = config["image_size"][1], config["image_size"][0]
    images = np.empty((len(frame), height, width, 3), dtype=np.uint8)
    elapsed = {split: 0.0 for split in frame.split.unique()}
    for i, row in enumerate(frame.itertuples(index=False)):
        start = perf_counter()
        images[i] = ler_rgb(caminho_seguro(config["data_root"], row.path), config["image_size"])
        elapsed[row.split] += perf_counter() - start
        if (i + 1) % 2000 == 0:
            print(f"Imagens lidas: {i + 1}/{len(frame)}", flush=True)
    np.savez(path, images=images, ids=frame.id.to_numpy().astype(str))
    salvar_json(
        metadata_path,
        {
            "signature": assinatura_imagens(config),
            "npz_sha256": sha256_file(path),
            "read_seconds": elapsed,
            "n": frame.split.value_counts().to_dict(),
        },
    )


def carregar_imagens(config, frame):
    """Devolve tensor uint8 NCHW na ordem do manifesto, apos verificar identidade."""
    destination = Path(config["processed_dir"])
    metadata = json.loads((destination / "imagens.json").read_text(encoding="utf-8"))
    if metadata["signature"] != assinatura_imagens(config):
        raise ValueError("Cache de imagens incompativel com manifesto ou tamanho")
    if sha256_file(destination / "imagens.npz") != metadata["npz_sha256"]:
        raise ValueError("Cache de imagens alterado")
    content = np.load(destination / "imagens.npz")
    if not np.array_equal(content["ids"], frame.id.to_numpy().astype(str)):
        raise ValueError("Ordem do cache difere do manifesto")
    return torch.from_numpy(content["images"]).permute(0, 3, 1, 2).contiguous(), metadata


def estatisticas_canal(images, indices):
    """Media e desvio por canal no intervalo [0, 1], calculados so nos indices de treino."""
    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros(3, dtype=torch.float64)
    count = 0
    for batch in torch.as_tensor(indices).split(1024):
        x = images[batch].double() / 255
        total += x.sum(dim=(0, 2, 3))
        squares += (x**2).sum(dim=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / count
    return mean.tolist(), (squares / count - mean**2).sqrt().tolist()


# ---------------------------------------------------------------- arquiteturas


class Normalizar(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class CNN3(nn.Module):
    """Tres blocos convolucionais e uma camada densa; saida e um logit de ocupacao."""

    def __init__(self, mean, std, filters=(32, 64, 128), dense=128, dropout=0.5, size=128):
        super().__init__()
        self.normalizar = Normalizar(mean, std)
        layers, channels = [], 3
        for out in filters:
            layers += [
                nn.Conv2d(channels, out, 3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            channels = out
        self.features = nn.Sequential(*layers)
        side = size // 2 ** len(filters)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * side * side, dense),
            nn.ReLU(),
            nn.Linear(dense, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(self.normalizar(x))).squeeze(1)

    def camada_cam(self):
        # Ultima ReLU antes do pooling final: mapa 32 x 32 com informacao espacial.
        return self.features[-2]


class MobileNetBinaria(nn.Module):
    def __init__(self, weights_name=None):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights[weights_name] if weights_name else None
        self.normalizar = Normalizar(MEDIA_IMAGENET, DESVIO_IMAGENET)
        self.rede = mobilenet_v2(weights=weights)
        self.rede.classifier[1] = nn.Linear(self.rede.last_channel, 1)

    def forward(self, x):
        return self.rede(self.normalizar(x)).squeeze(1)

    def camada_cam(self):
        return self.rede.features[-1]


def construir(config, arquitetura, mean=None, std=None, pretrained=True):
    if arquitetura == "cnn3":
        params = config["cnn"]
        return CNN3(
            mean,
            std,
            filters=tuple(params["filters"]),
            dense=params["dense"],
            dropout=params["dropout"],
            size=config["image_size"][0],
        )
    if arquitetura == "mobilenet":
        return MobileNetBinaria(config["mobilenet"]["weights"] if pretrained else None)
    raise ValueError(f"Arquitetura desconhecida: {arquitetura}")


def contar_parametros(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- aumento de dados


def aumentar(x, params, generator):
    """Transformacoes moderadas e plausiveis para uma camera fixa; sem espelhamento."""
    n = x.shape[0]

    def uniforme(low, high):
        return low + (high - low) * torch.rand(n, generator=generator)

    angle = uniforme(-params["rotation_deg"], params["rotation_deg"]) * math.pi / 180
    scale = uniforme(*params["zoom"])
    # Em coordenadas normalizadas [-1, 1], deslocar uma fracao t da largura equivale a 2t.
    tx = uniforme(-params["translate"], params["translate"]) * 2
    ty = uniforme(-params["translate"], params["translate"]) * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    x = x * uniforme(*params["brightness"]).view(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * uniforme(*params["contrast"]).view(n, 1, 1, 1) + mean
    return x.clamp(0, 1)


# ---------------------------------------------------------------- treino e predicao


def para_float(images, indices):
    return images[indices].float() / 255


@torch.no_grad()
def predizer_logits(model, images, indices, batch_size=256):
    model.eval()
    outputs = [model(para_float(images, batch)) for batch in torch.as_tensor(indices).split(batch_size)]
    return torch.cat(outputs).numpy() if outputs else np.empty(0, dtype=np.float32)


def resumo(labels, logits):
    labels = np.asarray(labels)
    predicted = (np.asarray(logits) > 0).astype(int)
    loss = F.binary_cross_entropy_with_logits(
        torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32)
    ).item()
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1_macro": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
    }


def chave_epoca(row):
    return (-round(row["val_f1_macro"], 4), row["val_loss"], row["epoch"])


def treinar_rede(config, arquitetura, images, labels, train_idx, val_idx, augment, seed, nome=""):
    """Treina com parada antecipada pela validacao e devolve o melhor modelo e o historico."""
    configurar_torch(config, seed)
    params = config["cnn"] if arquitetura == "cnn3" else config["mobilenet"]
    mean, std = estatisticas_canal(images, train_idx)
    model = construir(config, arquitetura, mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.as_tensor(train_idx), torch.as_tensor(val_idx)
    y_all = torch.as_tensor(labels, dtype=torch.float32)
    history, best, best_key, best_epoch = [], None, None, 0
    started = perf_counter()
    for epoch in range(1, params["max_epochs"] + 1):
        epoch_start = perf_counter()
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        train_logits, train_labels = [], []
        for batch in order.split(config["batch_size"]):
            x = para_float(images, batch)
            if augment:
                x = aumentar(x, config["augmentation"], generator)
            y = y_all[batch]
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_logits.append(logits.detach())
            train_labels.append(y)
        # Metricas de treino da passada online (com dropout e aumento), como em Keras.
        train = resumo(torch.cat(train_labels).numpy(), torch.cat(train_logits).numpy())
        val = resumo(labels[val_idx.numpy()], predizer_logits(model, images, val_idx))
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"val_{k}": v for k, v in val.items()},
            "seconds": perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"{nome} epoca {epoch}: perda {train['loss']:.4f}/{val['loss']:.4f}, "
            f"F1 {train['f1_macro']:.4f}/{val['f1_macro']:.4f}, {row['seconds']:.0f}s",
            flush=True,
        )
        if best_key is None or chave_epoca(row) < best_key:
            best_key, best_epoch = chave_epoca(row), epoch
            best = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= params["patience"]:
            break
    model.load_state_dict(best)
    info = {
        "architecture": arquitetura,
        "augmentation": bool(augment),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": perf_counter() - started,
        "parameters": contar_parametros(model),
        "channel_mean": mean,
        "channel_std": std,
        "best": {k: v for k, v in history[best_epoch - 1].items()},
    }
    return model, history, info


def carregar_modelo(config, info, path):
    model = construir(config, info["architecture"], info["channel_mean"], info["channel_std"], pretrained=False)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def executar(config, nome, arquitetura, images, labels, train_idx, val_idx, augment, seed, subdir=""):
    """Treina uma rodada ou reaproveita a ja salva, verificando o hash dos pesos."""
    models_dir = Path(config["models_dir"]) / subdir
    record_dir = Path(config["results_dir"]) / "treino" / subdir
    model_path, record_path = models_dir / f"{nome}.pt", record_dir / f"{nome}.json"
    if model_path.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if sha256_file(model_path) != record["model_sha256"]:
            raise ValueError(f"Pesos de {nome} mudaram depois do treino")
        print(f"Rodada reaproveitada: {nome}", flush=True)
        return carregar_modelo(config, record, model_path), record
    model, history, info = treinar_rede(
        config, arquitetura, images, labels, train_idx, val_idx, augment, seed, nome
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    record = {"name": nome, **info, "model_file": model_path.name, "model_sha256": sha256_file(model_path)}
    record["history"] = history
    salvar_json(record_path, record)
    return model, record


# ---------------------------------------------------------------- Grad-CAM


def grad_cam(model, x):
    """Mapa de relevancia da classe prevista para um lote x em [0, 1] (N, 3, H, W)."""
    model.eval()
    store = {}
    layer = model.camada_cam()
    forward = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    backward = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            # Para previsao "livre", explicar o logit negativo (evidencia contra ocupacao).
            target = torch.where(logits > 0, logits, -logits).sum()
            model.zero_grad()
            target.backward()
    finally:
        forward.remove()
        backward.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.flatten(1).max(dim=1).values.view(-1, 1, 1).clamp_min(1e-8)
    return (cam / peak).detach().numpy(), logits.detach().numpy()
