"""Relatorio final unico gerado das evidencias congeladas das duas etapas (classica e redes).

Todos os numeros vem dos arquivos de resultados; o texto interpretativo fica em
relatorios/final/modelo.md. Um marcador nao preenchido interrompe a geracao.
"""

import json
import os
from pathlib import Path

import pandas as pd

from .comparacao import NOMES
from .configuracao import carregar_config
from .relatorio import preencher, renderizar_pdf, substituicoes_t1, tabela


def br(value, digits=3):
    """Numero com virgula decimal, como no restante do texto em portugues."""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}%".replace(".", ",")


def milhar(value):
    return f"{int(value):,}".replace(",", ".")


def intervalo(row, prefix="f1"):
    return f"[{br(row[f'{prefix}_low'])}; {br(row[f'{prefix}_high'])}]"


def substituicoes_t2(config, report_dir):
    base = Path(config["results_dir"])
    metrics = base / "metricas"
    read_json = lambda name: json.loads((metrics / name).read_text(encoding="utf-8"))
    freeze, evaluation = read_json("congelamento.json"), read_json("avaliacao.json")
    comparison = read_json("comparacao.json")
    selection = pd.read_csv(metrics / "selecao.csv")
    test = pd.read_csv(metrics / "teste.csv").set_index("model")
    summary = pd.read_csv(metrics / "comparacao.csv").set_index("model")
    pairs = pd.read_csv(metrics / "pares.csv")
    climate = pd.read_csv(metrics / "clima_comparado.csv")
    overlap = pd.read_csv(metrics / "sobreposicao_erros.csv").set_index("classico")
    config_t1 = carregar_config(config["t1_config"])
    t1_test = pd.read_csv(Path(config_t1["results_dir"]) / "metricas/teste.csv").set_index("descriptor")
    t1_freeze = json.loads(
        (Path(config_t1["results_dir"]) / "metricas/congelamento.json").read_text(encoding="utf-8")
    )
    primary, extension = freeze["primary"], freeze["extension"]
    runs = freeze["runs"]
    r = {}

    # Escolha da linha de base classica: melhor C de cada representacao na validacao.
    # Os resultados de teste entram apenas na Tabela 4, junto com as redes.
    r["tabela_selecao_classica"] = tabela(
        ["Representação", "Dimensão", "C escolhido", "F1 macro val.", "Ajuste (s)"],
        [
            [
                NOMES[model["descriptor"]].replace(" + SVM", ""),
                milhar(model["dimension"]),
                f"{model['C']:g}".replace(".", ","),
                br(model["validation_f1_macro"], 4),
                br(model["training_seconds"], 3),
            ]
            for model in t1_freeze["models"]
        ],
    )

    # Selecao e treino
    chosen = selection.set_index("augmentation")
    r["aumento_escolhido"] = "com aumento" if freeze["augmentation"] == "com" else "sem aumento"
    for option in ("sem", "com"):
        r[f"val_f1_{option}"] = br(chosen.loc[option, "validation_f1_macro"], 4)
        r[f"epoca_{option}"] = str(int(chosen.loc[option, "best_epoch"]))
        r[f"epocas_rodadas_{option}"] = str(int(chosen.loc[option, "epochs_run"]))
    r["tabela_selecao"] = tabela(
        ["Configuração", "Época salva", "Épocas rodadas", "F1 macro val.", "Perda val.", "Treino (min)"],
        [
            [
                "CNN-3 " + ("com aumento" if row.augmentation == "com" else "sem aumento"),
                int(row.best_epoch),
                int(row.epochs_run),
                br(row.validation_f1_macro, 4),
                br(row.validation_loss, 4),
                br(row.training_seconds / 60, 1),
            ]
            for row in selection.itertuples()
        ],
    )
    mobile = runs[extension]
    r["val_f1_mobilenet"] = br(mobile["best"]["val_f1_macro"], 4)
    r["epoca_mobilenet"] = str(mobile["best_epoch"])
    r["parametros_cnn"] = milhar(runs[primary]["parameters"])
    r["parametros_mobilenet"] = milhar(mobile["parameters"])
    seeds = [runs[name] for name in freeze["seeds"]]
    r["val_f1_sementes"] = "; ".join(br(s["best"]["val_f1_macro"], 4) for s in seeds)
    r["epocas_sementes"] = ", ".join(str(s["best_epoch"]) for s in seeds)

    # Teste principal
    cnn, mob = summary.loc["cnn3"], summary.loc["mobilenet"]
    lbp, comb = summary.loc["lbp"], summary.loc["combinado"]
    for key, row in (("cnn", cnn), ("mobilenet", mob), ("lbp", lbp), ("combinado", comb)):
        r[f"f1_{key}"] = br(row.f1_macro, 4)
        r[f"acc_{key}"] = br(row.accuracy, 4)
        r[f"ic_{key}"] = intervalo(row)
    seed_f1 = test.loc[freeze["seeds"], "f1_macro"]
    r["f1_sementes_media"] = br(seed_f1.mean(), 4)
    r["f1_sementes_dp"] = br(seed_f1.std(ddof=1), 4)
    r["f1_sementes_min"] = br(seed_f1.min(), 4)
    r["f1_sementes_max"] = br(seed_f1.max(), 4)
    r["tabela_comparacao"] = tabela(
        ["Modelo", "Abordagem", "Acurácia", "F1 macro", "IC 95% do F1 macro"],
        [
            [
                NOMES[key],
                "Clássica" if key in t1_test.index else "Rede neural",
                br(row.accuracy, 4),
                br(row.f1_macro, 4),
                "—" if key == "majoritario" else intervalo(row),
            ]
            for key, row in summary.loc[
                ["majoritario", "hsv", "hog", "lbp", "combinado", "cnn3", "cnn3_s43", "cnn3_s44", "mobilenet"]
            ].iterrows()
        ],
    )
    pair_names = {
        ("cnn3", "lbp"): "CNN-3 − LBP",
        ("cnn3", "combinado"): "CNN-3 − HSV+LBP+HOG",
        ("mobilenet", "cnn3"): "MobileNetV2 − CNN-3",
        ("mobilenet", "combinado"): "MobileNetV2 − HSV+LBP+HOG",
    }
    r["tabela_pares"] = tabela(
        ["Comparação", "Diferença de F1 macro", "IC 95%", "Só A acerta", "Só B acerta", "p (McNemar)"],
        [
            [
                pair_names[(row.a, row.b)],
                br(row.f1_diff, 4),
                f"[{br(row.diff_low, 4)}; {br(row.diff_high, 4)}]",
                row.only_a_correct,
                row.only_b_correct,
                "< 0,0001" if row.mcnemar_p < 1e-4 else br(row.mcnemar_p, 4),
            ]
            for row in pairs.itertuples()
        ],
    )
    for row in pairs.itertuples():
        tag = f"{row.a}_{row.b}"
        r[f"diff_{tag}"] = br(row.f1_diff, 4)
        r[f"diff_ic_{tag}"] = f"[{br(row.diff_low, 4)}; {br(row.diff_high, 4)}]"
        r[f"so_a_{tag}"] = str(row.only_a_correct)
        r[f"so_b_{tag}"] = str(row.only_b_correct)
        r[f"p_{tag}"] = "< 0,0001" if row.mcnemar_p < 1e-4 else br(row.mcnemar_p, 4)
    r["clusters"] = str(comparison["clusters"])
    r["reamostragens"] = milhar(comparison["resamples"])

    # Classes
    class_rows = []
    for key in ("lbp", "combinado", "cnn3", "mobilenet"):
        source = (
            t1_test.loc[key] if key in t1_test.index else test.loc[primary if key == "cnn3" else extension]
        )
        class_rows.append(
            [
                NOMES[key],
                *[
                    br(source[k])
                    for k in (
                        "precision_livre",
                        "recall_livre",
                        "f1_livre",
                        "precision_ocupada",
                        "recall_ocupada",
                        "f1_ocupada",
                    )
                ],
            ]
        )
    r["tabela_classes"] = tabela(
        ["Modelo", "P livre", "R livre", "F1 livre", "P ocup.", "R ocup.", "F1 ocup."], class_rows
    )
    cnn_pred = pd.read_csv(base / "predicoes" / f"{primary}.csv")
    r["cnn_ocupada_livre"] = str(int(((cnn_pred.label == 1) & (cnn_pred.predicted == 0)).sum()))
    r["cnn_livre_ocupada"] = str(int(((cnn_pred.label == 0) & (cnn_pred.predicted == 1)).sum()))
    r["cnn_erros"] = str(int((cnn_pred.label != cnn_pred.predicted).sum()))
    lbp_pred = pd.read_csv(Path(config_t1["results_dir"]) / "predicoes/lbp.csv")
    r["lbp_erros"] = str(int((lbp_pred.label != lbp_pred.predicted).sum()))
    alternative = test.loc[[n for n in selection.run if n != primary]]
    r["f1_alternativa"] = br(alternative.f1_macro.iloc[0], 4)
    r["recall_ocupada_cnn"] = br(test.loc[primary, "recall_ocupada"], 4)
    r["recall_livre_cnn"] = br(test.loc[primary, "recall_livre"], 4)

    # Custo
    t1_processed = Path(config_t1["processed_dir"])
    images_meta = json.loads((Path(config["processed_dir"]) / "imagens.json").read_text(encoding="utf-8"))
    cost_rows = []
    for key in ("lbp", "combinado"):
        meta = json.loads((t1_processed / f"{key}.json").read_text(encoding="utf-8"))
        row = t1_test.loc[key]
        prep_train = meta["extraction_seconds"]["treino"]
        cost_rows.append(
            [
                NOMES[key],
                milhar(row.dimension),
                br(prep_train, 1),
                br(row.training_seconds, 3),
                br(row.extraction_ms_per_image, 3),
                br(row.inference_ms_per_image, 4),
                br(row.extraction_ms_per_image + row.inference_ms_per_image, 3),
            ]
        )
    for key, run in (("cnn3", primary), ("mobilenet", extension)):
        row = test.loc[run]
        cost_rows.append(
            [
                NOMES[key],
                milhar(row.parameters),
                br(images_meta["read_seconds"]["treino"], 1),
                br(row.training_seconds, 0),
                br(row.preparation_ms_per_image, 3),
                br(row.inference_ms_per_image, 3),
                br(row.preparation_ms_per_image + row.inference_ms_per_image, 3),
            ]
        )
    r["tabela_custo"] = tabela(
        [
            "Modelo",
            "Dim./parâm.",
            "Preparo treino (s)",
            "Ajuste (s)",
            "Preparo ms/img",
            "Predição ms/img",
            "Total ms/img",
        ],
        cost_rows,
    )
    r["treino_cnn_min"] = br(test.loc[primary, "training_seconds"] / 60, 1)
    r["treino_mobilenet_min"] = br(test.loc[extension, "training_seconds"] / 60, 1)
    r["treino_lbp_s"] = br(t1_test.loc["lbp", "training_seconds"], 3)
    r["treino_combinado_s"] = br(t1_test.loc["combinado", "training_seconds"], 2)
    total_cnn = test.loc[primary, "preparation_ms_per_image"] + test.loc[primary, "inference_ms_per_image"]
    total_lbp = t1_test.loc["lbp", "extraction_ms_per_image"] + t1_test.loc["lbp", "inference_ms_per_image"]
    total_comb = (
        t1_test.loc["combinado", "extraction_ms_per_image"]
        + t1_test.loc["combinado", "inference_ms_per_image"]
    )
    r["total_cnn_ms"], r["total_lbp_ms"], r["total_comb_ms"] = br(total_cnn), br(total_lbp), br(total_comb)
    r["inferencia_cnn_ms"] = br(test.loc[primary, "inference_ms_per_image"])
    r["inferencia_mobilenet_ms"] = br(test.loc[extension, "inference_ms_per_image"])
    search_t2 = sum(rec["training_seconds"] for rec in runs.values())
    r["busca_t2_min"] = br(search_t2 / 60, 0)
    t1_validation = pd.read_csv(Path(config_t1["results_dir"]) / "metricas/validacao.csv")
    r["busca_t1_s"] = br(t1_validation.training_seconds.sum(), 0)

    # Clima
    weathers = ["SUNNY", "OVERCAST", "RAINY"]
    pivot = climate.set_index(["weather", "model"])
    r["tabela_clima"] = tabela(
        ["Clima", "N (livre/ocup.)", "LBP", "HSV+LBP+HOG", "CNN-3", "MobileNetV2"],
        [
            [
                w,
                (
                    f"{int(pivot.loc[(w, 'cnn3'), 'n'])} ({int(pivot.loc[(w, 'cnn3'), 'n_livre'])}/"
                    f"{int(pivot.loc[(w, 'cnn3'), 'n_ocupada'])})"
                ),
                *[br(pivot.loc[(w, k), "f1_macro"]) for k in ("lbp", "combinado", "cnn3", "mobilenet")],
            ]
            for w in weathers
        ],
    )
    for w in weathers:
        for k in ("lbp", "combinado", "cnn3", "mobilenet"):
            row = pivot.loc[(w, k)]
            r[f"clima_{k}_{w.lower()}"] = br(row.f1_macro)
            r[f"clima_ic_{k}_{w.lower()}"] = intervalo(row)

    # Sobreposicao de erros
    for other in ("lbp", "combinado"):
        row = overlap.loc[other]
        for field in ("ambos_corretos", "so_cnn_errou", "so_classico_errou", "ambos_erraram"):
            r[f"{field}_{other}"] = str(int(row[field]))

    # Generalizacao
    general = metrics / "generalizacao.csv"
    if general.exists():
        pooled = pd.read_csv(general).set_index("model")
        per_camera = pd.read_csv(metrics / "generalizacao_cameras.csv")
        loco_pairs = pd.read_csv(metrics / "generalizacao_pares.csv").set_index("b")
        r["tabela_generalizacao"] = tabela(
            [
                "Modelo",
                "F1 câmera vista",
                "F1 câmera nova",
                "IC 95% (nova)",
                "Diferença",
                "IC 95% da diferença",
            ],
            [
                [
                    NOMES[k],
                    br(row.f1_vista, 4),
                    br(row.f1_nova, 4),
                    f"[{br(row.f1_nova_low)}; {br(row.f1_nova_high)}]",
                    br(row["diff"], 4),
                    f"[{br(row.diff_low, 4)}; {br(row.diff_high, 4)}]",
                ]
                for k, row in pooled.loc[["lbp", "combinado", "cnn3"]].iterrows()
            ],
        )
        for k, row in pooled.iterrows():
            r[f"gen_vista_{k}"] = br(row.f1_vista, 4)
            r[f"gen_nova_{k}"] = br(row.f1_nova, 4)
            r[f"gen_diff_{k}"] = br(row["diff"], 4)
            r[f"gen_diff_ic_{k}"] = f"[{br(row.diff_low, 4)}; {br(row.diff_high, 4)}]"
            r[f"acc_nova_{k}"] = pct(row.acc_nova)
            r[f"gen_queda_{k}"] = br(-row["diff"], 4)
        for other, row in loco_pairs.iterrows():
            r[f"gen_par_{other}"] = br(row.f1_diff, 4)
            r[f"gen_par_ic_{other}"] = f"[{br(row.diff_low, 4)}; {br(row.diff_high, 4)}]"
            r[f"gen_par_p_{other}"] = "< 0,0001" if row.mcnemar_p < 1e-4 else br(row.mcnemar_p, 4)
        for k in ("lbp", "combinado", "cnn3"):
            rows = per_camera[per_camera.model == k]
            worst = rows.sort_values("f1_nova").iloc[0]
            r[f"gen_pior_{k}"] = (
                f"câmera {int(worst.camera)} (F1 macro {br(worst.f1_nova)}, n={int(worst.n)})"
            )
            drop = rows.f1_vista - rows.f1_nova
            r[f"gen_quedas_{k}"] = str(int((drop > 0).sum()))
        new_camera = pd.read_csv(base / "generalizacao/predicoes_camera_nova.csv", dtype={"id": str})
        seen_lbp = pd.read_csv(Path(config_t1["results_dir"]) / "predicoes/lbp.csv", dtype={"id": str})
        same = (
            new_camera.pred_lbp.to_numpy()
            == seen_lbp.set_index("id").loc[new_camera.id, "predicted"].to_numpy()
        )
        r["gen_lbp_iguais"] = f"{milhar(same.sum())} das {milhar(len(same))}"
        folds = sorted(Path(base / "generalizacao").glob("treino_camera*.csv"))
        loco_train = pd.concat([pd.read_csv(p) for p in folds])
        cnn_folds = loco_train[loco_train.model == "cnn3"]
        r["gen_tempo_cnn_min"] = br(cnn_folds.training_seconds.sum() / 60, 0)
        r["gen_epocas_cnn"] = f"{int(cnn_folds.best_epoch.min())} a {int(cnn_folds.best_epoch.max())}"

    # Anotacao cega dos erros
    annotation = report_dir / "anotacao_erros.csv"
    if annotation.exists():
        r.update(substituicoes_anotacao(config, annotation))

    r["figuras_t2"] = Path(os.path.relpath(base / "figuras", report_dir)).as_posix()
    r["commit_t2_treino"] = freeze["provenance"]["commit"][:12]
    r["commit_t2_teste"] = evaluation["provenance"]["commit"][:12]
    # As duas etapas usaram o mesmo ambiente; a das redes acrescenta torch e torchvision.
    r["tabela_versoes"] = tabela(["Componente", "Versão"], evaluation["provenance"]["versions"].items())
    r["versao_torch"] = evaluation["provenance"]["versions"]["torch"]
    r["versao_torchvision"] = evaluation["provenance"]["versions"]["torchvision"]
    r["threads"] = str(evaluation["provenance"]["torch_threads"])
    r["selecionado_t1"] = NOMES[t1_freeze["selected"]]
    return r


CATEGORIAS = {
    "sombra": "Sombra forte",
    "chuva": "Chuva ou reflexo",
    "oclusao": "Oclusão por vegetação ou objeto",
    "parcial": "Veículo parcial ou na borda",
    "escuro": "Baixa iluminação",
    "perspectiva": "Perspectiva ou vaga vizinha",
    "rotulo": "Rótulo duvidoso",
    "sem_causa": "Sem causa aparente",
}


def substituicoes_anotacao(config, annotation):
    """Cruza as categorias atribuidas as cegas com os erros de cada modelo."""
    base = Path(config["results_dir"])
    items = pd.read_csv(base / "anotacao/itens.csv", dtype={"id": str})
    notes = pd.read_csv(annotation)
    data = items.merge(notes, on="item", validate="one_to_one")
    freeze = json.loads((base / "metricas/congelamento.json").read_text(encoding="utf-8"))
    config_t1 = carregar_config(config["t1_config"])
    cnn = pd.read_csv(base / "predicoes" / f"{freeze['primary']}.csv", dtype={"id": str}).set_index("id")
    lbp = pd.read_csv(Path(config_t1["results_dir"]) / "predicoes/lbp.csv", dtype={"id": str}).set_index("id")
    data["erro_cnn"] = cnn.loc[data.id, "predicted"].to_numpy() != data.label.to_numpy()
    data["erro_lbp"] = lbp.loc[data.id, "predicted"].to_numpy() != data.label.to_numpy()
    data.to_csv(base / "metricas/anotacao_erros_cruzada.csv", index=False)
    rows = []
    for key, name in CATEGORIAS.items():
        group = data[data.categoria == key]
        rows.append(
            [
                name,
                int(group.erro_lbp.sum()),
                pct(group.erro_lbp.sum() / max(data.erro_lbp.sum(), 1), 0),
                int(group.erro_cnn.sum()),
                pct(group.erro_cnn.sum() / max(data.erro_cnn.sum(), 1), 0),
            ]
        )
    rows.append(["Total", int(data.erro_lbp.sum()), "100%", int(data.erro_cnn.sum()), "100%"])
    counts = {}
    for key in CATEGORIAS:
        group = data[data.categoria == key]
        counts[f"anot_{key}_lbp"] = str(int(group.erro_lbp.sum()))
        counts[f"anot_{key}_cnn"] = str(int(group.erro_cnn.sum()))
    return {
        **counts,
        "tabela_anotacao": tabela(["Categoria", "Erros LBP", "% LBP", "Erros CNN-3", "% CNN-3"], rows),
        "anotacao_itens": str(len(data)),
    }


def relatorio_final(config):
    report_dir = Path(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    config_t1 = carregar_config(config["t1_config"])
    replacements = substituicoes_t1(config_t1, report_dir)
    replacements.update(substituicoes_t2(config, report_dir))
    # Valores herdados do relatorio classico usam ponto; aqui o texto usa virgula decimal.
    for key in ("c", "val_f1", "val_f1_combinado"):
        replacements[key] = replacements[key].replace(".", ",")
    content = preencher(report_dir / "modelo.md", replacements)
    source = report_dir / "relatorio.md"
    source.write_text(content, encoding="utf-8")
    renderizar_pdf(
        source,
        source.with_suffix(".pdf"),
        cabecalho="INE5443 | Ocupação de vagas: de descritores clássicos a redes convolucionais",
        titulo="Classificação de ocupação de vagas: de descritores clássicos a redes convolucionais",
        recuo=True,
    )
