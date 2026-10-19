"""Relatorio Markdown/PDF gerado das tabelas finais e de um texto revisavel."""

import json
import os
from pathlib import Path

import pandas as pd

from .dados import carregar_manifesto
from .visualizacao import resultados_verificados


def tabela(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def escrever_markdown(config):
    evaluation = resultados_verificados(config)
    base = Path(config["results_dir"])
    read = lambda name: json.loads((base / "metricas" / name).read_text(encoding="utf-8"))
    audit, freeze = read("auditoria.json"), read("congelamento.json")
    frame = carregar_manifesto(config)
    test = pd.read_csv(base / "metricas/teste.csv").set_index("descriptor")
    validation = pd.read_csv(base / "metricas/validacao.csv")
    climate = pd.read_csv(base / "metricas/clima.csv")
    winner = freeze["selected"]
    best, dummy = test.loc[winner], test.loc["majoritario"]
    selected = next(r for r in freeze["models"] if r["descriptor"] == winner)
    prediction = pd.read_csv(base / "predicoes" / f"{winner}.csv")
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    replacements = {
        "n_total": f"{len(frame):,}".replace(",", "."),
        "n_treino": f"{int(counts.loc['treino'].sum()):,}".replace(",", "."),
        "n_teste": f"{int(counts.loc['teste'].sum()):,}".replace(",", "."),
        "vencedor": winner,
        "c": f"{selected['C']:g}",
        "val_f1": f"{selected['validation_f1_macro']:.4f}",
        "teste_f1": f"{best.f1_macro:.4f}",
        "teste_acc": f"{best.accuracy:.4f}",
        "dummy_f1": f"{dummy.f1_macro:.4f}",
        "ganho_f1": f"{best.f1_macro - dummy.f1_macro:.4f}",
        "conflitos": str(audit["conflicting_labels"]),
        "ilegíveis": str(audit["unreadable"]),
        "duplicatas": str(audit["duplicates_removed"]),
        "manifest_hash": audit["manifest_sha256"],
        "ocupada_livre": str(((prediction.label == 1) & (prediction.predicted == 0)).sum()),
        "livre_ocupada": str(((prediction.label == 0) & (prediction.predicted == 1)).sum()),
        "recall_ocupada": f"{best.recall_ocupada:.4f}",
        "ambiente": evaluation["provenance"]["platform"],
        "processador": evaluation["provenance"]["processor"],
        "commit_treino": freeze["provenance"]["commit"][:12],
        "commit_teste": evaluation["provenance"]["commit"][:12],
    }
    replacements["tabela_dados"] = tabela(
        ["Partição", "Livre", "Ocupada", "Total"],
        [
            [s, counts.loc[s, 0], counts.loc[s, 1], counts.loc[s].sum()]
            for s in ("treino", "validacao", "teste")
        ],
    )
    replacements["tabela_validacao"] = tabela(
        ["Descritor", "C", "Dimensão", "F1 macro", "Ajuste (s)"],
        [
            [
                r.descriptor,
                f"{r.C:g}",
                r.dimension,
                f"{r.validation_f1_macro:.4f}",
                f"{r.training_seconds:.3f}",
            ]
            for r in validation.itertuples()
        ],
    )
    replacements["tabela_teste"] = tabela(
        ["Descritor", "Acurácia", "F1 macro", "C"],
        [
            [n, f"{r.accuracy:.4f}", f"{r.f1_macro:.4f}", "-" if n == "majoritario" else f"{r.C:g}"]
            for n, r in test.iterrows()
        ],
    )
    replacements["tabela_classes"] = tabela(
        ["Descritor", "P livre", "R livre", "F1 livre", "P ocup.", "R ocup.", "F1 ocup."],
        [
            [
                n,
                *[
                    f"{r[k]:.3f}"
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
            for n, r in test.iterrows()
        ],
    )
    replacements["tabela_tempos"] = tabela(
        ["Descritor", "Dim.", "Ajuste (s)", "Extr. ms/img", "Pred. ms/img"],
        [
            [
                n,
                int(r.dimension),
                f"{r.training_seconds:.3f}",
                f"{r.extraction_ms_per_image:.3f}",
                f"{r.inference_ms_per_image:.4f}",
            ]
            for n, r in test.iterrows()
        ],
    )
    pivot = climate.pivot(index="weather", columns="descriptor", values="f1_macro")
    replacements["tabela_clima"] = tabela(
        ["Clima", "N", "Livre", "Ocup.", *test.index.tolist()],
        [
            [
                w,
                int(g.n.iloc[0]),
                int(g.n_livre.iloc[0]),
                int(g.n_ocupada.iloc[0]),
                *[f"{pivot.loc[w, n]:.3f}" for n in test.index],
            ]
            for w, g in climate.groupby("weather")
        ],
    )
    replacements["tabela_versoes"] = tabela(
        ["Componente", "Versão"], evaluation["provenance"]["versions"].items()
    )
    worst = climate[climate.descriptor == winner].sort_values("f1_macro").iloc[0]
    replacements["pior_clima"] = f"{worst.weather} (F1 macro {worst.f1_macro:.4f}, n={int(worst.n)})"
    for field in ("camera", "date", "capture", "slot"):
        replacements[f"sobreposicao_{field}"] = str(audit["overlap_selected"][field]["treino/teste"])
    ranked = test.drop(index="majoritario").f1_macro.idxmax()
    replacements["nota_ranking"] = (
        "O vencedor de validação também alcançou o maior F1 macro de teste entre os métodos clássicos."
        if ranked == winner
        else f"No teste, {ranked} obteve o maior F1 macro. Isso não altera a seleção realizada pela validação."
    )
    report_dir = Path(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis = report_dir / "analise_erros.md"
    replacements["analise_erros"] = (
        analysis.read_text(encoding="utf-8")
        if analysis.exists()
        else "Inspeção manual dos exemplos ainda não registrada."
    )
    template = Path(config["project_root"]) / "relatorios/t1/modelo.md"
    content = template.read_text(encoding="utf-8")
    replacements["figuras"] = Path(os.path.relpath(base / "figuras", report_dir)).as_posix()
    for key, value in replacements.items():
        content = content.replace("{{" + key + "}}", value)
    if "{{" in content:
        raise ValueError("Marcador do relatorio nao preenchido")
    source = report_dir / "relatorio.md"
    source.write_text(content, encoding="utf-8")
    return source


def renderizar_pdf(source, output):
    from markdown_it import MarkdownIt
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    source = Path(source)
    md = MarkdownIt("default", {"html": True}).enable("table")
    tokens = md.parse(source.read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "BodyT1", fontName="Helvetica", fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=8
        )
    )
    styles.add(ParagraphStyle("CellT1", fontName="Helvetica", fontSize=8, leading=10))
    styles["Title"].fontSize, styles["Title"].leading = 22, 26
    styles["Heading2"].fontSize, styles["Heading2"].leading = 15, 19
    styles["Heading2"].textColor = colors.HexColor("#173c69")
    width = A4[0] - 4 * cm

    def inline(token):
        html = md.renderer.renderInline(token.children or [], md.options, {})
        for a, b in (("strong", "b"), ("em", "i")):
            html = html.replace(f"<{a}>", f"<{b}>").replace(f"</{a}>", f"</{b}>")
        return html.replace("<code>", '<font name="Courier">').replace("</code>", "</font>")

    story, i, list_depth = [], 0, 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "html_block" and "PAGEBREAK" in token.content:
            story.append(PageBreak())
        elif token.type == "heading_open":
            name = {"h1": "Title", "h2": "Heading2", "h3": "Heading3"}.get(token.tag, "Heading3")
            story.append(Paragraph(inline(tokens[i + 1]), styles[name]))
            i += 2
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif token.type == "paragraph_open":
            child = tokens[i + 1]
            images = [c for c in child.children or [] if c.type == "image"]
            if images:
                for image in images:
                    graphic = Image(str(source.parent / image.attrGet("src")))
                    scale = min(width / graphic.imageWidth, 10.5 * cm / graphic.imageHeight)
                    graphic.drawWidth, graphic.drawHeight = (
                        graphic.imageWidth * scale,
                        graphic.imageHeight * scale,
                    )
                    story.extend([graphic, Spacer(1, 6)])
            else:
                story.append(Paragraph(("- " if list_depth else "") + inline(child), styles["BodyT1"]))
            i += 2
        elif token.type == "table_open":
            rows, cells = [], []
            i += 1
            while tokens[i].type != "table_close":
                if tokens[i].type == "tr_open":
                    cells = []
                elif tokens[i].type == "inline":
                    cells.append(Paragraph(inline(tokens[i]), styles["CellT1"]))
                elif tokens[i].type == "tr_close":
                    rows.append(cells)
                i += 1
            column_widths = [width / len(rows[0])] * len(rows[0])
            if len(rows[0]) == 9:
                column_widths = [66] + [(width - 66) / 8] * 8
            table = Table(rows, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eef7")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#173c69")),
                    ]
                )
            )
            story.extend([table, Spacer(1, 9)])
        i += 1

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536477"))
        canvas.drawString(2 * cm, A4[1] - 1.25 * cm, "INE5443 | T1 | Ocupacao de vagas com metodos classicos")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=1.7 * cm,
        title="T1 - Classificacao de ocupacao de vagas",
        author="Projeto VisaoComputacional",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)


def relatorio(config):
    source = escrever_markdown(config)
    renderizar_pdf(source, source.with_suffix(".pdf"))
