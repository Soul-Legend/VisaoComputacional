from pypdf import PdfReader

from visao_computacional.relatorio import renderizar_pdf


def test_quebras_do_markdown_nao_aparecem_como_texto(tmp_path):
    source = tmp_path / "fonte.md"
    source.write_text(
        "# Primeiro\n\nTexto.\n\n<!-- PAGEBREAK -->\n\n## Segundo\n\nOutro texto.\n", encoding="utf-8"
    )
    output = tmp_path / "relatorio.pdf"
    renderizar_pdf(source, output)
    pages = PdfReader(output).pages
    assert len(pages) == 2
    assert "Primeiro" in pages[0].extract_text()
    assert "Segundo" in pages[1].extract_text()
    assert "PAGEBREAK" not in "".join(p.extract_text() for p in pages)
