"""Interface de linha de comando do T1."""

import argparse

from .configuracao import carregar_config


def main():
    parser = argparse.ArgumentParser(description="Experimentos de ocupacao de vagas do T1")
    parser.add_argument("--config", default="configs/t1.yaml")
    parser.add_argument(
        "comando",
        choices=[
            "baixar",
            "preparar",
            "explorar",
            "extrair",
            "treinar",
            "avaliar",
            "figuras",
            "ilustracoes",
            "relatorio",
        ],
    )
    args = parser.parse_args()
    config = carregar_config(args.config)
    from .avaliacao import avaliar
    from .dados import baixar, preparar
    from .descritores import extrair
    from .ilustracoes import ilustracoes
    from .relatorio import relatorio
    from .treinamento import treinar
    from .visualizacao import explorar, figuras

    {
        "baixar": baixar,
        "preparar": preparar,
        "explorar": explorar,
        "extrair": extrair,
        "treinar": treinar,
        "avaliar": avaliar,
        "figuras": figuras,
        "ilustracoes": ilustracoes,
        "relatorio": relatorio,
    }[args.comando](config)


if __name__ == "__main__":
    main()
