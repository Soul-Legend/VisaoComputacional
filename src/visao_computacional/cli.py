"""Interface de linha de comando do T1 e do T2."""

import argparse
import importlib

from .configuracao import carregar_config

COMANDOS_T1 = (
    "baixar",
    "preparar",
    "explorar",
    "extrair",
    "treinar",
    "avaliar",
    "figuras",
    "ilustracoes",
    "relatorio",
)
COMANDOS_T2 = (
    "imagens",
    "redes-treinar",
    "redes-avaliar",
    "generalizacao",
    "comparar",
    "redes-figuras",
    "relatorio-final",
)


def main():
    parser = argparse.ArgumentParser(description="Experimentos de ocupacao de vagas (T1 e T2)")
    parser.add_argument("--config", default=None, help="padrao: configs/t1.yaml ou configs/t2.yaml")
    parser.add_argument("comando", choices=[*COMANDOS_T1, *COMANDOS_T2])
    args = parser.parse_args()
    if args.comando in COMANDOS_T1:
        config = carregar_config(args.config or "configs/t1.yaml")
        from .avaliacao import avaliar
        from .dados import baixar, preparar
        from .descritores import extrair
        from .ilustracoes import ilustracoes
        from .relatorio import relatorio
        from .treinamento import treinar
        from .visualizacao import explorar, figuras

        actions = {
            "baixar": baixar,
            "preparar": preparar,
            "explorar": explorar,
            "extrair": extrair,
            "treinar": treinar,
            "avaliar": avaliar,
            "figuras": figuras,
            "ilustracoes": ilustracoes,
            "relatorio": relatorio,
        }
    else:
        from .redes import carregar_config_t2

        config = carregar_config_t2(args.config or "configs/t2.yaml")
        # Importacao tardia: cada etapa carrega apenas o modulo de que precisa.
        module, function = {
            "imagens": ("redes", "preparar_imagens"),
            "redes-treinar": ("t2", "treinar_redes"),
            "redes-avaliar": ("t2", "avaliar_redes"),
            "generalizacao": ("t2", "generalizacao"),
            "comparar": ("comparacao", "comparar"),
            "redes-figuras": ("t2_figuras", "redes_figuras"),
            "relatorio-final": ("relatorio_final", "relatorio_final"),
        }[args.comando]
        actions = {args.comando: getattr(importlib.import_module(f".{module}", __package__), function)}
    actions[args.comando](config)


if __name__ == "__main__":
    main()
