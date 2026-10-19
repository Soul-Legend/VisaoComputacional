# Visao Computacional — T1

Classificação de recortes de vagas em **livre (0)** ou **ocupada (1)** com HSV, LBP, HOG e SVM linear.

Código reutilizável em `src/`, parâmetros em `configs/`, documentação em `docs/`, evidências em
`resultados/t1/` e fonte/PDF do relatório em `relatorios/t1/`.

## Ambiente

Python 3.11 ou superior (experimento executado com Python 3.12).

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,report]"
```

Consulte [reprodução](docs/reproducao.md), [dataset](docs/dataset.md),
[metodologia](docs/metodologia.md) e [planejamento](docs/planejamento.md).
Veja também a [explicação visual de HSV, LBP e HOG](docs/metodos.md), com exemplos reais de treino.

Datasets, caches e modelos não são versionados. Não há CNN nem detector de vagas neste T1.

## Execução

```powershell
.venv\Scripts\python -m visao_computacional.cli --help
.venv\Scripts\python -m visao_computacional.cli baixar
.venv\Scripts\python -m visao_computacional.cli explorar
.venv\Scripts\python -m visao_computacional.cli extrair
```

O manifesto e os resultados publicados estão congelados. Para reconstruir auditoria, treinar e
avaliar sem sobrescrevê-los, siga o estudo separado de [reprodução](docs/reproducao.md).
Os comandos `treinar` e `avaliar` recusam sobrescrita no mesmo estudo.

## Resultados do experimento

10.000 imagens do CNR-EXT: 7.000 treino, 1.500 validação e 1.500 teste. Foram excluídos
38 registros com pixels idênticos e rótulos conflitantes antes da amostragem.

| Representação | F1 macro de validação (melhor C) | Acurácia de teste | F1 macro de teste |
| --- | --- | --- | --- |
| HOG | 0,8762 | 0,9127 | 0,9114 |
| HSV | 0,7463 | 0,8080 | 0,8051 |
| LBP | **0,9028** | 0,9020 | 0,8998 |
| HSV + LBP + HOG | 0,9027 | **0,9307** | **0,9295** |
| Classe majoritária | — | 0,5747 | 0,3649 |

O protocolo selecionou **LBP + SVM, C=0,01**, pela validação. A combinação teve o maior
F1 de teste; isso não alterou a seleção. São resultados de um único subconjunto, sem teste
de significância. As datas são distintas, mas as mesmas câmeras e vagas aparecem nas partições.

- [Relatório em Markdown](relatorios/t1/relatorio.md) e [PDF](relatorios/t1/relatorio.pdf).
- [Métricas completas](resultados/t1/metricas/teste.csv) e [validação](resultados/t1/metricas/validacao.csv).
- [Auditoria](resultados/t1/metricas/auditoria.json), [manifesto](data/manifests/t1.csv) e
  [predições individuais](resultados/t1/predicoes/).

Os testes usam dados sintéticos somente para verificar o software; os números acima vêm de imagens reais.
