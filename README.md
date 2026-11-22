# Visao Computacional — T1 e T2

Classificação de recortes de vagas em **livre (0)** ou **ocupada (1)**. O T1 usa HSV, LBP, HOG e SVM
linear; o T2 usa uma CNN treinada do zero (CNN-3) e, como extensão, MobileNetV2 pré-treinada, no mesmo
manifesto, nas mesmas partições e no mesmo teste.

**Relatório final combinado (T1 + T2):** [Markdown](relatorios/final/relatorio.md) e [PDF](relatorios/final/relatorio.pdf).

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

Datasets, caches e modelos não são versionados. Não há detector de vagas: as posições são conhecidas.

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

## Resultados do T2

Protocolo registrado antes do treino em [docs/planejamento_t2.md](docs/planejamento_t2.md). A validação
escolheu a CNN-3 sem aumento de dados; o teste foi aberto uma única vez depois do congelamento.
Intervalos de 95% por bootstrap agrupado por captura (695 capturas, 2.000 reamostragens).

| Modelo | Acurácia de teste | F1 macro de teste | IC 95% do F1 macro |
| --- | --- | --- | --- |
| LBP + SVM (escolhido no T1) | 0,9020 | 0,8998 | [0,883; 0,916] |
| HSV + LBP + HOG + SVM | 0,9307 | 0,9295 | [0,916; 0,942] |
| **CNN-3** (semente 42) | 0,9833 | **0,9829** | [0,976; 0,989] |
| MobileNetV2 (extensão) | 0,9913 | 0,9911 | [0,985; 0,996] |

CNN-3 menos HSV + LBP + HOG: +0,0534 de F1 macro, IC [0,040; 0,067], McNemar p < 0,0001.
As sementes 43 e 44 da CNN-3 obtiveram 0,9871 e 0,9870. Com a câmera do teste fora do treino
(uma câmera por vez), a CNN-3 caiu para 0,9693, contra 0,9054 da combinação e 0,8963 do LBP no mesmo
cenário. A análise cega dos erros está no relatório final.

- [Métricas do T2](resultados/t2/metricas/), [predições](resultados/t2/predicoes/) e [figuras](resultados/t2/figuras/).
- Comandos do T2 em [reprodução](docs/reproducao.md#t2-redes-neurais-e-relatório-combinado).
