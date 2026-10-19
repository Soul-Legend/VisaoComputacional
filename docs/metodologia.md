# Metodologia experimental

## Representações e classificador

Entrada RGB de 128 × 128, redimensionada com interpolação de área. OpenCV lê BGR e o código
converte explicitamente para RGB. Cinza em uint8 para HOG e LBP.

- HSV: histogramas separados com 16 bins; H no intervalo [0,180), S/V em [0,256).
  Cada histograma soma 1; concatenação tem 48 componentes.
- LBP uniforme: P=8, R=1; histograma de P+2=10 bins, normalizado.
- HOG: 9 orientações, células 8 × 8, blocos 2 × 2, L2-Hys; 8.100 componentes.
- Combinação: HSV + LBP + HOG, nessa ordem; 8.158 componentes.

`Pipeline(StandardScaler, LinearSVC)`; C em {0,01; 0,1; 1; 10}, sem pesos de classe,
sem ajuste de limiar ou probabilidades. O escalonador aprende exclusivamente no treino.
`dual="auto"`, tolerância 1e-4, máximo 20.000 iterações. Não convergência interrompe o treinamento.

## Seleção e congelamento

Maior F1 macro na validação, arredondado a quatro casas; empate favorece menor dimensão e
depois menor C. O treinamento salva um modelo por representação, uma referência majoritária
e a identidade do vencedor global. Não há retreino com a validação.

O teste permanece fora da seleção. A avaliação verifica hashes dos artefatos congelados e
produz um lote com todos os modelos; nova execução não sobrescreve os resultados existentes.
Comparações de teste são descritivas: não mudam a seleção nem autorizam novos ajustes.

## Métricas e custo

Acurácia, F1 macro e precisão/recall/F1/suporte por classe. Classes fixas [0,1], com
zero para denominadores nulos. Matrizes: linhas verdadeiras, colunas previstas.
Clima é uma desagregação do teste, com contagens e distribuição de classes.

Medir tempo total de extração por representação, ajuste incluindo escalonamento e inferência
em lote incluindo escalonamento. Reportar ms/imagem; não confundir extração com inferência.
Tempos dependem da máquina, cache e carga. Uma execução não fornece intervalo de confiança.

A análise visual seleciona até seis erros de cada direção do vencedor (até 12), com semente
fixa. É ilustrativa; não estima prevalência de causas. O custo de uma vaga ocupada prevista
como livre deve ser discutido separadamente.

## Interfaces e artefatos

CLI `visao-t1 --config configs/t1.yaml COMANDO`. Etapas: `baixar`, `preparar`, `explorar`,
`extrair`, `treinar`, `avaliar`, `figuras` e `relatorio`. Caminhos relativos são resolvidos pela
raiz do projeto (diretório pai de `configs/`). Arquivos externos podem usar caminhos absolutos.
Manifesto, características, modelos e predições preservam a ordem e identidade das amostras.
Uma impressão digital conecta configuração, manifesto, caches e modelos para impedir reuso
silencioso de resultados incompatíveis.
