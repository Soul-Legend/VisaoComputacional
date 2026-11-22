# Planejamento e protocolo do T2

Registrado antes de qualquer treinamento de rede neural. O teste do T1 já foi aberto para os
modelos clássicos; para as redes, ele só será consultado depois do congelamento descrito abaixo.

## Objetivo e escopo

Mesmo problema do T1: classificar um recorte de vaga como livre (0) ou ocupada (1). Muda apenas
o método: uma CNN treinada do zero e, como extensão, MobileNetV2 com transferência de aprendizado.
O objetivo é comparar as abordagens no **mesmo manifesto, nas mesmas partições, no mesmo teste e
com as mesmas métricas**. Sem detector de vagas, vídeo, rastreamento ou YOLO.

Pergunta central: em que medida uma CNN supera descritores clássicos com SVM linear nesse
problema, considerando desempenho, robustez a clima e a câmera nova, e custo computacional?

## Dados e pré-processamento (inalterados)

- Manifesto congelado `data/manifests/t1.csv` (SHA-256 `f7ac99e8...`): 7.000 / 1.500 / 1.500.
- Leitura idêntica ao T1: BGR→RGB, redimensionamento para 128 × 128 com interpolação de área.
- Cache de pixels `uint8` com assinatura do manifesto e do tamanho; tempo de leitura medido por partição.
- CNN: normalização por canal com média e desvio do treino. MobileNetV2: médias do ImageNet,
  exigidas pelos pesos pré-treinados.

## Modelos

**CNN-3 (modelo principal do T2).** Três blocos `Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2`,
com 32, 64 e 128 filtros; `Flatten` (128 × 16 × 16) → Dropout 0,5 → Dense 128 + ReLU → Dense 1.
A saída é um *logit*; a sigmoide o converte em probabilidade de ocupação. Limiar fixo 0,5.

**MobileNetV2 (extensão).** Pesos ImageNet (torchvision, IMAGENET1K_V1), classificador trocado
por Dense 1, todas as camadas ajustadas. Sem busca própria: usa a escolha de aumento da CNN-3.

## Treinamento

Adam, perda de entropia cruzada binária, lote 64, *weight decay* 1e-4, sem pesos de classe.
CNN-3: taxa 1e-3, até 25 épocas. MobileNetV2: taxa 1e-4, até 15 épocas.
A cada época mede-se perda e F1 macro no treino e na validação. O ponto salvo é a época de maior
F1 macro de validação (arredondado a 4 casas; empate: menor perda de validação, depois época
mais cedo). Parada antecipada após 6 épocas sem melhora (4 na MobileNetV2).
Semente 42, algoritmos determinísticos do PyTorch e 8 threads de CPU.

## Seleção pela validação

Única escolha da CNN-3: aumento de dados **sem** ou **com**. "Com" aplica, por imagem:
rotação de ±8°, zoom de 0,9 a 1,1, translação de ±6%, brilho e contraste de 0,8 a 1,2.
Sem espelhamentos. Vence o maior F1 macro de validação; empate favorece "sem" (mais simples).

## Variabilidade e congelamento

Depois da escolha, a configuração vencedora é treinada também com as sementes 43 e 44, com o
mesmo critério de época. O modelo principal é o da semente 42, definido antes do teste.
Todos os pesos, curvas e hashes são congelados. Só então ocorre **um único lote de teste**
com CNN-3 (três sementes) e MobileNetV2. Repetições recusam sobrescrita.

## Comparação com o T1

Os modelos clássicos não são retreinados: usam-se as predições congeladas do T1.
Referência clássica principal: **LBP + SVM** (escolhido pela validação do T1). A combinação
HSV + LBP + HOG, melhor no teste do T1, também é comparada, com a ressalva de ter sido
destacada depois do teste.

Métricas idênticas: acurácia, F1 macro, precisão/recall/F1 por classe e matriz de confusão.
Novidades estatísticas: intervalos de 95% por *bootstrap* agrupado por captura (câmera, data
e horário; 2.000 reamostragens, semente 42), intervalo da diferença pareada de F1 macro e teste
exato de McNemar nos pares discordantes. O teste tem 695 capturas, mas apenas 5 dias;
a correlação entre dias não é coberta pelo bootstrap.

## Clima

Desagregação do mesmo teste em SUNNY, OVERCAST e RAINY, com contagens por grupo.

## Generalização para câmera nova

Validação cruzada deixando uma câmera de fora (9 dobras). Para cada câmera k: treino e validação
sem a câmera k; teste com as imagens da câmera k do teste congelado. Hiperparâmetros fixos:
LBP e HSV + LBP + HOG com C=0,01 (valores do T1) e a configuração escolhida da CNN-3.
A comparação é pareada: as mesmas 1.500 imagens de teste, previstas por modelos que viram ou
não a sua câmera. Ressalvas registradas antes: 114 das 164 vagas aparecem em mais de uma câmera,
então a dobra testa novo ponto de vista, não necessariamente vaga nova; o treino de cada dobra
perde cerca de 1/9 das imagens. MobileNetV2 não entra nessa etapa por custo.

## Análise de erros

Painéis de erros da CNN-3 e de discordâncias entre CNN-3 e LBP. Contagem de erros
exclusivos e compartilhados. Categorização visual dos erros da união LBP ∪ CNN-3 feita
**às cegas quanto ao modelo**: as imagens são mostradas embaralhadas, apenas com o rótulo real.
Categorias: sombra, chuva/reflexo, oclusão, veículo parcial, baixa iluminação, perspectiva,
rótulo duvidoso, sem causa aparente. Um único anotador; análise interpretativa, não causal.
Grad-CAM na CNN-3 para ilustrar as regiões que mais influenciam a decisão.

## Custo

Treino: tempo total de parede (todas as épocas, incluindo validação) contra o ajuste da SVM.
Inferência em CPU: preparação (leitura + redimensionamento) mais predição em lote, por imagem,
comparada à extração mais predição do T1. Uma execução, sem intervalo de variabilidade.

## Entregáveis

Código em `src/visao_computacional/`, configuração `configs/t2.yaml`, evidências em
`resultados/t2/` e relatório único combinado (T1 + T2) em `relatorios/final/`.
Escopo mínimo: CNN-3 + comparação. Ideal: + clima, matrizes, erros. Extra: MobileNetV2 e câmera nova.
