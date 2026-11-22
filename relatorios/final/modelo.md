# Classificação de ocupação de vagas: de descritores clássicos a redes convolucionais

Relatório final | INE5443 - Reconhecimento de Padrões | 2026-2

Repositório: Soul-Legend/VisaoComputacional

### Resumo

Este trabalho estuda como decidir, a partir do recorte de uma vaga de estacionamento, se ela está
livre ou ocupada. O estudo começa por uma linha de base clássica, em que a imagem é descrita por
características definidas manualmente (cor, textura e gradientes) e classificada por uma SVM linear.
Em seguida, verifica se uma rede neural convolucional, que aprende a própria representação a partir
dos pixels, supera os limites dessa linha de base. Todos os modelos usaram as mesmas {{n_total}}
imagens do CNR-EXT, as mesmas partições e o mesmo teste de {{n_teste}} imagens, consultado só depois
que cada modelo estava congelado.

No teste, a melhor representação clássica, a combinação HSV + LBP + HOG, alcançou F1 macro de
{{f1_combinado}}, e o LBP com SVM, escolhido pela validação, {{f1_lbp}}. A CNN-3, uma rede pequena
treinada do zero, chegou a {{f1_cnn}} (IC 95% {{ic_cnn}}). A vantagem apareceu nos três climas e
reduziu de {{lbp_erros}} para {{cnn_erros}} os erros em relação ao LBP, principalmente em sombras e
em carros bem visíveis. Uma MobileNetV2 pré-treinada no ImageNet chegou a {{f1_mobilenet}}.
Treinada sem as imagens de uma câmera e testada nela, a CNN-3 caiu para {{gen_nova_cnn3}}, ainda
acima dos dois métodos clássicos no mesmo cenário. Em contrapartida, levou {{treino_cnn_min}}
minutos para treinar em CPU, contra frações de segundo da SVM, e seus erros restantes concentram-se
em vagas encobertas por vegetação e em imagens de chuva.

<!-- COLS 1.1,3.4 -->
| Sigla ou termo | Significado neste relatório |
| --- | --- |
| CNN | Rede neural convolucional (*convolutional neural network*): aprende filtros que extraem características da imagem. |
| CNN-3 | A rede convolucional de três blocos treinada do zero neste trabalho (seção 4). |
| SVM | Máquina de vetores de suporte (*support vector machine*): classificador que traça uma fronteira entre as classes. |
| HSV, LBP, HOG | Descritores clássicos de cor, textura e gradientes, explicados na seção 3. |
| F1 macro | Média do F1 das classes livre e ocupada, com o mesmo peso para as duas (seção 2.4). |
| IC 95% | Intervalo de confiança de 95%, estimado por *bootstrap* agrupado por captura (seção 2.4). |
| Captura | Uma fotografia de uma câmera em um instante; dela saem vários recortes de vagas. |
| Época | Uma passagem completa da rede por todas as imagens de treino. |
| Grad-CAM | Técnica que destaca as regiões da imagem que mais influenciaram a decisão da rede. |

Quadro 1. Siglas e termos usados com frequência.

<!-- PAGEBREAK -->

## 1. Introdução e objetivo

Saber quais vagas estão livres reduz o tempo que motoristas gastam circulando em um estacionamento.
Câmeras já instaladas podem fornecer essa informação, mas a aparência de uma mesma vaga muda muito
ao longo do dia. Um carro altera cores, texturas e contornos; sombras, reflexos, chuva e vegetação
também alteram essas pistas, sem que a vaga mude de estado. O desafio é reconhecer a ocupação apesar
dessas variações.

O escopo foi mantido estreito de propósito. O sistema recebe uma imagem já recortada de uma vaga cuja
posição é conhecida e responde livre (0) ou ocupada (1). Ele não detecta vagas na imagem completa,
não rastreia veículos, não lê placas e não processa vídeo. Essa delimitação permite concentrar o
estudo na pergunta que a disciplina propõe: **como uma abordagem clássica de visão computacional se
compara a uma rede neural no mesmo problema, considerando desempenho, robustez e custo?**

O trabalho avançou em duas etapas, e a segunda parte do que a primeira revelou. Começamos pela
abordagem clássica, em que a representação da imagem é fixa e escolhida por nós: histogramas de
cor, códigos de textura ou orientações de gradientes. Só a fronteira de decisão da SVM é aprendida.
Essa linha de base acertou a maioria das vagas, mas errava de forma sistemática em algumas
situações, como dias de sol com sombras fortes, o que aponta para um limite da própria
representação. A segunda etapa testa se uma rede neural convolucional (CNN), que aprende ao mesmo
tempo os filtros que descrevem a imagem e a decisão final, supera esse limite. Esperávamos que essa
flexibilidade desse vantagem à CNN, mas também que ela custasse mais tempo de treino e fosse menos
transparente.

Para que a comparação seja justa, a segunda etapa manteve tudo o que não faz parte do método: a
mesma lista congelada de imagens (o manifesto), as mesmas partições de treino, validação e teste, o
mesmo redimensionamento e as mesmas métricas. Os modelos clássicos não foram retreinados; suas
predições no teste foram registradas uma única vez e reutilizadas na comparação.

Além da comparação principal, investigamos duas perguntas de robustez. A primeira é se o desempenho
muda entre dias de sol, nublados e chuvosos. A segunda é o que acontece quando o modelo precisa
classificar imagens de uma câmera que não apareceu no treino. Por fim, examinamos os erros de cada
abordagem para entender em que situações cada uma falha.

O texto segue a mesma ordem. A seção 2 apresenta dados, protocolo e métricas; a seção 3 descreve a
linha de base clássica e os limites que motivaram a segunda etapa; a seção 4 descreve as redes
convolucionais. As seções 5 a 7 comparam as abordagens no teste, em condições adversas e pelos
erros que cometem, e as seções 8 e 9 discutem o que os resultados permitem concluir. As informações
para reproduzir o estudo ficam ao final.

<!-- PAGEBREAK -->

## 2. Dados e protocolo experimental

Esta seção descreve a origem das imagens, a divisão em partições, a forma de escolher modelos sem
consultar o teste e as métricas de avaliação. Tudo o que ela apresenta vale igualmente para os
métodos clássicos e para as redes.

### 2.1. Origem e limpeza das imagens

O CNRPark+EXT foi publicado por Amato e colaboradores para estudar a ocupação de vagas [1].
Utilizamos apenas a extensão CNR-EXT, com 144.965 recortes de nove câmeras capturados em 23 dias.
As imagens estão organizadas por clima: SUNNY (ensolarado), OVERCAST (nublado) e RAINY (chuvoso).
As listas dos autores fornecem os rótulos e as partições oficiais de treino, validação e teste.

Antes da amostragem, conferimos todas as imagens dessas listas. Encontramos {{ilegíveis}} imagens
ilegíveis e excluímos {{conflitos}} registros com pixels idênticos e rótulos conflitantes; nenhuma
duplicata adicional precisou ser removida. Depois, sorteamos {{n_total}} imagens com semente 42,
preservando aproximadamente as proporções de classe, câmera e clima de cada partição oficial
(Tabela 1). Nenhuma imagem mudou de partição.

{{tabela_dados}}

Tabela 1. Composição do subconjunto usado no estudo.

A Figura 1 mostra que o treino é quase balanceado, enquanto validação e teste preservam as
proporções das listas oficiais. A validação tem {{ocupadas_validacao}} de vagas ocupadas e nenhuma
imagem de chuva, porque a lista oficial de validação não contém RAINY. O teste tem
{{ocupadas_teste}} de ocupadas e {{rainy_teste}} de imagens chuvosas. Essa diferença é herdada do
dataset e limita o quanto a validação representa o teste, ponto que volta na discussão.

![Distribuição dos dados]({{figuras}}/distribuicoes.png)

Figura 1. Quantidade de imagens por classe e clima em cada partição. A ausência de RAINY na
validação vem da partição oficial do dataset.

### 2.2. O que muda entre treino e teste

Os exemplos da Figura 2 mostram por que a tarefa não se resume a encontrar uma mancha com formato
de carro. A perspectiva muda entre câmeras, árvores encobrem parte das vagas e sombras alteram a
aparência do asfalto. Todas as imagens foram redimensionadas para 128 × 128 pixels com o mesmo
procedimento para todos os modelos.

![Exemplos do treinamento]({{figuras}}/exemplos_treino.png)

Figura 2. Exemplos de treino nas duas classes e nos três climas. Origem: CNRPark+EXT.

Treino e teste não compartilham nenhuma data nem nenhuma captura, mas compartilham as
{{sobreposicao_camera}} câmeras e as {{sobreposicao_slot}} vagas. Portanto, o teste principal
mede o desempenho em **dias novos das mesmas câmeras e vagas**. O experimento da seção 6.2 retira
uma câmera do treino para avaliar um cenário mais difícil.

### 2.3. Aprender, escolher e avaliar sem consultar o teste

Cada partição tem um papel. O treino ajusta os parâmetros do modelo. A validação escolhe entre
configurações candidatas, como o valor de C da SVM ou o uso de aumento de dados na CNN. O teste
estima o desempenho da configuração escolhida em imagens que não influenciaram nenhuma decisão.

Para todos os modelos, a regra de escolha foi o maior F1 macro na validação, arredondado a quatro
casas, com empates resolvidos em favor da opção mais simples. Depois da escolha, os modelos foram
congelados e identificados por *hashes* (impressões digitais dos arquivos que revelam qualquer
alteração). Só então foram avaliados no teste, em **um único lote de avaliação por etapa**. O
código recusa repetir esse lote, para impedir ajustes depois de ver o resultado.

A ordem das etapas exige um cuidado adicional. Quando as redes foram projetadas, o resultado
clássico no teste já era conhecido. Para que esse conhecimento não influenciasse as redes, o
protocolo delas, com arquiteturas, candidatos e regra de escolha, foi escrito antes do primeiro
treino (docs/planejamento_t2.md), e nenhuma decisão sobre as redes usou o teste.

### 2.4. Como medir e comparar o desempenho

A acurácia é a fração de previsões corretas. Ela pode esconder diferenças entre classes: prever
sempre "ocupada" acerta muitos casos quando essa classe predomina, sem reconhecer nenhuma vaga
livre. Por isso a seleção usou o F1 macro. Para cada classe, a precisão indica quantas previsões
daquela classe estavam certas, e o *recall* indica quantos exemplos reais da classe foram
encontrados. O F1 é a média harmônica desses dois valores, e o F1 macro é a média simples do F1
das duas classes. Nas matrizes de confusão, as linhas são os rótulos reais e as colunas, as previsões.

Um único número de teste não diz o quanto ele poderia variar com outra amostra. Para estimar essa
incerteza, usamos *bootstrap*: sorteamos com reposição muitos conjuntos de teste do mesmo tamanho
e recalculamos a métrica em cada um [2]. Recortes da mesma fotografia compartilham iluminação e
clima e, portanto, não são independentes. Por isso o sorteio foi feito por captura: as
{{clusters}} capturas do teste foram reamostradas inteiras, {{reamostragens}} vezes. O IC 95% é
o intervalo entre os percentis 2,5 e 97,5 dessas repetições. O teste cobre apenas cinco dias, e
a correlação entre imagens do mesmo dia não é capturada por esse procedimento.

Para comparar dois modelos nas mesmas imagens, calculamos o intervalo da diferença de F1 macro
nas mesmas reamostragens e aplicamos o teste exato de McNemar [3]. Esse teste olha apenas os
pares discordantes, isto é, imagens em que um modelo acerta e o outro erra. Se os dois modelos
fossem equivalentes, essas discordâncias se dividiriam meio a meio. O McNemar trata as imagens
como independentes; o intervalo agrupado é a evidência mais conservadora das duas.

<!-- PAGEBREAK -->

## 3. Linha de base clássica: descritores e SVM

Na primeira etapa, cada recorte é convertido em um vetor numérico por um descritor escolhido
manualmente, e uma SVM linear aprende a separar os vetores de vagas livres e ocupadas. As Figuras 3 a 5 usam o mesmo
par de imagens de treino, a vaga 204 da câmera 6, livre às 07h46 e ocupada às 08h46 de 12/11/2015.
O mesmo par reaparece na seção 4 para mostrar o que a CNN extrai dele.

### 3.1. HSV: distribuição das cores

O espaço HSV separa matiz (H, a tonalidade), saturação (S, o quanto a cor se afasta do cinza) e
valor (V, a intensidade). O descritor conta a fração de pixels em 16 intervalos de cada canal e
concatena os três histogramas, formando 48 características. A Figura 3 mostra que o carro espalha
a saturação e a intensidade por mais intervalos do que o asfalto. A representação é compacta, mas
perde a posição das cores: imagens diferentes podem ter histogramas parecidos.

![Canais e histogramas HSV]({{ilustracoes}}/metodo_hsv.png)

Figura 3. Canais HSV e seus histogramas para a vaga livre e ocupada. Origem: CNRPark+EXT.

### 3.2. LBP: texturas locais

O *Local Binary Patterns* (LBP) compara cada pixel em tons de cinza com oito vizinhos em um círculo
de raio um e atribui um código ao padrão dessas comparações [4]. Na versão uniforme usada aqui, há
dez códigos possíveis; o descritor é o histograma normalizado desses códigos, com dez
características. A Figura 4 mostra que os padrões de textura se distribuem de forma diferente nos
dois recortes. O histograma, porém, descarta onde cada textura aparece.

![Texturas e histogramas LBP]({{ilustracoes}}/metodo_lbp.png)

Figura 4. Da imagem em cinza ao mapa de códigos LBP e ao histograma de textura.

### 3.3. HOG: contornos e orientações

O gradiente mede a mudança de intensidade entre pixels vizinhos e é forte nas bordas, como os
limites da carroceria. O *Histograms of Oriented Gradients* (HOG) resume a direção dessas mudanças
em células de 8 × 8 pixels, com nove orientações e normalização em blocos de 2 × 2 células [5].
Para recortes de 128 × 128, o vetor tem 8.100 características e preserva a organização espacial
dos contornos (Figura 5).

![Gradientes e representação HOG]({{ilustracoes}}/metodo_hog.png)

Figura 5. Recortes originais, intensidade dos gradientes e orientações por célula.

### 3.4. Classificação e escolha pela validação

O mesmo fluxo foi usado para cada descritor e para a concatenação HSV + LBP + HOG, com 8.158
características (Figura 6). Cada componente é padronizado com média e desvio do treino
(`StandardScaler`, do scikit-learn) e uma SVM linear (`LinearSVC`) aprende a fronteira [6].
O parâmetro C controla o quanto a SVM penaliza erros de treino e foi escolhido na validação entre
0,01; 0,1; 1 e 10.

![Fluxo de classificação]({{ilustracoes}}/fluxo_classificacao.png)

Figura 6. Fluxo clássico: o mesmo pipeline é aplicado a cada descritor e à combinação.

A Tabela 2 mostra o melhor C de cada representação na validação. Pela regra da seção 2.3, o modelo
escolhido foi **{{vencedor}}, com C={{c}}**, com F1 macro de {{val_f1}}, praticamente empatado com
a combinação dos três descritores ({{val_f1_combinado}}). Nenhuma SVM levou mais de dez segundos
para ajustar. O LBP e a combinação servem de referências clássicas na comparação da seção 5.

<!-- COLS 2,1.2,1.3,1.4,1.2 -->
{{tabela_selecao_classica}}

Tabela 2. Escolha da linha de base clássica na validação, com o melhor C de cada representação.

### 3.5. Limites de uma representação fixa

Os três descritores mostram escolhas que a abordagem clássica obriga a fazer antes de ver os dados:
o que medir e quanto da posição preservar. HSV e LBP resumem o recorte inteiro em um histograma e
descartam onde cada cor ou textura aparece; uma sombra forte ou um canteiro de grama pode, assim,
produzir um histograma parecido com o da lataria de um carro. O HOG preserva a posição dos
contornos, mas com células e orientações fixas, e a SVM linear só combina essas características de
forma aditiva. A segunda etapa remove essas escolhas: em vez de projetar a representação, deixamos
que ela seja aprendida a partir dos próprios exemplos.

<!-- PAGEBREAK -->

## 4. Redes convolucionais: representação aprendida

Nesta etapa, a representação deixa de ser escolhida por nós. Uma rede neural convolucional recebe os
pixels e ajusta, durante o treino, tanto os filtros que extraem características quanto a regra que
decide a classe [7]. Esta seção explica os componentes da rede, como ela foi treinada e como a
configuração final foi escolhida, com a mesma regra de validação usada para a SVM.

### 4.1. Arquitetura da CNN-3

Uma camada convolucional desliza pequenos filtros de 3 × 3 pixels sobre a imagem e produz um mapa
que indica onde cada padrão aparece. Nas primeiras camadas, esses padrões costumam ser bordas e
contrastes; nas seguintes, combinações deles. Cada convolução é seguida por normalização em lote
(*batch normalization*, que estabiliza a escala dos valores durante o treino [8]), pela função
ReLU (que zera valores negativos) e por *max pooling* (que reduz o mapa à metade, mantendo o maior
valor de cada janela 2 × 2).

A Figura 7 mostra a arquitetura. Três blocos reduzem a imagem de 128 × 128 para 16 × 16 posições
enquanto aumentam o número de mapas de 3 para 128. O resultado é achatado em um vetor de 32.768
valores (*flatten*) e passa por uma camada densa de 128 unidades. Durante o treino, o *dropout*
desliga aleatoriamente metade das entradas dessa camada, o que reduz a memorização do treino [9].
A última camada produz um único número, convertido pela função sigmoide em probabilidade de
ocupação. A vaga é classificada como ocupada quando essa probabilidade passa de 0,5. A rede tem
{{parametros_cnn}} parâmetros ajustáveis, a maioria na camada densa.

![Arquitetura da CNN-3]({{figuras_t2}}/arquitetura_cnn.png)

Figura 7. Arquitetura da CNN-3. A altura de cada bloco é proporcional à resolução espacial e a
largura cresce com o número de mapas.

### 4.2. Treinamento e aumento de dados

O treino ajusta os parâmetros para reduzir a entropia cruzada binária, uma perda que cresce quando
a rede atribui baixa probabilidade à classe correta. O otimizador Adam [10] atualizou os pesos em
lotes de 64 imagens, com taxa de aprendizado 0,001 e penalização leve dos pesos (*weight decay*
de 0,0001). Os pixels foram normalizados com a média e o desvio de cada canal no treino.

Redes com milhões de parâmetros podem decorar o treino. Para controlar isso, medimos o F1 macro
da validação ao fim de cada época e guardamos os pesos da melhor época. O treino parava após seis
épocas sem melhora, com limite de 25 épocas. Essa parada antecipada usa apenas a validação.

O aumento de dados (*data augmentation*) cria, a cada época, versões ligeiramente modificadas das
imagens de treino. Usamos apenas transformações plausíveis para uma câmera fixa: rotação de até 8°,
zoom de 0,9 a 1,1, deslocamento de até 6% e variações de brilho e contraste de ±20% (Figura 8).
Não houve espelhamento, porque inverter a cena criaria geometrias que as câmeras não produzem.

![Exemplos de aumento de dados]({{figuras_t2}}/aumento.png)

Figura 8. Uma imagem de treino e cinco variações sorteadas pelo aumento de dados.

### 4.3. Escolha da configuração pela validação

A única escolha da CNN-3 feita pela validação foi usar ou não o aumento de dados, papel equivalente
ao do parâmetro C na SVM. As duas configurações foram treinadas com a mesma semente. A Tabela 3 mostra que a versão
**{{aumento_escolhido}}** obteve o maior F1 macro de validação e foi a escolhida.

{{tabela_selecao}}

Tabela 3. Seleção da CNN-3 na validação. A época salva é a de maior F1 macro de validação.

As curvas da Figura 9 ajudam a interpretar essa escolha. Sem aumento, a perda de treino continuou
caindo até perto de 0,01, enquanto a perda de validação ficou entre 0,1 e 0,2 e oscilou. Esse
afastamento indica que a rede passou a memorizar detalhes do treino. Mesmo assim, o F1 macro de
validação seguiu oscilando em patamar alto, e o pico ocorreu na época 21. Com aumento, as curvas de
treino e validação ficaram mais próximas, mas a parada antecipada ocorreu na época 12, com pico na
época 6. A diferença entre as duas opções ({{val_f1_sem}} contra {{val_f1_com}}) é menor que a
variação entre sementes descrita a seguir. Portanto, a validação não separou as duas opções com
segurança, e a regra de escolha apenas registrou a ligeira vantagem observada.

![Curvas de treinamento]({{figuras_t2}}/curvas_treino.png)

Figura 9. Perda (em escala logarítmica) e F1 macro por época no treino e na validação. A linha
tracejada marca a época salva. As métricas de treino são medidas durante a própria passada, com
*dropout* e aumento ativos.

Uma rede também depende da semente aleatória, que define os pesos iniciais e a ordem das imagens.
Para estimar essa variação, a configuração escolhida foi treinada mais duas vezes, com as sementes
43 e 44. O modelo principal, de semente 42, foi definido antes do teste. Os F1 macro de validação
das três sementes foram {{val_f1_sementes}}, com épocas salvas {{epocas_sementes}}.

### 4.4. Extensão: MobileNetV2 com transferência de aprendizado

Como extensão, avaliamos a MobileNetV2 [11], uma rede projetada para dispositivos com pouca
capacidade de processamento. Em vez de partir de pesos aleatórios, ela partiu de pesos já treinados
no ImageNet [12], um conjunto de mais de um milhão de fotografias de objetos variados. Essa
estratégia, chamada transferência de aprendizado (*transfer learning*), reaproveita filtros gerais
de bordas, texturas e formas. Trocamos a última camada por uma saída única e ajustamos toda a rede
com taxa de aprendizado menor (0,0001), por até 15 épocas, com a mesma escolha de aumento de dados
da CNN-3. A MobileNetV2 tem {{parametros_mobilenet}} parâmetros e atingiu F1 macro de
{{val_f1_mobilenet}} na validação, na época {{epoca_mobilenet}}.

### 4.5. O que a CNN-3 extrai da imagem

A Figura 10 retoma o par usado nas Figuras 3 a 5. As colunas intermediárias mostram mapas internos
da CNN-3 treinada: dois mapas do primeiro bloco (os de maior ativação média no par) e a média dos
mapas dos blocos 2 e 3. A última coluna mostra o Grad-CAM [13], que combina os mapas do bloco 3
ponderados pela influência de cada um na decisão.

![Mapas internos da CNN-3]({{figuras_t2}}/mapas_cnn.png)

Figura 10. Mapas internos da CNN-3 e Grad-CAM para o mesmo par de treino das Figuras 3 a 5.
Cores mais claras indicam ativação maior. Origem das imagens: CNRPark+EXT.

No primeiro bloco, os mapas ainda se parecem com a imagem: realçam regiões claras e bordas, de forma
semelhante a um filtro de contraste. Nos blocos 2 e 3, a diferença entre as classes fica nítida. Na
vaga ocupada, as ativações se concentram em faróis, retrovisor, placa e contorno das rodas; na vaga
livre, quase não há ativação, exceto em marcações e bordas. O Grad-CAM da vaga livre destaca as
faixas e o meio-fio, o que sugere que a rede usa a presença dessas marcações visíveis como evidência
de vaga vazia.

A comparação com as Figuras 3 a 5 retoma os limites apontados na seção 3.5. Em vez de um
histograma que descarta a posição ou de células com regra fixa, a CNN-3 preserva onde cada padrão
aparece e aprende quais combinações importam para a decisão. Estes mapas ilustram dois exemplos e
não provam que a rede use sempre as mesmas pistas.

<!-- PAGEBREAK -->

## 5. Comparação no teste

Todos os números desta seção vêm das mesmas {{n_teste}} imagens de teste, consultadas só depois que
cada modelo estava congelado (seção 2.3).

### 5.1. Desempenho geral

A Tabela 4 reúne todos os modelos, com duas referências clássicas. O LBP é a escolha da validação,
prevista pelo protocolo. A combinação HSV + LBP + HOG teve o melhor resultado clássico no teste e,
por ter sido destacada depois de ver o teste, é uma referência favorável à abordagem clássica.

A CNN-3 obteve F1 macro de {{f1_cnn}}, com IC 95% {{ic_cnn}}. O intervalo não se sobrepõe ao de
nenhum método clássico: a combinação ficou em {{f1_combinado}} {{ic_combinado}}, e o LBP, em
{{f1_lbp}} {{ic_lbp}}. A MobileNetV2 alcançou {{f1_mobilenet}} {{ic_mobilenet}}.

<!-- COLS 2.3,1.5,1,1,1.5 -->
{{tabela_comparacao}}

Tabela 4. Desempenho no teste. O IC 95% vem do *bootstrap* agrupado por captura. As linhas de
semente 43 e 44 repetem a CNN-3 com outra inicialização; a linha principal usa a semente 42.

Na Figura 11, os métodos clássicos ocupam a faixa de 0,80 a 0,93, e as redes, a faixa acima de
0,98. A variação entre as três sementes da CNN-3 é pequena diante dessa separação: média de
{{f1_sementes_media}}, desvio-padrão de {{f1_sementes_dp}}, de {{f1_sementes_min}} a
{{f1_sementes_max}}; a semente principal, fixada antes do teste, ficou com o menor valor. A
configuração com aumento de dados, não escolhida, obteve {{f1_alternativa}}, o que confirma que
essa escolha teve pouco efeito.

![Comparação com intervalos de confiança]({{figuras_t2}}/comparacao_ic.png)

Figura 11. F1 macro no teste com IC 95%. A separação entre métodos clássicos e redes é maior
que a incerteza de cada estimativa.

<!-- PAGEBREAK -->

### 5.2. As diferenças são consistentes?

Os intervalos da Tabela 4 tratam cada modelo isoladamente. Como todos foram avaliados nas mesmas
imagens, uma comparação pareada é mais informativa. A Tabela 5 apresenta a diferença de F1 macro
entre pares de modelos, seu intervalo e o teste de McNemar.

<!-- COLS 2.3,1.2,1.5,0.9,0.9,1 -->
{{tabela_pares}}

Tabela 5. Comparações pareadas no teste. "Só A acerta" conta imagens em que apenas o primeiro
modelo do par acerta; "Só B acerta", o contrário.

Diante do LBP, a CNN-3 acertou {{so_a_cnn3_lbp}} imagens que o LBP errou e errou apenas
{{so_b_cnn3_lbp}} que o LBP acertou. Diante da combinação clássica, a proporção foi de
{{so_a_cnn3_combinado}} para {{so_b_cnn3_combinado}}. Nos dois casos, o intervalo da diferença
fica inteiro acima de zero e o McNemar dá p {{p_cnn3_combinado}}. A vantagem da CNN-3, portanto,
não depende de poucas imagens.

A comparação entre as duas redes é mais apertada. A MobileNetV2 superou a CNN-3 em
{{diff_mobilenet_cnn3}}, com intervalo {{diff_ic_mobilenet_cnn3}} e p = {{p_mobilenet_cnn3}}. A
diferença é positiva, mas pequena, e equivale a {{so_a_mobilenet_cnn3}} contra {{so_b_mobilenet_cnn3}} imagens discordantes. Ela também é
menor que a variação observada entre sementes da própria CNN-3. Vale ainda lembrar que a MobileNetV2
foi treinada com uma única semente. Por isso, a evidência sugere uma vantagem modesta da
transferência de aprendizado, sem estabelecê-la com firmeza.

### 5.3. Desempenho por classe

A Tabela 6 detalha precisão, *recall* e F1 de cada classe. A CNN-3 melhora as duas classes, mas não
de forma simétrica: o *recall* de ocupadas chega a {{recall_ocupada_cnn}}, enquanto o de livres fica
em {{recall_livre_cnn}}.

<!-- COLS 2.3,1,1,1,1,1,1 -->
{{tabela_classes}}

Tabela 6. Precisão (P), *recall* (R) e F1 por classe no teste. Os suportes estão na Tabela 1.

A Figura 12 mostra essa assimetria nas matrizes de confusão. A CNN-3 classificou
{{cnn_ocupada_livre}} vagas ocupadas como livres e {{cnn_livre_ocupada}} livres como ocupadas.
Na prática, o primeiro erro é o mais incômodo, porque leva um motorista a uma vaga indisponível,
e foi justamente o mais raro. O LBP dividia os erros quase igualmente entre as duas direções, e a
combinação clássica errava mais vagas ocupadas do que livres. O limiar de 0,5 foi mantido em todos
os casos.

![Matrizes de confusão comparadas]({{figuras_t2}}/confusao_comparada.png)

Figura 12. Matrizes de confusão no teste. Cada célula traz a contagem e o percentual da classe
verdadeira.

### 5.4. Custo computacional

O ganho de desempenho tem um preço no treino, mas não necessariamente na inferência. A Tabela 7
separa as duas etapas. "Preparo" é tudo que antecede o classificador: ler e redimensionar a imagem
e, nos métodos clássicos, calcular o descritor.

<!-- COLS 2.2,1.3,1.2,1,1.1,1.1,1 -->
{{tabela_custo}}

Tabela 7. Custo das configurações selecionadas em CPU (AMD Ryzen 7 9800X3D, 8 threads). O ajuste
da CNN inclui todas as épocas e as avaliações de validação. Predição medida em lote, sobre imagens
já carregadas.

A SVM do LBP ajustou-se em {{treino_lbp_s}} s e a da combinação, em {{treino_combinado_s}} s. A
CNN-3 precisou de {{treino_cnn_min}} minutos e a MobileNetV2, de {{treino_mobilenet_min}} minutos.
Considerando todas as configurações testadas, a busca da linha de base clássica somou cerca de
{{busca_t1_s}} s de ajuste, e a das redes, cerca de {{busca_t2_min}} minutos. Na inferência, a situação muda: a CNN-3 gasta
{{total_cnn_ms}} ms por imagem, próximo do LBP ({{total_lbp_ms}} ms) e abaixo da combinação
clássica ({{total_comb_ms}} ms), cujo HOG domina o tempo de preparo. Esses tempos vêm de uma única
execução, sem estimativa de variabilidade. A leitura das imagens pelas redes provavelmente se beneficiou
de arquivos já em cache do sistema operacional, o que favorece a coluna de preparo das redes.

<!-- PAGEBREAK -->

## 6. Robustez: clima e câmera nova

O teste principal mede dias novos das mesmas câmeras. Esta seção examina se a vantagem da CNN-3 se
mantém quando as condições mudam: primeiro entre climas, depois diante de uma câmera ausente do treino.

### 6.1. Desempenho por clima

A Tabela 8 e a Figura 13 dividem o teste por clima. A CNN-3 superou os métodos clássicos em todas
as condições. A maior diferença ocorreu no sol: F1 macro de {{clima_cnn3_sunny}} contra
{{clima_lbp_sunny}} do LBP e {{clima_combinado_sunny}} da combinação. O sol era justamente a
condição mais difícil para o LBP, possivelmente por causa das sombras fortes. Para a CNN-3, a condição mais
difícil foi a chuva ({{clima_cnn3_rainy}}, IC {{clima_ic_cnn3_rainy}}), embora ainda bem acima dos
clássicos nessa mesma condição.

<!-- COLS 1.2,1.6,1,1.3,1,1.2 -->
{{tabela_clima}}

Tabela 8. F1 macro por clima no teste, com o número de vagas livres e ocupadas de cada grupo.

![Desempenho por clima]({{figuras_t2}}/clima_comparado.png)

Figura 13. F1 macro por clima com IC 95%. Os grupos têm tamanhos e proporções de classe diferentes.

Dois cuidados limitam essa leitura. Primeiro, a validação não tinha imagens de chuva, de modo que
nenhum modelo foi escolhido considerando essa condição. Segundo, clima, data, iluminação e ocupação
do estacionamento variam juntos; os grupos descrevem onde o desempenho mudou, sem isolar a causa.
No nublado, os intervalos são largos porque o grupo tem apenas 301 imagens e 81 vagas livres.

### 6.2. Generalização para uma câmera nova

No teste principal, todas as câmeras e vagas também aparecem no treino. Um modelo poderia, em
princípio, aprender a aparência de cada vaga em vez do conceito de ocupação. Para medir esse
risco, repetimos o treino nove vezes, cada vez sem uma das câmeras, e testamos cada modelo apenas
nas imagens de teste da câmera retirada. Os hiperparâmetros ficaram fixos: C = 0,01 para as SVMs e
a configuração escolhida da CNN-3. Ao final, cada uma das {{n_teste}} imagens de teste recebeu uma
previsão de um modelo que nunca viu a sua câmera, o que torna pareada a comparação com o teste
principal.

Há duas ressalvas. Como várias câmeras observam a mesma área, 114 das 164 vagas aparecem em mais de
uma câmera; o experimento testa, portanto, um novo ponto de vista, e nem sempre uma vaga nova.
Cada dobra também treina com cerca de 1/9 menos imagens.

<!-- COLS 2.2,1.1,1.1,1.4,1,1.6 -->
{{tabela_generalizacao}}

Tabela 9. F1 macro com a câmera vista no treino (teste principal) e com a câmera retirada do treino,
nas mesmas 1.500 imagens. A diferença é câmera nova menos câmera vista.

A Tabela 9 mostra que a CNN-3 perdeu {{gen_queda_cnn3}} de F1 macro quando a câmera saiu do treino
(IC 95% da diferença: {{gen_diff_ic_cnn3}}). A queda é real, pois o intervalo não inclui zero, mas é pequena: a rede
continuou acertando {{acc_nova_cnn3}} das imagens. A combinação HSV + LBP + HOG perdeu mais,
{{gen_queda_combinado}} (IC 95% da diferença: {{gen_diff_ic_combinado}}), o que sugere que parte do seu desempenho no
teste principal vinha da aparência específica de cada vista. O LBP praticamente não mudou
({{gen_diff_lbp}}, IC 95% {{gen_diff_ic_lbp}}). Seu histograma de textura tem poucas dimensões e a SVM
usa regularização forte, de modo que retirar um nono do treino quase não move a fronteira de decisão:
{{gen_lbp_iguais}} previsões ficaram iguais às do teste principal.

Mesmo com a câmera nova, a CNN-3 ficou {{gen_par_lbp}} acima do LBP (IC 95% {{gen_par_ic_lbp}};
McNemar p {{gen_par_p_lbp}}) e {{gen_par_combinado}} acima da combinação (IC 95%
{{gen_par_ic_combinado}}; p {{gen_par_p_combinado}}). Ela supera, inclusive, o melhor método clássico
avaliado com a câmera vista. A Figura 14 detalha o efeito por câmera. A CNN-3 piorou em
{{gen_quedas_cnn3}} das nove câmeras, e o pior caso foi a {{gen_pior_cnn3}}, que também teve o pior
resultado para os dois métodos clássicos. Essa câmera já era a mais difícil para o LBP no teste
principal, e a Seção 7 mostra um recorte dela em que um carro vizinho ocupa boa parte da vaga. Sem exemplos
dessa vista no treino, esses casos ambíguos passam a pesar mais. A melhor época de cada dobra da CNN-3
variou de {{gen_epocas_cnn}}, e as nove dobras somaram {{gen_tempo_cnn_min}} minutos de
treino em CPU.

![Generalização por câmera]({{figuras_t2}}/generalizacao.png)

Figura 14. F1 macro por câmera do teste, com a câmera vista (círculo) e retirada do treino
(losango). Quando os dois valores coincidem, só o losango aparece. A câmera 2 tem apenas 43 imagens
de teste.

<!-- PAGEBREAK -->

## 7. O que os erros revelam

As métricas dizem quanto cada abordagem erra; esta seção examina onde. Primeiro, comparamos quais
imagens cada modelo errou. Depois, classificamos visualmente os erros e, por fim, usamos o Grad-CAM
para observar em que regiões a CNN-3 baseou algumas decisões.

### 7.1. Erros compartilhados e exclusivos

Das {{n_teste}} imagens de teste, o LBP e a CNN-3 acertaram juntos {{ambos_corretos_lbp}} e
erraram juntos apenas {{ambos_erraram_lbp}}. Outras {{so_classico_errou_lbp}} foram erradas só pelo
LBP e {{so_cnn_errou_lbp}}, só pela CNN-3. Em relação à combinação clássica, os números foram
{{ambos_erraram_combinado}} erros em comum, {{so_classico_errou_combinado}} exclusivos do clássico e
{{so_cnn_errou_combinado}} exclusivos da CNN-3. Os erros da CNN-3, portanto, não são um subconjunto
dos erros clássicos: ela resolve a maior parte deles, mas também falha em imagens que o LBP acertou.

A Figura 15 mostra exemplos das duas situações, sorteados com semente fixa. Na linha de cima, a
CNN-3 acerta vagas livres com sombras de postes, grama e um carro vizinho invadindo o recorte, além
de carros escuros sob chuva. Na linha de baixo, ela erra vagas livres com galhos, manchas de umidade
no asfalto e bordas de canteiro, que o histograma de textura do LBP resolveu.

![Discordâncias entre CNN-3 e LBP]({{figuras_t2}}/discordancias.png)

Figura 15. Imagens em que apenas um dos modelos acerta. Origem: CNRPark+EXT.

### 7.2. Categorias de erro

Para ir além de exemplos isolados, classificamos as {{anotacao_itens}} imagens erradas pelo LBP
ou pela CNN-3 em categorias visuais. A anotação foi feita às cegas quanto ao modelo: as imagens
apareceram embaralhadas, apenas com o rótulo oficial, clima e câmera, e cada uma recebeu uma
categoria principal segundo regras escritas antes da inspeção (relatorios/final/anotacao_regras.md).
Só depois as categorias foram cruzadas com os erros de cada modelo (Tabela 10).

<!-- COLS 2.6,1,1,1,1 -->
{{tabela_anotacao}}

Tabela 10. Categoria visual dos erros de cada modelo. Uma imagem errada pelos dois modelos conta
nas duas colunas.

A CNN-3 eliminou quase todos os erros associados a sombras (de {{anot_sombra_lbp}} para {{anot_sombra_cnn}}) e quase todos os erros
sem causa aparente (de {{anot_sem_causa_lbp}} para {{anot_sem_causa_cnn}}). Este segundo grupo inclui muitas vagas com um carro inteiro e
bem visível que o LBP classificou como livres, como mostram os exemplos de erros do LBP na Figura 16.
Esses casos indicam que o histograma global de textura perdia informação que a CNN-3 consegue usar.

![Erros do LBP]({{figuras}}/erros_vencedor.png)

Figura 16. Erros do LBP + SVM no teste, até seis por direção, amostrados com semente 42.

Os erros que restam na CNN-3 têm outro perfil (Figura 17). A categoria mais frequente é a oclusão
por vegetação ({{anot_oclusao_cnn}} dos {{cnn_erros}} erros), seguida por chuva e sombras. Nas câmeras 3 e 7, galhos e arbustos
cobrem boa parte do veículo ou criam texturas que se confundem com um carro. Uma das imagens
ocupadas da câmera 9 é praticamente preta e também foi errada pelo LBP; nela, a ocupação não pode
ser confirmada visualmente. Na câmera 1, um carro vizinho ocupa grande parte de um recorte rotulado
como livre. A CNN-3 o classificou como ocupado com probabilidade 1,00, situação em que a fronteira
entre erro do modelo e limitação do recorte é discutível.

![Erros da CNN-3]({{figuras_t2}}/erros_cnn.png)

Figura 17. Erros da CNN-3: até seis vagas de cada direção de erro ({{cnn_ocupada_livre}} ocupadas previstas como livres e {{cnn_livre_ocupada}} livres previstas
como ocupadas no total), com a probabilidade de ocupação p. Origem: CNRPark+EXT.

A anotação tem limites claros. Foi feita por um único anotador, cada imagem recebeu uma única
categoria e as categorias se sobrepõem na prática, como sombra e vegetação. Os números indicam
tendências, não frequências precisas, e não provam que a categoria tenha causado o erro.

### 7.3. Onde a CNN-3 procura evidência

A Figura 18 aplica o Grad-CAM a dois exemplos de cada célula da matriz de confusão. Nos acertos de
vagas ocupadas, as regiões destacadas ficam sobre o veículo, em faróis, para-brisa e contorno da
carroceria. Nos acertos de vagas livres, o destaque recai sobre o asfalto e as marcações. Nos
erros, o mapa se espalha: na vaga livre prevista como ocupada com p = 0,58, ele cobre texturas
irregulares do asfalto; na vaga ocupada encoberta por vegetação, os galhos dominam o mapa.

![Grad-CAM no teste]({{figuras_t2}}/gradcam.png)

Figura 18. Grad-CAM da CNN-3 em dois exemplos de cada célula da matriz de confusão: acertos de
livres, acertos de ocupadas, livres previstas como ocupadas e ocupadas previstas como livres.

O Grad-CAM mostra onde a decisão é sensível, em 32 × 32, sem explicar toda a rede; nos exemplos
examinados, a CNN-3 olha para o veículo quando ele existe.

<!-- PAGEBREAK -->

## 8. Discussão

A pergunta central era como uma abordagem clássica se compara a uma rede neural no mesmo problema.
Nas condições avaliadas, a resposta é clara quanto ao desempenho: a CNN-3 superou todas as
representações clássicas em todas as métricas, nas duas classes e nos três climas, por margens
muito maiores que a incerteza estimada. A diferença de F1 macro em relação à melhor combinação
clássica foi de {{diff_cnn3_combinado}}, com intervalo {{diff_ic_cnn3_combinado}}.

A análise de erros confirma o limite antecipado na seção 3.5. Os erros que a CNN-3 eliminou eram,
sobretudo, sombras e carros bem visíveis, situações em que um histograma global confunde o que está
na vaga. A CNN-3 aprende, a partir dos exemplos, combinações de padrões em posições específicas,
como um farol acima de uma placa. Os mapas internos e o Grad-CAM são compatíveis com essa
explicação, embora não a provem.

Isso não torna os métodos clássicos inúteis. O LBP descreve uma vaga com dez números, treina em
milissegundos e é inteiramente interpretável. Em um cenário com pouquíssimos dados rotulados, sem
recursos para treinar redes ou com exigência de explicar cada decisão, essas vantagens pesam. Também
há casos em que o LBP acertou e a CNN-3 errou, sobretudo vagas livres com galhos e manchas, o que
indica que as abordagens capturam informações parcialmente diferentes. O custo de treino é a maior
desvantagem da CNN-3 aqui: cerca de {{treino_cnn_min}} minutos em CPU, contra frações de segundo.
Na inferência, porém, a rede não se mostrou mais lenta que a combinação clássica.

A transferência de aprendizado foi a melhor opção em desempenho e convergiu em duas épocas, mas sua
vantagem sobre a CNN-3 é pequena e se apoia em uma única semente. A escolha entre as duas redes
dependeria mais de restrições práticas, como memória e tempo de treino, do que dos números deste
teste.

O experimento de câmera nova responde a uma objeção natural: a vantagem da CNN-3 poderia vir de
memorizar a aparência de cada vaga vista no treino. Sem a câmera do teste, a rede perdeu
{{gen_queda_cnn3}} de F1 macro e manteve uma vantagem de {{gen_par_lbp}} sobre o LBP, pouco menor
que a obtida no teste principal (Tabela 5). A memorização, portanto, explica no máximo uma parte pequena do resultado. O
experimento não substitui um teste em outro estacionamento, porque as câmeras retiradas ainda
compartilham vagas, iluminação e clima com as que ficaram no treino.

As conclusões têm limites que precisam ficar explícitos. Todas as imagens vêm de um único
estacionamento, com nove câmeras e 23 dias. O teste cobre apenas cinco dias, e o *bootstrap* por
captura não representa a correlação entre imagens do mesmo dia. A validação não tinha chuva e era
mais desbalanceada que o teste. Não exploramos outras arquiteturas, taxas de aprendizado ou
limiares, e a CNN-3 com aumento de dados mostrou que pequenas mudanças de configuração afetam
pouco o resultado, mas isso foi verificado apenas para essa escolha. Os tempos vêm de uma execução
em uma única máquina. Por fim, a anotação dos erros é interpretativa e de um único anotador.

<!-- PAGEBREAK -->

## 9. Conclusão

Este trabalho partiu de uma linha de base clássica e, sob o mesmo protocolo, verificou se uma rede
neural convolucional supera seus limites na classificação de vagas como livres ou ocupadas. A
linha de base chegou a {{f1_combinado}} de F1 macro no teste com HSV + LBP + HOG e SVM linear, e a
{{f1_lbp}} com o LBP escolhido pela validação. Nas mesmas imagens, uma CNN de três blocos treinada
do zero chegou a {{f1_cnn}}, e uma MobileNetV2 pré-treinada, a {{f1_mobilenet}}.

A vantagem das redes foi consistente entre classes, climas e sementes, e a análise pareada mostrou
que ela não depende de poucos casos. Os erros que a CNN-3 eliminou estavam ligados principalmente a
sombras e a carros visíveis que o histograma de textura não reconhecia; os erros que permaneceram
envolvem vegetação sobre os veículos, chuva e recortes de interpretação ambígua.
Quando uma câmera inteira foi retirada do treino, a CNN-3 caiu de {{gen_vista_cnn3}} para
{{gen_nova_cnn3}} e continuou acima dos dois métodos clássicos, o que indica que a rede aprendeu
características de ocupação, e não apenas a aparência de cada vaga.

Quando usar cada abordagem? Nas condições estudadas, com milhares de imagens rotuladas e uma CPU
comum, a CNN é a escolha mais adequada: treina em minutos, classifica uma vaga em poucos
milissegundos e erra muito menos. A abordagem clássica continua útil como referência rápida e
interpretável, ou quando dados e recursos de treino são escassos. Trabalhos futuros poderiam
avaliar estacionamentos diferentes, anotar os erros com mais de um avaliador e testar se a
combinação das duas abordagens corrige os erros que cada uma comete sozinha.

### Reprodução do estudo

O README e docs/reproducao.md apresentam os comandos de cada etapa. O ambiente foi {{ambiente}},
com processador AMD Ryzen 7 9800X3D (identificado pelo sistema como {{processador}}) e PyTorch
{{versao_torch}} [14] com {{threads}} threads. A linha de base clássica foi registrada nos commits
{{commit_treino}} (treino) e {{commit_teste}} (teste). As redes foram treinadas e testadas a partir
do commit {{commit_t2_treino}}, com alterações registradas depois no commit 6926c45; para
compensar, os *hashes* SHA-256 de cada arquivo de código usado no treino e no teste estão em
resultados/t2/metricas/codigo_treino.json e avaliacao.json.

{{tabela_versoes}}

Tabela 11. Versões das principais bibliotecas, as mesmas nas duas etapas. O ambiente completo está
em requirements-lock-t2.txt.

A identidade SHA-256 do manifesto, comum a todos os experimentos, permite conferir se uma
reprodução usa exatamente as mesmas imagens:

`{{manifest_hash}}`

<!-- PAGEBREAK -->

## Referências

[1] Amato, G.; Carrara, F.; Falchi, F.; Gennaro, C.; Meghini, C.; Vairo, C. Deep learning for
decentralized parking lot occupancy detection. Expert Systems with Applications, 72, 327-334, 2017.
DOI: 10.1016/j.eswa.2016.10.055. Dataset: http://cnrpark.it/ .

[2] Efron, B.; Tibshirani, R. J. An Introduction to the Bootstrap. Chapman & Hall/CRC, 1993.

[3] McNemar, Q. Note on the sampling error of the difference between correlated proportions or
percentages. Psychometrika, 12(2), 153-157, 1947. DOI: 10.1007/BF02295996.

[4] Ojala, T.; Pietikäinen, M.; Mäenpää, T. Multiresolution Gray-Scale and Rotation Invariant
Texture Classification with Local Binary Patterns. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 24(7), 971-987, 2002. DOI: 10.1109/TPAMI.2002.1017623.

[5] Dalal, N.; Triggs, B. Histograms of Oriented Gradients for Human Detection. IEEE Conference on
Computer Vision and Pattern Recognition, 2005. DOI: 10.1109/CVPR.2005.177.

[6] Cortes, C.; Vapnik, V. Support-vector networks. Machine Learning, 20, 273-297, 1995.
DOI: 10.1007/BF00994018.

[7] LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11), 2278-2324, 1998. DOI: 10.1109/5.726791.

[8] Ioffe, S.; Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift. International Conference on Machine Learning (ICML), 2015.

[9] Srivastava, N.; Hinton, G.; Krizhevsky, A.; Sutskever, I.; Salakhutdinov, R. Dropout: A Simple
Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15,
1929-1958, 2014.

[10] Kingma, D. P.; Ba, J. Adam: A Method for Stochastic Optimization. International Conference on
Learning Representations (ICLR), 2015. arXiv:1412.6980.

[11] Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; Chen, L.-C. MobileNetV2: Inverted Residuals
and Linear Bottlenecks. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.
DOI: 10.1109/CVPR.2018.00474.

[12] Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; Fei-Fei, L. ImageNet: A large-scale
hierarchical image database. IEEE Conference on Computer Vision and Pattern Recognition, 2009.
DOI: 10.1109/CVPR.2009.5206848.

[13] Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; Batra, D. Grad-CAM: Visual
Explanations from Deep Networks via Gradient-Based Localization. IEEE International Conference on
Computer Vision, 2017. DOI: 10.1109/ICCV.2017.74.

[14] Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library.
Advances in Neural Information Processing Systems 32 (NeurIPS), 2019.

### Onde encontrar as evidências

As evidências da linha de base clássica estão em resultados/t1/, e as das redes e da comparação,
em resultados/t2/. Em resultados/t2/metricas/,
selecao.csv e curvas.csv registram a escolha pela validação; congelamento.json, os pesos congelados;
teste.csv e predicoes/, o lote único de teste; comparacao.csv, pares.csv e clima_comparado.csv, os
intervalos e testes pareados; generalizacao*.csv, o experimento de câmera nova; e
anotacao_erros_cruzada.csv, a anotação cega cruzada com os erros. As figuras das redes estão em
resultados/t2/figuras/. Os dados brutos e os pesos treinados ficam fora do Git; a documentação
explica como obtê-los ou recriá-los.
