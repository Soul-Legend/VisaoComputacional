# Classificação de ocupação de vagas com métodos clássicos

T1 | INE5443 - Reconhecimento de Padrões | 2026-2

Repositório: Soul-Legend/VisaoComputacional

## 1. Introdução e objetivo

Uma vaga de estacionamento pode mudar bastante de aparência ao longo do dia. A passagem de
um carro altera as cores, as texturas e os contornos da imagem, enquanto sombras, reflexos e
chuva também modificam essas mesmas pistas. O desafio deste trabalho é reconhecer a ocupação
da vaga apesar dessas variações.

Para estudar o problema, partimos de uma tarefa bem delimitada: receber uma imagem já
recortada de uma vaga e classificá-la como livre (0) ou ocupada (1). A localização das
vagas na imagem completa é considerada conhecida. Assim, podemos concentrar a análise na
forma de representar visualmente cada recorte e na qualidade da classificação obtida.

Comparamos três representações: HSV, que descreve cores; LBP, que resume texturas; e HOG,
que representa orientações de gradientes. Também avaliamos a combinação dos três descritores.
Em todos os casos, usamos o mesmo classificador, uma SVM linear. A pergunta central é:
**como descritores de cor, textura e gradientes se comparam na classificação de ocupação
de vagas, considerando desempenho, dimensionalidade e custo computacional?**

Para organizar a discussão, formulamos a hipótese exploratória de que LBP e HOG sejam
menos sensíveis às mudanças de iluminação do que histogramas globais de cor. Reunir essas
informações também pode favorecer a classificação em outros dias. Essa expectativa não
foi registrada antes dos experimentos, e o protocolo não isola o efeito da iluminação;
portanto, os resultados permitem discuti-la, mas não confirmá-la causalmente.

### Visão geral do experimento

Foram utilizadas {{n_total}} imagens reais do CNR-EXT [1], distribuídas entre treino, validação
e teste. O treino serve para aprender o classificador; a validação, para escolher a configuração;
e o teste, para avaliar a escolha em dados reservados. Os descritores foram comparados nas
mesmas imagens, seguindo um protocolo definido antes da consulta ao teste.

O modelo selecionado foi **{{vencedor}} + SVM, com C={{c}}**. Seu F1 macro foi {{val_f1}}
na validação e {{teste_f1}} no teste, com acurácia de {{teste_acc}}. Como referência,
um classificador que sempre prevê a classe mais frequente do treino obteve F1 macro de
{{dummy_f1}}. As seções seguintes explicam como esses números foram obtidos e o que
eles permitem concluir.

Primeiro, apresentamos os métodos com exemplos visuais. Em seguida, descrevemos os dados e
o protocolo experimental, para então analisar os resultados por classe, clima e tipo de erro.
Os detalhes necessários para reproduzir o estudo ficam reunidos ao final e na documentação
do repositório.

O trabalho cobre a abordagem clássica do T1. A comparação com redes neurais ficará para o
T2, usando o mesmo teste. Os resultados atuais se referem a dias distintos de um único
estacionamento; sua aplicação a novos locais exigirá outra avaliação.

<!-- PAGEBREAK -->

## 2. Como os métodos representam a imagem

Antes de classificar uma vaga, precisamos transformar seus pixels em uma descrição numérica.
Cada descritor destaca um aspecto da cena. Para tornar essa diferença concreta, as Figuras
1 a 3 usam o mesmo par de imagens de treino: a vaga 204 da câmera 6, livre às 07h46 e
ocupada às 08h46 de 12/11/2015. Como os horários diferem, os exemplos também apresentam
mudanças de iluminação. Sua finalidade é explicar os métodos.

### 2.1. HSV: descrever a distribuição das cores

O espaço HSV separa três propriedades: H (matiz) identifica a tonalidade; S (saturação)
indica o quanto a cor se afasta do cinza; e V (valor) representa sua intensidade. Na
Figura 1, os canais S e V aparecem em cinza: quanto mais claro o pixel, maior o valor do canal.

![Canais e histogramas HSV]({{ilustracoes}}/metodo_hsv.png)

Figura 1. Canais HSV e seus histogramas para uma vaga livre e ocupada. Origem: CNRPark+EXT.
O mapa de H usa cores para visualizar a matiz; ele não reconstrói a imagem original.

No exemplo, a vaga livre concentra pixels com baixa saturação. A presença do carro distribui
esses valores por mais intervalos e também modifica a intensidade. Em regiões quase cinzentas,
a matiz pode oscilar mesmo quando a diferença de cor é pouco perceptível; por isso, H deve
ser interpretado junto de S.

Para gerar o descritor, dividimos os valores de cada canal em 16 intervalos, chamados bins,
e contamos a fração de pixels em cada um. Cada histograma soma um. Ao reunir os três,
obtemos 48 características. H usa o intervalo [0,180) do OpenCV, e S e V usam [0,256).

Essa representação é compacta, mas perde a posição das cores: imagens diferentes podem
ter histogramas parecidos. A próxima representação procura informação na textura local.

<!-- PAGEBREAK -->

### 2.2. LBP: resumir as texturas locais

Uma textura descreve como as intensidades variam em pequenas regiões. Mesmo sem considerar
a cor, asfalto, vidros, carroceria e detalhes do veículo podem produzir padrões locais
diferentes. O Local Binary Patterns (LBP) explora essa informação comparando cada pixel
com seus vizinhos [3].

Primeiro, convertemos a imagem para cinza. Para cada pixel, comparamos sua intensidade com
oito vizinhos distribuídos em um círculo de raio um. O padrão dessas comparações recebe
um código. Na versão uniforme adotada aqui, os códigos 0 a 8 representam padrões uniformes,
com até duas transições entre comparações positivas e negativas ao percorrer o círculo;
o código 9 reúne os demais padrões.

![Texturas e histogramas LBP]({{ilustracoes}}/metodo_lbp.png)

Figura 2. Da imagem em cinza ao mapa de códigos LBP e ao histograma de textura.
As cores indicam códigos locais; não identificam automaticamente carro ou pavimento.

O mapa conserva a posição de cada código e ajuda a enxergar onde os padrões mudam. No
exemplo ocupado, é possível acompanhar estruturas do veículo. Já o vetor entregue ao
classificador contém somente a frequência dos códigos: contamos quantas vezes cada um
aparece e normalizamos o histograma para soma um. São P + 2 = 8 + 2 = 10 bins,
portanto dez características.

Os gráficos mostram que a distribuição desses padrões difere entre os dois recortes.
Isso ilustra a informação disponível ao modelo, mas dois exemplos não bastam para afirmar
que uma textura identifica a classe em todas as condições. Sombras, vegetação e marcações
também podem modificar as frequências.

LBP oferece uma descrição pequena e rápida de extrair. Em contrapartida, seu histograma
global perde a localização das texturas e, nesta configuração, examina apenas uma escala.
Para preservar mais informação sobre a organização dos contornos, avaliamos também o HOG.

<!-- PAGEBREAK -->

### 2.3. HOG: representar contornos e suas orientações

O gradiente mede a mudança de intensidade entre pixels próximos. Mudanças fortes aparecem
em bordas, como nos limites da carroceria e dos vidros. O Histograms of Oriented Gradients
(HOG) resume a direção dessas mudanças em pequenas regiões da imagem [2].

![Gradientes e representação HOG]({{ilustracoes}}/metodo_hog.png)

Figura 3. Recortes originais, intensidade dos gradientes e orientações por célula.
O contraste foi realçado com a mesma escala nos dois exemplos, somente para visualização.

Os contornos do carro produzem várias orientações; marcações também geram gradientes na
vaga livre. Usamos células 8 × 8, nove orientações, blocos 2 × 2 e L2-Hys. Para recortes
128 × 128, são 8.100 características. O desenho mostra orientações por célula; o vetor
do treinamento inclui a normalização dos blocos.

### 2.4. Das características à classificação

![Fluxo de classificação]({{ilustracoes}}/fluxo_classificacao.png)

Figura 4. O mesmo fluxo é usado para cada descritor e para sua combinação.

Manter a SVM linear concentra a comparação no efeito dos descritores [4]. É uma escolha
de custo controlado para HOG e HSV + LBP + HOG, com vetores de alta dimensão. Os resultados
continuam condicionados a esse classificador, sem estabelecer superioridade sob qualquer modelo.

StandardScaler usa estatísticas do treino. LinearSVC emprega regularização L2, perda
quadrática hinge e C para penalizar erros. RBF poderia melhorar o ajuste, mas acrescentaria
a busca de gamma e a comparação de kernels, ampliando o escopo além dos descritores.

HSV + LBP + HOG soma 8.158 componentes, padronizados individualmente e sem pesos por bloco.

<!-- PAGEBREAK -->

## 3. Dataset e preparação dos dados

O CNRPark+EXT foi disponibilizado por Amato e colaboradores para estudar a ocupação de
vagas [1]. Utilizamos apenas sua extensão CNR-EXT, composta por 144.965 recortes de nove
câmeras, capturados em 23 dias. As imagens estão organizadas por clima: SUNNY (ensolarado),
OVERCAST (nublado) e RAINY (chuvoso). As listas dos autores fornecem os rótulos e as
partições de treino, validação e teste.

Antes da amostragem, conferimos todas as imagens dessas listas. Encontramos {{ilegíveis}}
imagens ilegíveis e excluímos {{conflitos}} registros com pixels idênticos e rótulos
conflitantes. Foram removidas {{duplicatas}} duplicatas adicionais. A identidade dos
arquivos foi verificada pelo conteúdo e pelos pixels decodificados, incluindo suas dimensões.
Essa limpeza ocorreu antes do treinamento.

Em seguida, selecionamos {{n_total}} imagens, preservando aproximadamente as proporções
de classe, câmera e clima dentro de cada partição oficial. A amostragem usa semente 42 e
cotas proporcionais, com distribuição dos maiores restos para fechar os tamanhos exatos.
Nenhuma imagem foi transferida entre as partições.

{{tabela_dados}}

Tabela 1. Composição do subconjunto utilizado no estudo.

![Distribuição dos dados]({{figuras}}/distribuicoes.png)

Figura 5. Quantidade de imagens por classe e clima. A ausência de RAINY na validação
é herdada da partição oficial do dataset.

A lista oficial val.txt contém 6.346 imagens SUNNY, 12.301 OVERCAST e nenhuma RAINY.
A ausência de chuva não foi causada pela limpeza ou pela nossa amostragem: preservamos
a composição dessa lista. O treino contém imagens de chuva, mas a validação não avalia
essa condição durante a escolha do descritor e de C. Isso limita sua representatividade
em relação ao teste, que inclui chuva, e contextualiza a mudança de ranking discutida na seção 6.

O teste também preserva as proporções das listas, sem balanceamento artificial. Como as
proporções das classes variam, usamos F1 macro na seleção, além de apresentar a acurácia total.

O arquivo foi obtido da release oficial dos autores, que informa licença ODbL 1.0. URL,
tamanho e hash do download estão registrados na auditoria. Os exemplos visuais deste
relatório são derivados do CNRPark+EXT e mantêm sua atribuição.

<!-- PAGEBREAK -->

## 4. O que muda entre treino e teste

![Exemplos do treinamento]({{figuras}}/exemplos_treino.png)

Figura 6. Exemplos de treino nas duas classes e em diferentes condições climáticas.
Origem: CNRPark+EXT, subconjunto CNR-EXT.

Os exemplos mostram que a classificação depende de mais do que a presença de uma grande
região correspondente a um carro. A perspectiva muda entre câmeras; árvores e outros
elementos encobrem partes das vagas; e sombras podem alterar a aparência do pavimento.
Essas variações ajudam a compreender por que descritores diferentes podem cometer erros
diferentes.

Todas as imagens são redimensionadas para 128 × 128 antes da extração, para manter a
dimensão dos vetores constante. A leitura converte explicitamente BGR para RGB. HOG e LBP
recebem a versão em cinza inteiro; HSV utiliza a imagem colorida. Não aplicamos aumento
de dados ou transformações aleatórias.

Na amostra final, treino e teste compartilham {{sobreposicao_camera}} câmeras e
{{sobreposicao_slot}} identificadores de vagas. Por outro lado, há {{sobreposicao_date}}
datas e {{sobreposicao_capture}} capturas de câmera e horário em comum entre eles.
Portanto, o experimento avalia dias diferentes das mesmas câmeras e vagas.

Essa distinção orienta a leitura dos resultados. A separação por dias reduz a proximidade
temporal entre treino e teste, mas o fundo e a geometria das vagas permanecem reconhecíveis.
Para avaliar uma câmera ou um estacionamento novo, seria necessário reservar esses cenários
em outro protocolo. O estudo atual fornece uma referência para o problema nas condições
descritas, sem demonstrar essa generalização mais ampla.

<!-- PAGEBREAK -->

## 5. Como os modelos foram comparados

### Aprender, escolher e avaliar

Cada configuração foi ajustada somente nas {{n_treino}} imagens de treino. O StandardScaler
aprende a média e o desvio nesse conjunto e apenas transforma validação e teste. Para cada
representação, avaliamos quatro valores de C: 0,01; 0,1; 1; 10. Os parâmetros dos
descritores permaneceram fixos, conforme a seção 2.

A escolha considera o maior F1 macro na validação, arredondado a quatro casas decimais.
Empates favorecem o vetor de menor dimensão e depois o menor C. A regra define a melhor
configuração de cada representação e o modelo escolhido para a discussão principal. Os
modelos não foram reajustados com os dados de validação.

Depois da escolha, os modelos e suas configurações foram congelados. Só então executamos
um lote de avaliação no teste, usando as mesmas imagens para todos. O resultado desse lote
não foi usado para ajustar os métodos. Incluímos ainda uma referência simples: prever
sempre a classe mais frequente no treino, para verificar o quanto cada método supera
uma decisão que não observa a imagem.

### Como interpretar as métricas

A acurácia é a fração total de previsões corretas. Ela oferece uma visão geral, mas
pode ocultar diferenças entre as classes. Por exemplo, prever sempre ocupada acerta muitos
casos quando essa classe é predominante, embora nenhuma vaga livre seja reconhecida.

A precisão de uma classe indica quantas de suas previsões estão corretas. O recall
indica quantos exemplos reais dessa classe foram encontrados. O F1 combina precisão
e recall pela média harmônica. Por fim, o F1 macro calcula a média dos F1 das duas
classes, atribuindo a elas o mesmo peso. Por esse motivo, ele foi adotado para a seleção.

Nas matrizes de confusão, as linhas representam os rótulos reais e as colunas, as previsões.
Também apresentamos métricas por clima, acompanhadas do tamanho e da composição de cada
grupo. Os rótulos usados nos cálculos são sempre [0,1], com resultado zero quando um
denominador é nulo. As pontuações da SVM não são tratadas como probabilidades calibradas.

### Condições de execução

O SVM usa semente 42, escolha automática do modo dual, tolerância 1e-4 e limite de 20.000
iterações. Uma falha de convergência interrompe o ajuste. A extração mede leitura,
verificação da imagem, redimensionamento e cálculo do descritor; o treinamento inclui
a padronização; e a inferência mede padronização e previsão em lote. Os tempos por
imagem são médias desse lote, sem estimativa de variabilidade entre execuções.

O manifesto e os artefatos são identificados por hashes. O código verifica sua integridade
e bloqueia a sobrescrita do teste já avaliado. As instruções de reprodução estão no
repositório, junto dos registros de parâmetros e versões.

<!-- PAGEBREAK -->

## 6. Resultados de validação e teste

A Tabela 2 apresenta todos os ajustes e a escolha feita antes de abrir o teste.

{{tabela_validacao}}

Tabela 2. F1 macro na validação e tempo de ajuste, incluindo a padronização.

A validação selecionou {{vencedor}} + SVM, C={{c}}, com F1 macro {{val_f1}}, quase
empatado com HSV + LBP + HOG ({{val_f1_combinado}}), sem demonstração de superioridade estatística.

{{tabela_teste}}

Tabela 3. Avaliação nas mesmas {{n_teste}} imagens reservadas para teste.

O modelo selecionado alcançou F1 macro {{teste_f1}} e acurácia {{teste_acc}}.
O ganho de F1 macro sobre a referência majoritária foi {{ganho_f1}}. {{nota_ranking}}

A validação tem {{ocupadas_validacao}} de vagas ocupadas e {{overcast_validacao}} de imagens
nubladas, sem chuva; o teste, {{ocupadas_teste}} de ocupadas e {{rainy_teste}} de chuva.
No teste, HSV + LBP + HOG supera LBP no sol ({{f1_combinado_sunny}} contra {{f1_lbp_sunny}})
e na chuva ({{f1_combinado_rainy}} contra {{f1_lbp_rainy}}), mas não no nublado
({{f1_combinado_overcast}} contra {{f1_lbp_overcast}}). Isso é compatível com uma contribuição
complementar dos descritores nos dias avaliados; a composição não explica causalmente a mudança.

<!-- PAGEBREAK -->

## 7. O desempenho de cada classe

![Comparação no teste]({{figuras}}/comparacao.png)

Figura 7. Acurácia e F1 macro no teste. As diferenças são descritivas, sem teste de significância.

{{tabela_classes}}

Tabela 4. Precisão (P), recall (R) e F1 por classe. Os suportes estão na Tabela 1.

![Matriz de confusão]({{figuras}}/confusao_vencedor.png)

Figura 8. Matriz do modelo selecionado: contagens e percentuais por classe verdadeira.

O modelo classificou {{ocupada_livre}} vagas ocupadas como livres e {{livre_ocupada}}
livres como ocupadas. O primeiro erro pode encaminhar alguém a uma vaga indisponível;
o segundo pode ocultar uma oportunidade real. O recall da classe ocupada foi
{{recall_ocupada}}. Mantivemos o limiar padrão, sem favorecer uma direção de erro.

<!-- PAGEBREAK -->

## 8. Custo computacional e condições climáticas

{{tabela_tempos}}

Tabela 5. Custo de cada configuração selecionada. O ajuste corresponde ao melhor C de
cada representação; o tempo total da busca é maior.

LBP descreve cada imagem com dez valores, HOG com 8.100 e HSV + LBP + HOG com 8.158.
O vetor conjunto recalcula os três descritores. Ainda assim, sua extração medida foi menor
que a de HOG isolado: {{tempo_combinado}} contra {{tempo_hog}} ms/imagem. Como há uma
única medição por configuração e nenhuma estimativa de variabilidade, essa diferença não
evidencia vantagem intrínseca de velocidade. Cache, carga e outras variações de execução
são explicações possíveis, não causas verificadas. Uma comparação de eficiência precisaria
de repetições em condições controladas; a latência de aplicação exigiria o fluxo completo.

{{tabela_clima}}

Tabela 6. F1 macro por clima, com número de exemplos livres e ocupados em cada grupo.

![Resultados por clima]({{figuras}}/clima.png)

Figura 9. Desempenho dos mesmos modelos congelados nas condições climáticas do teste.

Para o modelo selecionado, o menor F1 macro ocorreu em {{pior_clima}}. A diferença
deve ser lida junto das contagens: os grupos têm tamanhos e proporções de classe diferentes.
Além disso, clima, data, iluminação e composição das cenas variam juntos. Esses resultados
descrevem onde o desempenho mudou, sem isolar a causa ou avaliar um clima inédito.

<!-- PAGEBREAK -->

## 9. O que os erros ajudam a entender

![Exemplos de erros]({{figuras}}/erros_vencedor.png)

Figura 10. Até seis exemplos de cada direção de erro, amostrados com semente 42.
Origem: CNRPark+EXT. Os identificadores estão em erros_selecionados.csv.

{{analise_erros}}

Esses casos ajudam a relacionar os números às imagens, mas não medem a frequência das
possíveis causas. Uma análise sistemática exigiria examinar e anotar um conjunto maior.
Aqui, a inspeção serve para discutir os limites do modelo; ela não foi usada para ajustar
os parâmetros ou modificar o teste.

<!-- PAGEBREAK -->

## 10. Discussão e conclusão

O estudo mostrou que descritores clássicos conseguem distinguir vagas livres e ocupadas
nas condições avaliadas. A configuração {{vencedor}} + SVM, C={{c}}, escolhida na
validação, atingiu {{teste_f1}} de F1 macro no teste e superou a referência majoritária
em {{ganho_f1}}. {{nota_ranking}}

O contraste entre LBP e HSV + LBP + HOG é o principal resultado. LBP foi competitivo com
um vetor compacto, enquanto reunir cor, textura e gradientes favoreceu o desempenho nos
dias de teste. A validação é mais desbalanceada e concentra imagens nubladas; o teste
inclui chuva e mais imagens de sol, condições nas quais o vetor conjunto teve F1 macro
maior. Isso sugere que a complementaridade pode depender das cenas avaliadas, sem provar
qual fator produziu a mudança. F1 macro não é linear: o total não deve ser obtido pela
média ponderada dos F1 de cada clima.

HOG e LBP superaram HSV no teste, resultado compatível com a hipótese exploratória.
Como a iluminação não foi manipulada de forma controlada, não podemos atribuir essa
vantagem especificamente à robustez luminosa. Desempenho, dimensão e custo devem ser
considerados juntos; o experimento não define um vencedor universal.

Há limitações importantes: as imagens vêm de um único estacionamento, com as mesmas
câmeras e vagas nas partições; utilizamos uma amostragem e uma execução de tempo; e não
calculamos intervalos de confiança. Um bootstrap futuro deverá considerar a correlação
entre imagens e os agrupamentos de captura/dia, em vez de assumir imagens independentes.
Também não investigamos outros kernels, parâmetros
dos descritores ou pesos para os blocos da combinação. As conclusões ficam restritas às
configurações e ao subconjunto apresentados.

Para o T2, o próximo passo é comparar essa referência clássica com uma CNN, preservando
o manifesto, as classes, as métricas e o teste. Uma avaliação de câmeras ou estacionamentos
novos deverá ser planejada separadamente.

### Reprodução do estudo

O README e docs/reproducao.md apresentam os comandos. O ambiente foi {{ambiente}};
o processador é identificado pelo sistema como {{processador}}. Os commits que registram
o código de treinamento e avaliação são {{commit_treino}} e {{commit_teste}}, respectivamente.

{{tabela_versoes}}

O arquivo requirements-lock.txt registra o ambiente completo. A identidade SHA-256 do
manifesto permite conferir se uma reprodução utiliza exatamente as mesmas imagens:

`{{manifest_hash}}`

<!-- PAGEBREAK -->

## Referências

[1] Amato, G.; Carrara, F.; Falchi, F.; Gennaro, C.; Meghini, C.; Vairo, C.
Deep learning for decentralized parking lot occupancy detection. Expert Systems with
Applications, 72, 327-334, 2017. DOI: 10.1016/j.eswa.2016.10.055.
Dataset: http://cnrpark.it/ . Arquivos oficiais: github.com/fabiocarrara/deep-parking.

[2] Dalal, N.; Triggs, B. Histograms of Oriented Gradients for Human Detection.
IEEE Conference on Computer Vision and Pattern Recognition, 2005.
DOI: 10.1109/CVPR.2005.177.

[3] Ojala, T.; Pietikäinen, M.; Mäenpää, T. Multiresolution Gray-Scale and Rotation
Invariant Texture Classification with Local Binary Patterns. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 24(7), 971-987, 2002. DOI: 10.1109/TPAMI.2002.1017623.

[4] Cortes, C.; Vapnik, V. Support-vector networks. Machine Learning, 20, 273-297,
1995. DOI: 10.1007/BF00994018.

### Implementações utilizadas

OpenCV: docs.opencv.org . scikit-image: scikit-image.org/docs/stable/api/skimage.feature.html .
scikit-learn: scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html .
As versões efetivamente utilizadas constam da seção de reprodução e dos registros do experimento.

### Onde encontrar as evidências

O repositório reúne o manifesto em data/manifests/t1.csv; auditoria, parâmetros, métricas
e tempos em resultados/t1/metricas/; e as previsões individuais em resultados/t1/predicoes/.
Os arquivos validacao.csv, congelamento.json, teste.csv e clima.csv permitem acompanhar,
respectivamente, a seleção, a escolha preservada, a avaliação final e sua divisão por clima.

As figuras de resultados ficam em resultados/t1/figuras/. As ilustrações dos métodos estão
em docs/imagens/, junto dos identificadores e hashes dos exemplos de treino. Os comandos
para recriá-las e a explicação de cada painel estão em docs/metodos.md.

A fonte do relatório é Markdown, acompanhada de referencias.bib e do PDF. Os dados brutos
e os modelos treinados ficam fora do Git; a documentação explica como obtê-los ou recriá-los.
