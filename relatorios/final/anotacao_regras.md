# Regras da anotação cega dos erros

Definidas antes de abrir as montagens em `resultados/t2/anotacao/`. Cada item recebe **uma**
categoria principal. As montagens mostram apenas número do item, rótulo oficial, clima e câmera;
não indicam qual modelo errou. Anotador único; análise interpretativa, sem verificação cruzada.

| Código | Quando usar |
| --- | --- |
| `rotulo` | O conteúdo visível contradiz o rótulo oficial (carro inteiro na vaga rotulada livre, ou vaga visivelmente vazia rotulada ocupada). |
| `escuro` | Imagem tão escura ou sem contraste que a ocupação não se confirma visualmente. |
| `oclusao` | Vegetação, poste ou outro objeto cobre parte relevante da vaga. |
| `perspectiva` | Vaga livre com parte de um veículo vizinho dentro do recorte. |
| `parcial` | Vaga ocupada em que o veículo aparece só em parte ou na borda do recorte. |
| `sombra` | Sombra forte e contrastada atravessa a vaga. |
| `chuva` | Reflexos de água, poças ou gotas alteram a aparência da vaga. |
| `sem_causa` | Nenhuma das situações acima é visível. |

Prioridade quando mais de uma se aplica: `rotulo` > `escuro` > `oclusao` > `perspectiva`/`parcial`
> `sombra` > `chuva` > `sem_causa`. O resultado fica em `anotacao_erros.csv` (colunas `item`,
`categoria`) e só depois é cruzado com os erros de cada modelo.
