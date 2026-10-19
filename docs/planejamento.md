# Planejamento do T1

## Objetivo e escopo

Classificar uma imagem recortada de vaga como livre ou ocupada. Comparar cor (HSV), textura
(LBP), forma (HOG) e sua concatenação mantendo o classificador SVM linear fixo.
Sem CNN, vídeo, detector, aplicação web ou pôster nesta entrega.

## Marcos

1. Dados: baixar CNR-EXT, conferir rótulos, auditar imagens e congelar manifesto.
2. Baseline: fechar o fluxo de HOG com SVM; acrescentar HSV, LBP e concatenação.
3. Experimentos: selecionar por validação, congelar e avaliar o lote de modelos no teste.
4. Relatório: interpretar evidências, revisar PDF e documentar reprodução.

Estimativa humana: 30–40 horas somadas da dupla. Um integrante lidera dados/descritores;
outro, treinamento/avaliação. Ambos revisam protocolo e texto. Não há prazo de calendário definido.

## Critérios de conclusão

Código executável, configuração e manifesto preservados, quatro representações comparadas,
referência majoritária, resultados por classe/clima, análise de erros, PDF revisado e instruções.
Não há meta mínima de acurácia. Conclusões devem refletir os resultados, inclusive negativos.

Cada funcionalidade terá seu commit; documentação e infraestrutura terão commits próprios.
Desenvolvimento em `feat/t1-classificacao-vagas`, sem merge automático nem squash.

## Requisitos e escolhas

O plano de ensino fornecido exige Python 3, dupla e entrega no Moodle; avalia adequação dos
métodos (50%) e aplicação (50%). A extensão de 8–12 páginas é proposta do projeto, não norma
oficial. O arquivo fornecido é da disciplina INE5443 — Reconhecimento de Padrões, semestre 2026-2.
Dados e partições serão preservados para a comparação posterior com redes neurais no T2.
