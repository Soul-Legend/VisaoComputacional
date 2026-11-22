# Reprodução

## Instalação

Python 3.11+; o experimento publicado usa Python 3.12. Execute na raiz do repositório.
No Windows, os exemplos usam diretamente o Python do ambiente, sem exigir ativação.

```powershell
git clone https://github.com/Soul-Legend/VisaoComputacional.git
cd VisaoComputacional
git switch feat/t1-classificacao-vagas
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install --no-deps -e .
```

Para desenvolvimento sem reproduzir as versões exatas:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,report]"
```

Linux/macOS: substitua `.venv\Scripts\python` por `.venv/bin/python`.
O arquivo de versões foi capturado em Windows/Python 3.12; instalações em outras plataformas
podem precisar resolver as dependências de `pyproject.toml` em vez de usar o lock literal.

## Dados oficiais e manifesto publicado

```powershell
.venv\Scripts\python -m visao_computacional.cli baixar
```

O comando baixa cerca de 450 MB da release dos autores e extrai os recortes (144.965 arquivos).
Reserve espaço para o ZIP, imagens extraídas, caches e ambiente (aproximadamente 3 GB).
O ZIP é mantido para auditoria eficiente e sua identidade consta em `aquisicao.json`.
Se o download for interrompido, o arquivo `.part` não é tratado como concluído.

**O manifesto publicado já está congelado.** Depois do download, use esse manifesto diretamente
para reproduzir extração e treinamento. `preparar` recusa sobrescrevê-lo.

Para auditar e reconstruir tudo em um estudo separado, copie `configs/t1.yaml` para
`configs/reproducao.yaml` e altere os caminhos:

```yaml
manifest: data/manifests/reproducao.csv
processed_dir: data/processed/reproducao
models_dir: modelos/reproducao
results_dir: resultados/reproducao
report_dir: relatorios/reproducao
```

Mantenha todos os demais parâmetros e `data_root`. Execute:

```powershell
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml preparar
```

O hash do novo manifesto deve coincidir com o de `resultados/t1/metricas/auditoria.json`.
Como a amostragem ordena os caminhos e fixa a semente, a ordem das listas não altera a seleção.
Mudanças em versões, bytes do arquivo ou algoritmo exigem verificar novamente essa identidade.

## Experimento completo em diretórios separados

Os comandos abaixo usam `configs/reproducao.yaml` para preservar os resultados publicados.

```powershell
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml explorar
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml extrair
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml treinar
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml avaliar
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml figuras
.venv\Scripts\python -m visao_computacional.cli --config configs/reproducao.yaml relatorio
```

`extrair` reaproveita somente caches com assinatura e hash compatíveis. O manifesto, os modelos
congelados e a abertura do teste têm proteção contra sobrescrita. Para outro estudo, escolha
outros diretórios; não apague o marcador do teste para ajustar modelos à mesma avaliação.
Se uma avaliação falhar depois de abrir o teste, preserve o registro da falha e investigue-a;
a proteção é deliberadamente conservadora e exige outro diretório para uma recuperação documentada.

O relatório usa `relatorios/t1/modelo.md` e as tabelas do estudo. Para incluir análise manual,
escreva `analise_erros.md` no `report_dir` escolhido antes de gerar o relatório. Sem esse arquivo,
o texto informa que a inspeção ainda não foi registrada. Não use conclusões do estudo original
como se fossem observações de uma nova execução.

## Verificações

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m ruff format --check src tests
```

Os testes de integração usam imagens sintéticas temporárias para verificar o pipeline e as
proteções. Os resultados científicos em `resultados/t1/` vêm exclusivamente do dataset real.
Métricas devem coincidir com as predições; tempos e bytes de modelos podem variar entre execuções.
O manifesto deve ser o mesmo para a comparação com a CNN no T2.

## PDF

Para recriar as figuras didáticas dos métodos com imagens de treino, execute
`.venv\Scripts\python -m visao_computacional.cli ilustracoes` antes de gerar o relatório.
Veja [a explicação das figuras](metodos.md). Essas imagens ficam em `docs/imagens/`,
separadas das figuras de resultados, e sua geração não reabre o teste.

A CLI renderiza o Markdown gerado com ReportLab. Não requer LaTeX, Word ou serviço externo.
Após qualquer alteração, renderize as páginas com Poppler (`pdftoppm`) ou outro leitor e confira
legibilidade, tabelas, figuras, paginação e correspondência com o Markdown. O alvo de extensão
do relatório é uma escolha do projeto, não uma norma de formatação do professor.

## T2: redes neurais e relatório combinado

O T2 usa o mesmo manifesto, os mesmos dados brutos e as predições congeladas do T1.
Instale o PyTorch para CPU (o experimento publicado usou torch 2.15.0 e torchvision 0.29.2):

```powershell
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-deps -e .
```

`requirements-lock-t2.txt` registra o ambiente completo do T2. Os pesos ImageNet da MobileNetV2
são baixados pelo torchvision (`mobilenet_v2-b0353104.pth`) na primeira execução.
As etapas usam `configs/t2.yaml` por padrão, na ordem abaixo:

```powershell
.venv\Scripts\python -m visao_computacional.cli imagens
.venv\Scripts\python -m visao_computacional.cli redes-treinar
.venv\Scripts\python -m visao_computacional.cli redes-avaliar
.venv\Scripts\python -m visao_computacional.cli generalizacao
.venv\Scripts\python -m visao_computacional.cli comparar
.venv\Scripts\python -m visao_computacional.cli redes-figuras
.venv\Scripts\python -m visao_computacional.cli relatorio-final
```

- `imagens` lê os 10.000 recortes uma vez (128 × 128, RGB) e mede o tempo de preparo.
- `redes-treinar` escolhe pela validação entre CNN-3 sem e com aumento, treina as sementes
  extras e a MobileNetV2 e congela tudo. Leva cerca de 1h30 em CPU de 8 núcleos.
  Rodadas já salvas são reaproveitadas após verificação do hash, o que permite retomar.
- `redes-avaliar` abre o teste uma única vez e recusa repetir a operação.
- `generalizacao` treina, para cada uma das 9 câmeras, SVMs e CNN-3 sem aquela câmera
  (cerca de 1h em CPU). Dobras concluídas são reaproveitadas.
- `comparar` calcula intervalos por bootstrap agrupado por captura e testes pareados.
- `redes-figuras` gera as figuras e as montagens para a anotação cega dos erros em
  `resultados/t2/anotacao/`. As categorias atribuídas ficam em `relatorios/final/anotacao_erros.csv`.
- `relatorio-final` preenche `relatorios/final/modelo.md` e gera Markdown e PDF.

Com a mesma semente, as mesmas 8 threads e a mesma máquina, os pesos se repetem; em outro
processador ou com outra versão do PyTorch, pequenas diferenças numéricas são esperadas.
Para um estudo independente, copie `configs/t2.yaml` e troque `processed_dir`, `models_dir`,
`results_dir` e `report_dir`.
