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

Datasets, caches e modelos não são versionados. Não há CNN nem detector de vagas neste T1.
