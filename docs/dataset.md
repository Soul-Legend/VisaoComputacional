# Dataset e proveniência

Fonte primária: [CNRPark+EXT](http://cnrpark.it/), de Amato et al.
Utilizamos somente **CNR-EXT**, sem misturar os 12.584 recortes do CNRPark original.

- Arquivo: [CNR-EXT-Patches-150x150.zip](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip).
- Aproximadamente 450 MB comprimidos; 144.965 recortes, nove câmeras, 23 dias.
- Licença indicada pelos autores: [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
- `0` significa livre; `1` significa ocupada. Os rótulos vêm das listas oficiais em `LABELS/`.
- Estrutura oficial: `PATCHES/CLIMA/AAAA-MM-DD/cameraN/CLIMA_DATA_HORA_CAMERA_VAGA.jpg`.

O download fica em `data/raw/`, ignorado pelo Git. A CLI extrai o arquivo de modo seguro,
valida as imagens listadas e salva o SHA-256 do arquivo para conferir a aquisição.
As imagens selecionadas são referenciadas por caminhos relativos no manifesto.

A amostragem segue cotas proporcionais por classe, câmera e clima em cada partição, sem
transferir imagens entre partições oficiais. Semente 42. Alvos: 7.000 / 1.500 / 1.500.
Uma partição menor é usada integralmente. Classes e clima ausentes não são inventados.

Em particular, a lista oficial `LABELS/val.txt` contém 6.346 imagens SUNNY, 12.301 OVERCAST
e nenhuma RAINY. A ausência de chuva na validação antecede a limpeza e a amostragem do projeto.
O treino e o teste oficiais contêm chuva. Portanto, a seleção por validação não mede diretamente
o comportamento nessa condição, embora os modelos tenham imagens chuvosas no treino.

A auditoria registra arquivos ilegíveis, dimensões, hashes de bytes e pixels decodificados,
conflitos de rótulos e sobreposições de data, câmera e captura. Conflitos de rótulos em imagens
idênticas são excluídos. Para duplicatas, preserva-se primeiro treino, depois validação, depois
teste; também se conserva apenas uma cópia dentro da partição. Essa regra precede a amostragem.

O manifesto congelado e seu hash identificam o recorte experimental. Datas, vagas e câmeras
compartilhadas podem induzir correlações; ausência de duplicatas exatas não elimina todas as
imagens semelhantes. Este protocolo não demonstra generalização para estacionamentos novos.
Figuras derivadas devem atribuir a origem ao CNRPark+EXT; dados brutos não são redistribuídos.
