# OneVision YOLO — Biometria, Avaliação e Estimativa de Distância em Caprinos

Pipeline de visão computacional baseado em **YOLO Pose (Ultralytics)** com topologia anatômica de **17 keypoints do dataset AP-10K** para:

1. **Inferência biométrica autônoma (`main.py`)**: estima altura da cernelha, altura da garupa, comprimento corporal, perímetro torácico, grossura do pescoço e peso vivo (Equação de Schaeffer) a partir de uma imagem lateral do animal.
2. **Avaliação e validação em lote (`evaluate.py`)**: compara as estimativas do modelo contra as medidas reais do gabarito (`dados_biometria.json`), gerando gráficos (dispersão e barras emparelhadas) e relatórios consolidados (JSON e CSV).
3. **Estimativa e auditoria de distância da câmera (`estimate_distance.py`)**: verifica a consistência da distância anotada no nome dos arquivos em `inputs/` (ex.: `007-dist160-v1.jpg`), busca distâncias candidatas alternativas quando há divergência biométrica e salva cópias renomeadas em `inputs/distancia_estimada/`.

---

## Instalação e Preparação do Ambiente

1. Clone o repositório e acesse a pasta do projeto:
```bash
git clone https://github.com/kiro-ma/onevision-yolo.git
cd onevision-yolo
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

---

## Convenção dos Dados de Entrada

* **Imagens (`inputs/`)**: os arquivos de animais catalogados seguem o formato `<id>-dist<cm>-v<n>.jpg` (ex.: `007-dist160-v1.jpg`), onde:
  * `007` corresponde ao campo `id_array` (`7`) no arquivo `dados_biometria.json`;
  * `dist160` indica a distância anotada da câmera até o animal (`160` cm).
* **Gabarito (`dados_biometria.json`)**: contém as medidas biométricas reais de campo (`peso_kg`, `altura_cernelha`, `altura_garupa`, `comprimento`, `circ_do_torax`, `pescoco_circ_medio`, entre outras) indexadas por `id_array`.

---

## Guia de Uso dos Scripts Principais

### 1. `main.py` — Inferência Biométrica em Imagem Única

Executa a detecção dos 17 keypoints anatômicos em uma imagem individual, converte as distâncias em pixels para centímetros com base na distância da câmera, estima as 6 medidas biométricas e salva uma imagem anotada com o esqueleto e o painel lateral de resultados em `outputs/`.

#### Chamada padrão
Recebendo uma imagem de entrada (`inputs/007-dist160-v1.jpg`):
```bash
python main.py --image inputs/007-dist160-v1.jpg
```

Para exibir também a comparação direta com os valores reais de `dados_biometria.json` no terminal e no painel desenhado sobre a imagem:
```bash
python main.py --image inputs/007-dist160-v1.jpg --test-gt dados_biometria.json
```

#### Argumentos CLI (`main.py`)

| Argumento | Padrão no código | Descrição |
| :--- | :--- | :--- |
| `--image` | `inputs/010-dist160-v3.jpg` | Caminho da imagem de entrada do animal. |
| `--model` | `./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt` | Caminho para os pesos `.pt` do modelo YOLO Pose (com fallback automático para checkpoints disponíveis em `runs/pose/onevision-yolo/`). |
| `--dist` | `None` *(automático)* | Distância da câmera ao animal em cm. Se omitida, é extraída automaticamente do padrão `dist(\d+)` no nome do arquivo (ou assume `160.0` cm caso não conste no nome). |
| `--conf` | `0.25` | Limiar mínimo de confiança para validação da detecção/keypoints. |
| `--imgsz` | `1280` | Resolução de inferência do YOLO (preserva precisão nas extremidades/patas). |
| `--test-gt` | `None` | Caminho opcional do arquivo JSON de gabarito (ex.: `dados_biometria.json`) para exibir o comparativo Real vs. Estimado. |

#### Saída gerada
* **Terminal**: previsão das 6 medidas biométricas (e relatório comparativo de erro absoluto/percentual caso `--test-gt` seja informado).
* **Imagem anotada**: salva em `outputs/pose_<nome_imagem>_<tag_modelo>-<n>.jpg` com incremento numérico automático (`-1`, `-2`, ...) para nunca sobrescrever execuções anteriores.

---

### 2. `evaluate.py` — Avaliação e Validação do Modelo em Lote

Filtra automaticamente todas as imagens de uma pasta de entrada cujo prefixo numérico (`006`, `007`, etc.) possua correspondência em `dados_biometria.json`, executa a inferência em lote e calcula **MAE**, **RMSE**, **MAPE (%)** e **Acurácia (`100% - MAPE`)** para cada medida biométrica.

Cada execução cria uma subpasta exclusiva em `outputs/evaluation/run_YYYYMMDD_HHMMSS_<tag_modelo>/` contendo gráficos e arquivos estruturados com todos os parâmetros utilizados.

#### Chamada padrão
Recebendo a pasta de entrada (`inputs/`):
```bash
python evaluate.py --inputs-dir inputs/
```

#### Exemplos adicionais
```bash
# Avaliar apenas medidas específicas (separadas por vírgula) e salvar as imagens de pose anotadas
python evaluate.py --inputs-dir inputs/ --measure peso,altura_cernelha,comprimento --save-poses

# Avaliar as imagens já corrigidas na pasta inputs/distancia_estimada/
python evaluate.py --inputs-dir inputs/distancia_estimada/
```

#### Argumentos CLI (`evaluate.py`)

| Argumento | Padrão no código | Descrição |
| :--- | :--- | :--- |
| `--inputs-dir` | `inputs` | Pasta contendo as imagens de entrada a serem avaliadas. |
| `--model` | `./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt` | Caminho para os pesos `.pt` do modelo YOLO Pose. |
| `--conf` | `0.25` | Limiar de confiança para a inferência do YOLO. |
| `--imgsz` | `1280` | Resolução da imagem na inferência. |
| `--measure` | `all` | Medida(s) para avaliar: `peso`, `altura_cernelha` (ou `altura`), `altura_garupa`, `comprimento`, `circ_torax`, `grossura_pescoco`, combinações separadas por vírgula ou `all` para todas. |
| `--gt` | `dados_biometria.json` | Caminho para o arquivo JSON com as medidas reais (ground truth). |
| `--out-dir` | `outputs/evaluation` | Diretório base onde cada execução cria sua subpasta exclusiva `run_YYYYMMDD_HHMMSS_<tag>/`. |
| `--dist` | `None` *(automático)* | Distância fixa da câmera em cm para todas as imagens (se omitida, deduz `dist(\d+)` do nome de cada arquivo). |
| `--plot-type` | `both` | Formato dos gráficos por medida: `both` (dispersão + barras lado a lado), `scatter` (apenas dispersão) ou `bars` (apenas barras emparelhadas). |
| `--save-poses` | `False` *(flag)* | Se ativado, salva as imagens individuais anotadas com o esqueleto dentro de `outputs/evaluation/<run_id>/poses/`. |

#### Saída gerada (em `outputs/evaluation/run_YYYYMMDD_HHMMSS_<tag>/`)
* `eval_<tag>_<medida>.png`: gráfico individual de cada medida avaliada com dispersão Real vs. Inferido, barras emparelhadas por imagem e painel com todos os parâmetros da run.
* `eval_<tag>_resumo_todas_medidas.png`: gráfico comparativo de acurácia global entre todas as medidas avaliadas (gerado quando mais de uma medida é avaliada).
* `relatorio_<tag>_<medida>.json`: relatório completo contendo parâmetros da run, resumo global e erros detalhados por imagem.
* `resumo_metricas_<tag>_<medida>.csv` e `detalhe_amostras_<tag>_<medida>.csv`: tabelas em CSV com parâmetros e métricas agregadas e por amostra.

---

### 3. `estimate_distance.py` — Estimativa de Distância e Auditoria de Anotações

Audita as imagens da pasta de entrada verificando se as medidas previstas na distância anotada no nome do arquivo (`dist(\d+)`) atingem o limiar mínimo de similaridade em relação ao gabarito (`dados_biometria.json`):

$$\text{sim} = \max\left(0.0,\; 1.0 - \frac{|\text{pred} - \text{real}|}{\text{real}}\right)$$

* Se todas as medidas avaliadas (excluindo as medidas configuradas em `--ignore-measures`) atingirem o limiar (`--sim-threshold`, padrão `0.90`), classifica a imagem como **`ok`**.
* Se alguma medida ficar abaixo do limiar, varre distâncias candidatas em torno da distância anotada (`dist_anotada ± search_range`, com passo `--search-step`), seleciona a distância de melhor ajuste biométrico e classifica como **`ajustada`** (quando há melhora/convergência válida) ou **`sem_melhora`**.
* Imagens sem o padrão `dist(\d+)`, sem registro correspondente no JSON ou sem detecção válida recebem um aviso no log e são classificadas de forma segura como **`sem_dado`**.
* Toda imagem com status **`ajustada`** tem uma cópia salva em `inputs/distancia_estimada/` já renomeada com a nova distância estimada (ex.: `007-dist160-v1.jpg` $\rightarrow$ `inputs/distancia_estimada/007-dist180-v1.jpg`).

#### Chamada padrão
Recebendo a pasta de entrada (`inputs/`):
```bash
python estimate_distance.py --inputs-dir inputs/
```

#### Exemplos adicionais
```bash
# Personalizar limiar de similaridade (ex.: 0.92), faixa/passo de busca e critério de seleção
python estimate_distance.py --inputs-dir inputs/ --sim-threshold 0.92 --search-range 60 --search-step 5 --criterion min_sim

# Ignorar múltiplas medidas na decisão (ex.: pescoço e peso) ou não ignorar nenhuma (--ignore-measures none)
python estimate_distance.py --inputs-dir inputs/ --ignore-measures grossura_pescoco,peso
```

#### Argumentos CLI (`estimate_distance.py`)

| Argumento | Padrão no código | Descrição |
| :--- | :--- | :--- |
| `--inputs-dir` | `inputs` | Pasta contendo as imagens de entrada a serem auditadas. |
| `--model` | `./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt` | Caminho para os pesos `.pt` do modelo YOLO Pose. |
| `--conf` | `0.25` | Limiar de confiança para a inferência do YOLO. |
| `--imgsz` | `1280` | Resolução da imagem na inferência. |
| `--gt` | `dados_biometria.json` | Arquivo JSON com os dados biométricos reais (ground truth). |
| `--out-dir` | `outputs/estimativa-distancias` | Diretório base onde cada execução cria sua subpasta exclusiva `run_YYYYMMDD_HHMMSS_<tag>/`. |
| `--copy-dir` | `inputs/distancia_estimada` | Diretório onde são salvas as cópias renomeadas das imagens com distância ajustada (`status='ajustada'`). |
| `--sim-threshold` | `0.90` | Limiar mínimo de similaridade (`0.0` a `1.0`) para considerar a medida consistente (`0.90` equivale a até 10% de erro relativo). |
| `--ignore-measures` | `grossura_pescoco` | Medidas ignoradas na decisão de consistência/busca, separadas por vírgula (aceita chaves ou aliases como `pescoco`, `peso`, `altura`, ou `none` para avaliar todas). |
| `--search-range` | `60.0` | Alcance de busca em cm para menos e para mais a partir da distância anotada (`dist_anotada ± 60 cm`). |
| `--search-step` | `5.0` | Passo em cm entre as distâncias candidatas testadas na busca. |
| `--min-dist` | `30.0` | Distância mínima positiva permitida na busca (em cm). |
| `--max-dist` | `500.0` | Distância máxima permitida na busca (em cm). |
| `--criterion` | `min_sim` | Critério de escolha da melhor distância candidata: `min_sim` (maior similaridade mínima entre medidas avaliadas), `mean_sim` (maior similaridade média / menor erro relativo médio) ou `converged_then_mean` (maior nº de medidas $\ge$ limiar). |

#### Saída gerada
* **Em `outputs/estimativa-distancias/run_YYYYMMDD_HHMMSS_<tag>/`**:
  * `relatorio_estimativa_distancias_<tag>.png`: painel visual completo contendo cabeçalho de parâmetros, resumo de contagens por status (`ok`, `ajustada`, `sem_melhora`, `sem_dado`), gráfico de Distância Anotada vs. Estimada, gráfico de Similaridade Mínima Antes vs. Depois e tabela detalhada por imagem (indicando medidas avaliadas, medidas ignoradas com `[IGN]` e arquivo copiado).
  * `relatorio_estimativa_distancias_<tag>.json` e `relatorio_estimativa_distancias_<tag>.csv`: dados estruturados completos da execução.
* **Em `inputs/distancia_estimada/`**:
  * Cópias das imagens classificadas como `ajustada`, renomeadas automaticamente com o novo valor `dist<estimada>` em centímetros.
