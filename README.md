# OneVision YOLO - Estimativa de Pose Animal

Projeto de visão computacional utilizando YOLOv11 (Ultralytics) para detectar coordenadas anatômicas (esqueleto) em caprinos e bovinos. 

## Estrutura do Projeto

* `train.py`: Script de treinamento para o dataset customizado (fine-tuning).
* `trainAP10k.py`: Script para treinamento base utilizando o dataset massivo AP-10K.
* `main.py`: Script de inferência e Prova de Conceito (PoC) para rodar os pesos treinados em novas imagens.

## Como Instalar

1. Clone o repositório:
```bash
git clone [https://github.com/kiro-ma/onevision-yolo.git](https://github.com/kiro-ma/onevision-yolo.git)
cd onevision-yolo
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv .venv
# No Windows: .\.venv\Scripts\activate
# No Linux: source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar
Para testar a inferência, coloque uma imagem na pasta inputs/ e execute:
```bash
python main.py
```
A imagem resultante, contendo o mapeamento anatômico, será gerada na pasta outputs/.