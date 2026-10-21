"""
filter_bovidae.py
-----------------
Filtra o dataset AP-10K localmente, copiando apenas as imagens e labels
de animais da família Bovidae (antelope, argali sheep, bison, buffalo, cow, sheep)
para um novo dataset YOLO pronto para fine-tuning.

Uso:
    python filter_bovidae.py

Saída:
    dataset_versions/ap-10k-bovidae/
        train/images/   (862 imagens)
        train/labels/   (862 labels .txt)
        valid/images/   (56 imagens)
        valid/labels/   (56 labels .txt)
        data.yaml
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# ── Configuração ──────────────────────────────────────────────────────────────
AP10K_DIR = Path("dataset_versions/ap-10k")
OUT_DIR    = Path("dataset_versions/ap-10k-bovidae")

SUPERCATEGORIA_ALVO = "Bovidae"

# Inclui também Suidae (porco) e Cervidae (cervo/alce) por similaridade morfológica
# com caprinos. Comente as linhas abaixo para manter apenas Bovidae puros.
SUPERCATEGORIAS_EXTRAS = []   # ex: ["Suidae", "Cervidae"]

SPLITS = {
    "train": {
        "annotations": ["ap10k-train-split1.json", "ap10k-train-split2.json", "ap10k-train-split3.json"],
        "img_dir":     AP10K_DIR / "train" / "images",
        "lbl_dir":     AP10K_DIR / "train" / "labels",
        "out_img":     OUT_DIR / "train" / "images",
        "out_lbl":     OUT_DIR / "train" / "labels",
    },
    "valid": {
        "annotations": ["ap10k-val-split1.json", "ap10k-val-split2.json", "ap10k-val-split3.json"],
        "img_dir":     AP10K_DIR / "valid" / "images",
        "lbl_dir":     AP10K_DIR / "valid" / "labels",
        "out_img":     OUT_DIR / "valid" / "images",
        "out_lbl":     OUT_DIR / "valid" / "labels",
    },
}

# ── Funções ───────────────────────────────────────────────────────────────────
def coletar_arquivos_bovidae(split_cfg: dict) -> set:
    """
    Varre os arquivos de anotação COCO do AP-10K e retorna o conjunto
    de file_names (ex: '000000000371.jpg') que pertencem às supercategorias-alvo.
    """
    supercats_alvo = {SUPERCATEGORIA_ALVO} | set(SUPERCATEGORIAS_EXTRAS)
    filenames_alvo = set()
    especie_count  = defaultdict(int)

    for ann_file in split_cfg["annotations"]:
        ann_path = AP10K_DIR / "annotations" / ann_file
        if not ann_path.exists():
            print(f"  [AVISO] Arquivo não encontrado: {ann_path}")
            continue

        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)

        # Mapeia category_id → (nome, supercategoria)
        cat_map = {
            c["id"]: (c["name"], c["supercategory"])
            for c in data["categories"]
        }

        # IDs das categorias alvo
        cat_ids_alvo = {
            cid for cid, (nome, supercat) in cat_map.items()
            if supercat in supercats_alvo
        }

        # image_id → file_name
        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        for ann in data["annotations"]:
            if ann["category_id"] in cat_ids_alvo:
                nome_especie = cat_map[ann["category_id"]][0]
                fname = img_map.get(ann["image_id"])
                if fname:
                    filenames_alvo.add(fname)
                    especie_count[nome_especie] += 1

    return filenames_alvo, especie_count


def copiar_split(split_name: str, split_cfg: dict) -> dict:
    """Copia imagens e labels Bovidae para o diretório de saída."""
    print("\n" + "-"*55)
    print("  Processando split: " + split_name.upper())
    print("-"*55)

    filenames, especie_count = coletar_arquivos_bovidae(split_cfg)
    print(f"  Espécies e contagem de anotações:")
    for esp, qtd in sorted(especie_count.items(), key=lambda x: -x[1]):
        print(f"    {esp:<20} {qtd:>5} anotações")

    img_src_dir = split_cfg["img_dir"]
    lbl_src_dir = split_cfg["lbl_dir"]
    out_img_dir = split_cfg["out_img"]
    out_lbl_dir = split_cfg["out_lbl"]

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copiados, sem_label, sem_imagem = 0, 0, 0

    for fname in sorted(filenames):
        img_src = img_src_dir / fname
        lbl_src = lbl_src_dir / fname.replace(".jpg", ".txt")

        if not img_src.exists():
            sem_imagem += 1
            continue
        if not lbl_src.exists():
            sem_label += 1
            continue

        shutil.copy2(img_src, out_img_dir / fname)
        shutil.copy2(lbl_src, out_lbl_dir / fname.replace(".jpg", ".txt"))
        copiados += 1

    print("  OK Copiados:        " + str(copiados) + " pares imagem+label")
    if sem_imagem:
        print("  AVISO Sem imagem:      " + str(sem_imagem))
    if sem_label:
        print("  AVISO Sem label YOLO:  " + str(sem_label))

    return {"copiados": copiados, "sem_imagem": sem_imagem, "sem_label": sem_label}


def gerar_data_yaml():
    """Gera o data.yaml do subconjunto Bovidae com topologia quadrupedal correta."""
    yaml_content = """# Dataset AP-10K filtrado — apenas família Bovidae
# Espécies: antelope, argali sheep, bison, buffalo, cow, sheep
# Morfologia quadrupedal — compatível com caprinos (Capridae)

train: ../train/images
val: ../valid/images

nc: 1
names: ['animal']

kpt_shape: [17, 3]
flip_idx: [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]
skeleton: [
  [0, 1], [0, 2], [1, 2],
  [2, 3],
  [3, 4],
  [3, 5], [5, 6], [6, 7],
  [3, 8], [8, 9], [9, 10],
  [4, 11], [11, 12], [12, 13],
  [4, 14], [14, 15], [15, 16]
]
"""
    out_yaml = OUT_DIR / "data.yaml"
    out_yaml.write_text(yaml_content, encoding="utf-8")
    print("\n  OK data.yaml gerado em: " + str(out_yaml))


# ── Execução ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AP-10K -> Filtro Bovidae")
    print(f"  Saída: {OUT_DIR.resolve()}")
    print("=" * 55)

    resultados = {}
    for split_name, split_cfg in SPLITS.items():
        resultados[split_name] = copiar_split(split_name, split_cfg)

    gerar_data_yaml()

    print("\n" + "="*55)
    print("  RESUMO FINAL")
    print("="*55)
    for split_name, res in resultados.items():
        print("  " + split_name.ljust(8) + ": " + str(res['copiados']) + " pares copiados")
    print("\n  Dataset pronto em: " + str(OUT_DIR.resolve()))
    print("  Proximo passo: python trainAP10k.py --data ap-10k-bovidae")
    print("="*55)
