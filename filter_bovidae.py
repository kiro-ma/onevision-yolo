"""
filter_bovidae.py
-----------------
Filtra o dataset AP-10K localmente, copiando apenas as imagens e labels
de animais das famílias Bovidae e Cervidae que possuem pose quadrupedal
completa e em pé (exigindo cernelha/neck, garupa/root_of_tail e pelo menos
3 patas visíveis), limpando close-ups de cabeça e animais deitados/ocultos.

Uso:
    python filter_bovidae.py

Saída:
    dataset_versions/ap-10k-bovidae/
        train/images/
        train/labels/
        valid/images/
        valid/labels/
        data.yaml
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

# ── Configuração ──────────────────────────────────────────────────────────────
AP10K_DIR = Path("dataset_versions/ap-10k")
OUT_DIR   = Path("dataset_versions/ap-10k-bovidae")

SUPERCATEGORIAS_ALVO = {"Bovidae", "Cervidae"}

# Índices dos keypoints no AP-10K (formato [x0, y0, v0, x1, y1, v1, ...])
IDX_NECK         = 3
IDX_ROOT_OF_TAIL = 4
IDX_PAWS         = (7, 10, 13, 16)  # L_F_Paw, R_F_Paw, L_B_Paw, R_B_Paw
MIN_PAWS_VISIVEIS = 3

# Como os diretórios train/ (8012 imgs) e valid/ (2003 imgs) no disco já estão
# particionados fisicamente, lemos os 3 arquivos do split1 (que cobrem 100% das
# 10.015 imagens do AP-10K sem triplicar contagens) e cruzamos com cada pasta.
ANNOTATION_FILES = [
    "ap10k-train-split1.json",
    "ap10k-val-split1.json",
    "ap10k-test-split1.json",
]

SPLITS = {
    "train": {
        "img_dir": AP10K_DIR / "train" / "images",
        "lbl_dir": AP10K_DIR / "train" / "labels",
        "out_img": OUT_DIR / "train" / "images",
        "out_lbl": OUT_DIR / "train" / "labels",
    },
    "valid": {
        "img_dir": AP10K_DIR / "valid" / "images",
        "lbl_dir": AP10K_DIR / "valid" / "labels",
        "out_img": OUT_DIR / "valid" / "images",
        "out_lbl": OUT_DIR / "valid" / "labels",
    },
}


# ── Funções ───────────────────────────────────────────────────────────────────
def validar_anotacao_anatomica(keypoints: list) -> bool:
    """
    Verifica se a anotação possui geometria útil para biometria lateral:
    - neck (cernelha/pescoço, idx 3) visível (v > 0)
    - root_of_tail (garupa, idx 4) visível (v > 0)
    - pelo menos 3 das 4 patas (idxs 7, 10, 13, 16) visíveis (v > 0)
    """
    if not keypoints or len(keypoints) < 51:
        return False

    neck_visivel = keypoints[IDX_NECK * 3 + 2] > 0
    tail_visivel = keypoints[IDX_ROOT_OF_TAIL * 3 + 2] > 0
    patas_visiveis = sum(1 for idx in IDX_PAWS if keypoints[idx * 3 + 2] > 0)

    return neck_visivel and tail_visivel and (patas_visiveis >= MIN_PAWS_VISIVEIS)


def coletar_arquivos_filtrados(split_cfg: dict) -> tuple[set, dict, dict]:
    """
    Varre os arquivos de anotação COCO do AP-10K e retorna apenas os file_names
    presentes no split de disco que pertençam a Bovidae/Cervidae e atendam à
    regra estrita de visibilidade anatômica.
    """
    img_src_dir = split_cfg["img_dir"]
    filenames_validos = set()
    especie_count = defaultdict(int)
    stats = {"total_candidatas": 0, "rejeitadas_anatomia": 0, "aceitas": 0}

    # Agrupa anotações da supercategoria-alvo por imagem presente neste split
    img_annotations = defaultdict(list)

    for ann_file in ANNOTATION_FILES:
        ann_path = AP10K_DIR / "annotations" / ann_file
        if not ann_path.exists():
            print(f"  [AVISO] Arquivo não encontrado: {ann_path}")
            continue

        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)

        cat_map = {
            c["id"]: (c["name"], c["supercategory"])
            for c in data["categories"]
        }
        cat_ids_alvo = {
            cid for cid, (_, supercat) in cat_map.items()
            if supercat in SUPERCATEGORIAS_ALVO
        }
        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        for ann in data["annotations"]:
            if ann["category_id"] in cat_ids_alvo:
                fname = img_map.get(ann["image_id"])
                if fname and (img_src_dir / fname).exists():
                    nome_especie = cat_map[ann["category_id"]][0]
                    img_annotations[fname].append((nome_especie, ann.get("keypoints", [])))

    for fname, anns in img_annotations.items():
        stats["total_candidatas"] += 1
        anns_validas = [esp for esp, kpts in anns if validar_anotacao_anatomica(kpts)]

        # Rejeita também imagens com close-ups puros de cabeça misturados (0 patas visíveis)
        tem_closeup_sem_pata = any(
            sum(1 for idx in IDX_PAWS if kpts[idx * 3 + 2] > 0) == 0
            for _, kpts in anns if len(kpts) >= 51
        )

        if anns_validas and not tem_closeup_sem_pata:
            filenames_validos.add(fname)
            stats["aceitas"] += 1
            for esp in anns_validas:
                especie_count[esp] += 1
        else:
            stats["rejeitadas_anatomia"] += 1

    return filenames_validos, especie_count, stats


def copiar_split(split_name: str, split_cfg: dict) -> dict:
    """Limpa o diretório antigo e copia apenas imagens e labels aprovados no filtro estrito."""
    print("\n" + "-" * 60)
    print(f"  Processando split: {split_name.upper()}")
    print("-" * 60)

    filenames, especie_count, stats = coletar_arquivos_filtrados(split_cfg)
    print(f"  Imagens Bovidae/Cervidae no split: {stats['total_candidatas']}")
    print(f"  Rejeitadas (close-up/deitado/<3 patas): {stats['rejeitadas_anatomia']}")
    print(f"  Aprovadas (corpo inteiro em pé):        {stats['aceitas']}")
    print("\n  Espécies e contagem de instâncias válidas:")
    for esp, qtd in sorted(especie_count.items(), key=lambda x: -x[1]):
        print(f"    {esp:<22} {qtd:>5} anotações")

    img_src_dir = split_cfg["img_dir"]
    lbl_src_dir = split_cfg["lbl_dir"]
    out_img_dir = split_cfg["out_img"]
    out_lbl_dir = split_cfg["out_lbl"]

    # Remove arquivos antigos e caches do YOLO para garantir que imagens rejeitadas não permaneçam
    if out_img_dir.exists():
        shutil.rmtree(out_img_dir)
    if out_lbl_dir.exists():
        shutil.rmtree(out_lbl_dir)
    cache_file = out_lbl_dir.with_suffix(".cache")
    if cache_file.exists():
        cache_file.unlink()

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

    print(f"\n  OK Copiados:        {copiados} pares imagem+label")
    if sem_imagem:
        print(f"  AVISO Sem imagem:   {sem_imagem}")
    if sem_label:
        print(f"  AVISO Sem label:    {sem_label}")

    return {"copiados": copiados, "rejeitadas": stats["rejeitadas_anatomia"]}


def gerar_data_yaml():
    """
    Gera o data.yaml do subconjunto Bovidae + Cervidae com topologia quadrupedal
    e kpt_oks_sigmas nativo do Ultralytics 8.4 para penalizar desvios nas patas.
    """
    yaml_content = """# Dataset AP-10K estrito — famílias Bovidae e Cervidae (apenas corpo inteiro em pé)
# Exige neck, root_of_tail e >= 3 patas visíveis (sem close-ups de cabeça ou animais deitados)

train: ../train/images
val: ../valid/images

nc: 1
names: ['animal']

kpt_shape: [17, 3]
flip_idx: [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]

# Sigmas OKS customizados por keypoint (nativo Ultralytics PoseTrainer/PoseValidator)
# Valores reduzidos em cotovelos/joelhos (0.030) e patas/cascos (0.020) para punir troca E/D
kpt_oks_sigmas: [
  0.025, 0.025, 0.026,
  0.035, 0.050,
  0.065, 0.030, 0.020,
  0.065, 0.030, 0.020,
  0.065, 0.030, 0.020,
  0.065, 0.030, 0.020
]

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
    print(f"\n  OK data.yaml gerado em: {out_yaml}")


# ── Execução ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AP-10K -> Filtro Estrito (Bovidae + Cervidae | Corpo Inteiro)")
    print(f"  Saída: {OUT_DIR.resolve()}")
    print("=" * 60)

    resultados = {}
    for split_name, split_cfg in SPLITS.items():
        resultados[split_name] = copiar_split(split_name, split_cfg)

    gerar_data_yaml()

    print("\n" + "=" * 60)
    print("  RESUMO FINAL")
    print("=" * 60)
    for split_name, res in resultados.items():
        print(f"  {split_name:<8}: {res['copiados']:>4} pares limpos copiados ({res['rejeitadas']} descartados)")
    print(f"\n  Dataset pronto em: {OUT_DIR.resolve()}")
    print("  Próximo passo: python trainAP10k.py")
    print("=" * 60)

