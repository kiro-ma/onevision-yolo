from ultralytics import YOLO
from ultralytics.utils import metrics, loss
import numpy as np


def configurar_oks_quadrupedes():
    """
    Atualiza IN-PLACE o vetor OKS_SIGMA tanto em `ultralytics.utils.metrics` quanto em
    `ultralytics.utils.loss` para garantir que a mesma referência de memória usada pela
    `v8PoseLoss` e pelo `PoseValidator` utilize os sigmas calibrados para quadrúpedes.
    (Complementa o campo nativo `kpt_oks_sigmas` do data.yaml no Ultralytics 8.4+).

    Ordem dos 17 keypoints do AP-10K:
    0: L_Eye, 1: R_Eye, 2: Nose, 3: Neck (Cernelha), 4: Root of tail (Garupa),
    5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw,
    8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw,
    11: L_Hip, 12: L_Knee, 13: L_B_Paw,
    14: R_Hip, 15: R_Knee, 16: R_B_Paw

    Nota: Na KeypointLoss da Ultralytics, sigma menor -> denominador (2*sigma^2) menor ->
    gradiente maior e penalidade mais severa para erros de posição e colapso Esquerda/Direita.
    """
    sigmas_quadrupedes = np.array([
        0.025, 0.025, 0.026,  # 0: L_Eye, 1: R_Eye, 2: Nose
        0.035, 0.050,         # 3: Neck (Cernelha), 4: Root of tail (Garupa)
        0.065, 0.030, 0.020,  # 5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw
        0.065, 0.030, 0.020,  # 8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw
        0.065, 0.030, 0.020,  # 11: L_Hip, 12: L_Knee, 13: L_B_Paw
        0.065, 0.030, 0.020,  # 14: R_Hip, 15: R_Knee, 16: R_B_Paw
    ], dtype=metrics.OKS_SIGMA.dtype)

    # Mutação in-place preserva o ponteiro de memória já importado em ultralytics.utils.loss
    metrics.OKS_SIGMA[:] = sigmas_quadrupedes
    if hasattr(loss, "OKS_SIGMA"):
        loss.OKS_SIGMA[:] = sigmas_quadrupedes


def treinar_bovidae_finetune():
    """
    Caminho 1: Refinamento de precisão (Stage-2 Fine-Tuning) partindo do melhor checkpoint
    quadrupedal já convergido (`caprine-bovidae-v2-oclusao/weights/best.pt`, mAP50=0.80).

    Vantagens desta configuração:
    - Preserva 100% do conhecimento prévio das 10.015 imagens do AP-10K.
    - Treina apenas nas 526 imagens estritas (Bovidae + Cervidae em pé, >=3 patas visíveis, sem close-ups).
    - Entra em `close_mosaic` já na época 55 (de 90), garantindo 35 épocas de ajuste fino sem cortes.
    - Consome apenas ~3.2 GB de VRAM com workers=2, evitando paginação em RAM e travamento do VSCode.
    """
    # 1. Atualiza in-place os sigmas da OKS Loss e mAP de validação
    configurar_oks_quadrupedes()

    # 2. Parte do melhor checkpoint quadrupedal já convergido (evita choque de domínio humano->animal)
    model = YOLO('./runs/pose/onevision-yolo/caprine-bovidae-v2-oclusao/weights/best.pt')

    results = model.train(
        data='./dataset_versions/ap-10k-bovidae/data.yaml',

        # ── Ciclo de Treino Enxuto com Fase Longa Pós-Mosaic ─────────────────
        epochs=90,           # Rápido (~20-25 min) pois já parte de mAP50=0.80
        patience=50,         # Garante que o modelo atravesse a fase pós-mosaic (épocas 55 a 90)
        batch=8,             # ~3.2 GB de VRAM dedicada (zero uso de RAM compartilhada no Windows)
        imgsz=1280,          # Alta resolução essencial para separar patas paralelas
        close_mosaic=35,     # Desliga Mosaic na época 55 (35 épocas inteiras de refino em 1280px puro!)

        # ── Pesos da Função de Loss (Penalização de Pose e Visibilidade) ─────
        box=7.5,             # Peso padrão de bounding box
        cls=0.5,             # Peso padrão de classe
        pose=16.0,           # Prioriza precisão espacial das articulações das patas
        kobj=2.5,            # Força a rede a cravar visibilidade/oclusão de cada membro

        # ── Transfer Learning e Otimizador ───────────────────────────────────
        freeze=0,            # Backbone descongelado para ajuste fino de sombra/oclusão
        optimizer='AdamW',
        cos_lr=True,         # Decaimento cosseno suave da época 1 até a 90
        lr0=0.00025,         # LR suave para polir o checkpoint v2 sem destruir pesos já aprendidos
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # ── Data Augmentation Focada em Oclusão e Vistas Laterais ────────────
        auto_augment=None,   # Desativa RandAugment padrão do Ultralytics 8.4
        mosaic=0.35,         # Leve até a época 55, depois 0.0 até a época 90
        mixup=0.0,           # Desativado: sobrepor transparências confunde patas E/D
        copy_paste=0.0,      # Desativado
        flipud=0.0,          # CRÍTICO: Animais não ficam de cabeça para baixo
        fliplr=0.5,          # Balanceia vistas laterais esquerda e direita com flip_idx correto
        degrees=5.0,         # Rotação leve para simular inclinações naturais do terreno
        translate=0.1,       # Translação moderada
        scale=0.20,          # Escala conservadora para manter resolução das canelas/cascos
        shear=2.0,           # Leve cisalhamento simula pequenas variações de ângulo lateral
        perspective=0.0,     # Desativado para não distorcer medidas biométricas lineares
        erasing=0.15,        # Random Erasing moderado para robustez a auto-oclusão

        # ── Ajustes Fotométricos (Sombra nas Patas em Segundo Plano) ─────────
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.45,          # Ajuda a distinguir a pata em 2º plano (mais escura/sombreada)

        # ── Hardware e Saída (Seguro para Windows + VSCode) ──────────────────
        device=0,
        workers=2,           # Evita sobrecarga de RAM/Pagefile no Windows (que fechou o VSCode antes)
        project='onevision-yolo',
        name='caprine-bovidae-v4-strict-clean',
        exist_ok=False,
        save_period=-1,      # Salva apenas best.pt e last.pt (evita dezenas de arquivos pesados no disco)
        plots=True,
    )
    return results


if __name__ == '__main__':
    treinar_bovidae_finetune()




