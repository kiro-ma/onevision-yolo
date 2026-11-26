"""
trainAP10k.py
-------------
Pipeline de treinamento DO ZERO (from scratch) para detecção de pose quadrupedal
(17 keypoints AP-10K) utilizando Ultralytics YOLO Pose a partir de arquivo de
arquitetura (.yaml), sem carregar pesos pré-treinados (.pt).

Preserva integralmente as otimizações calibradas no projeto OneVision:
  - Sigmas OKS customizados in-place para quadrúpedes (cabeça, tronco e membros).
  - Resolução de 1280px para preservar separação de patas e detalhes faciais.
  - Pesos de loss priorizando precisão espacial (pose=16.0) e visibilidade (kobj=2.5).
  - Data augmentation segura para biometria lateral (sem flip vertical, mixup ou perspectiva).
  - Configurações de hardware seguras para Windows + GPU local (batch=8, workers=2, cache=False).

Uso padrão:
    python trainAP10k.py

Exemplo customizando arquitetura, épocas e resolução:
    python trainAP10k.py --model-cfg yolo11s-pose.yaml --epochs 240 --imgsz 1280 --name caprine-v5-scratch
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import loss, metrics


# Sigmas OKS calibrados para anatomia quadrupedal (17 keypoints do AP-10K):
# 0: left_eye, 1: right_eye, 2: nose, 3: neck, 4: root_of_tail,
# 5..7: left front limb, 8..10: right front limb,
# 11..13: left back limb, 14..16: right back limb
OKS_SIGMAS_QUADRUPEDES: tuple[float, ...] = (
    0.025, 0.025, 0.026,  # 0: L_Eye, 1: R_Eye, 2: Nose
    0.035, 0.050,         # 3: Neck (Cernelha), 4: Root of tail (Garupa)
    0.065, 0.030, 0.020,  # 5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw
    0.065, 0.030, 0.020,  # 8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw
    0.065, 0.030, 0.020,  # 11: L_Hip, 12: L_Knee, 13: L_B_Paw
    0.065, 0.030, 0.020,  # 14: R_Hip, 15: R_Knee, 16: R_B_Paw
)


@dataclass
class ConfiguracaoTreino:
    """Parâmetros completos e tipados para o treinamento do zero no YOLO Pose."""

    # Arquitetura base (.yaml) e dataset
    model_cfg: str = "yolo11s-pose.yaml"
    data_yaml: str = "./dataset_versions/ap-10k-bovidae/data.yaml"
    resume: Optional[str] = None

    # Ciclo de treinamento
    epochs: int = 200
    patience: int = 60
    batch: int = 8
    imgsz: int = 1280
    close_mosaic: int = 35
    seed: int = 0

    # Pesos da função de perda (Loss)
    box: float = 7.5
    cls: float = 0.5
    pose: float = 16.0
    kobj: float = 2.5

    # Otimizador e agenda de Learning Rate (ajustados para treino do zero)
    optimizer: str = "AdamW"
    cos_lr: bool = True
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 5.0
    freeze: int = 0

    # Data Augmentation focada em biometria lateral e oclusão
    auto_augment: Optional[str] = None
    mosaic: float = 0.35
    mixup: float = 0.0
    copy_paste: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.20
    shear: float = 2.0
    perspective: float = 0.0
    erasing: float = 0.15

    # Ajustes fotométricos
    hsv_h: float = 0.015
    hsv_s: float = 0.4
    hsv_v: float = 0.45

    # Hardware, cache, logging e saída
    device: str = "0"
    workers: int = 2
    cache: bool = False
    amp: bool = True
    project: str = "onevision-yolo"
    name: str = "caprine-bovidae-v5-scratch"
    exist_ok: bool = False
    save_period: int = -1
    plots: bool = True
    verbose: bool = True
    sigmas_oks: tuple[float, ...] = field(default_factory=lambda: OKS_SIGMAS_QUADRUPEDES)


def configurar_oks_quadrupedes(sigmas: tuple[float, ...] = OKS_SIGMAS_QUADRUPEDES) -> np.ndarray:
    """
    Atualiza IN-PLACE o vetor OKS_SIGMA em `ultralytics.utils.metrics` e `ultralytics.utils.loss`
    para garantir que a mesma referência de memória usada pela `v8PoseLoss` e pelo `PoseValidator`
    utilize os sigmas calibrados para quadrúpedes.

    Nota: Na KeypointLoss da Ultralytics, sigma menor -> denominador (2 * sigma^2) menor ->
    gradiente maior e penalidade mais severa para erros de posição e colapso Esquerda/Direita.
    """
    if len(sigmas) != 17:
        raise ValueError(f"Esperados exatamente 17 sigmas OKS (AP-10K), mas recebidos {len(sigmas)}.")

    sigmas_array = np.array(sigmas, dtype=metrics.OKS_SIGMA.dtype)

    # Mutação in-place preserva o ponteiro de memória já importado em ultralytics.utils.loss
    metrics.OKS_SIGMA[:] = sigmas_array
    if hasattr(loss, "OKS_SIGMA"):
        loss.OKS_SIGMA[:] = sigmas_array

    return sigmas_array


def validar_configuracao(cfg: ConfiguracaoTreino) -> None:
    """Valida parâmetros críticos antes de inicializar a arquitetura e o treinamento."""
    if cfg.resume:
        ckpt_path = Path(cfg.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint para retomada (--resume) não encontrado: '{ckpt_path}'."
            )
        if ckpt_path.suffix.lower() != ".pt":
            raise ValueError(
                f"Para retomar o treinamento, '--resume' deve apontar para um arquivo '.pt' "
                f"(ex.: '.../weights/last.pt'), não '{cfg.resume}'."
            )
        return

    ext_cfg = Path(cfg.model_cfg).suffix.lower()
    if ext_cfg not in {".yaml", ".yml"}:
        raise ValueError(
            f"Para treinar do zero, '--model-cfg' deve apontar para uma arquitetura YAML "
            f"(ex.: 'yolo11n-pose.yaml', 'yolo11s-pose.yaml'), não '{cfg.model_cfg}'."
        )

    data_path = Path(cfg.data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração do dataset não encontrado: '{data_path}'. "
            f"Execute 'python filter_bovidae.py' ou informe '--data' válido."
        )

    if cfg.epochs <= 0:
        raise ValueError(f"'--epochs' deve ser maior que zero (recebido: {cfg.epochs}).")
    if cfg.imgsz <= 0 or cfg.imgsz % 32 != 0:
        raise ValueError(f"'--imgsz' deve ser positivo e múltiplo de 32 (recebido: {cfg.imgsz}).")
    if cfg.close_mosaic < 0 or cfg.close_mosaic > cfg.epochs:
        raise ValueError(
            f"'--close-mosaic' ({cfg.close_mosaic}) deve estar entre 0 e '--epochs' ({cfg.epochs})."
        )


def _registrar_callbacks_logging(model: YOLO) -> None:
    """Registra callbacks leves para acompanhar início e término das épocas de treino."""

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch_atual = int(getattr(trainer, "epoch", 0)) + 1
        total_epocas = int(getattr(trainer, "epochs", 0))
        metricas = getattr(trainer, "metrics", {}) or {}
        pose_map50 = metricas.get("metrics/mAP50(P)")
        pose_map50_95 = metricas.get("metrics/mAP50-95(P)")
        if pose_map50 is not None and pose_map50_95 is not None:
            print(
                f"[OneVision] Época {epoch_atual:03d}/{total_epocas:03d} | "
                f"Pose mAP50={pose_map50:.4f} | Pose mAP50-95={pose_map50_95:.4f}"
            )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _converter_device(device_str: str) -> Any:
    """Converte o identificador de dispositivo da CLI ('0', '0,1', 'cpu') para o formato do Ultralytics."""
    limpo = device_str.strip()
    if limpo.isdigit():
        return int(limpo)
    return limpo


def exibir_resumo_configuracao(cfg: ConfiguracaoTreino) -> None:
    """Imprime um resumo estruturado dos parâmetros de arquitetura, dataset e otimizações."""
    print("=" * 76)
    print("  ONEVISION YOLO POSE — TREINAMENTO DO ZERO (FROM SCRATCH)")
    print("=" * 76)
    print(f"  Arquitetura base (.yaml) : {cfg.model_cfg} (pretrained=False)")
    print(f"  Dataset (data.yaml)      : {cfg.data_yaml}")
    print(f"  Ciclo de treino          : epochs={cfg.epochs} | patience={cfg.patience} | close_mosaic={cfg.close_mosaic}")
    print(f"  Resolução / Lote         : imgsz={cfg.imgsz} | batch={cfg.batch} | amp={cfg.amp}")
    print(f"  Otimizador / LR          : {cfg.optimizer} | lr0={cfg.lr0} | lrf={cfg.lrf} | cos_lr={cfg.cos_lr} | warmup={cfg.warmup_epochs}")
    print(f"  Pesos de Loss            : box={cfg.box} | cls={cfg.cls} | pose={cfg.pose} | kobj={cfg.kobj}")
    print(f"  Augmentation lateral     : mosaic={cfg.mosaic} | fliplr={cfg.fliplr} | flipud={cfg.flipud} | scale={cfg.scale} | erasing={cfg.erasing}")
    print(f"  Hardware / Saída         : device={cfg.device} | workers={cfg.workers} | cache={cfg.cache} | out={cfg.project}/{cfg.name}")
    print("=" * 76)


def treinar_do_zero(cfg: ConfiguracaoTreino) -> Any:
    """
    Inicializa um modelo YOLO Pose do zero a partir da arquitetura YAML informada
    (ou retoma de um checkpoint last.pt se cfg.resume estiver definido)
    e executa o treinamento preservando as otimizações biométricas do projeto.
    """
    validar_configuracao(cfg)
    configurar_oks_quadrupedes(cfg.sigmas_oks)

    if cfg.resume:
        print("=" * 76)
        print("  ONEVISION YOLO POSE — RETOMANDO TREINAMENTO (RESUME)")
        print("=" * 76)
        print(f"  Checkpoint (.pt) : {cfg.resume}")
        print("  Sigmas OKS       : Calibrados para quadrúpedes (AP-10K 17 keypoints)")
        print("=" * 76)

        model = YOLO(cfg.resume, task="pose")
        _registrar_callbacks_logging(model)
        return model.train(resume=True)

    exibir_resumo_configuracao(cfg)

    # Instancia a arquitetura pura (.yaml), sem pesos pré-treinados (.pt)
    model = YOLO(cfg.model_cfg, task="pose")
    _registrar_callbacks_logging(model)

    results = model.train(
        data=cfg.data_yaml,
        pretrained=False,

        # ── Ciclo de Treino e Resolução ───────────────────────────────────────
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        close_mosaic=cfg.close_mosaic,
        seed=cfg.seed,

        # ── Pesos da Função de Loss (Penalização de Pose e Visibilidade) ──────
        box=cfg.box,
        cls=cfg.cls,
        pose=cfg.pose,
        kobj=cfg.kobj,

        # ── Otimizador e Agenda de Learning Rate ──────────────────────────────
        freeze=cfg.freeze,
        optimizer=cfg.optimizer,
        cos_lr=cfg.cos_lr,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,

        # ── Data Augmentation Focada em Oclusão e Vistas Laterais ─────────────
        auto_augment=cfg.auto_augment,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        erasing=cfg.erasing,

        # ── Ajustes Fotométricos ──────────────────────────────────────────────
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,

        # ── Hardware, Logging e Saída ─────────────────────────────────────────
        device=_converter_device(cfg.device),
        workers=cfg.workers,
        cache=cfg.cache,
        amp=cfg.amp,
        project=cfg.project,
        name=cfg.name,
        exist_ok=cfg.exist_ok,
        save_period=cfg.save_period,
        plots=cfg.plots,
        verbose=cfg.verbose,
    )
    return results


def construir_parser() -> argparse.ArgumentParser:
    """Constrói o parser de linha de comando com grupos claros por responsabilidade."""
    defaults = ConfiguracaoTreino()
    parser = argparse.ArgumentParser(
        description="OneVision - Treinamento do zero (from scratch) no YOLO Pose para caprinos/bovídeos (AP-10K)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grp_arch = parser.add_argument_group("Arquitetura e Dataset")
    grp_arch.add_argument(
        "--model-cfg",
        type=str,
        default=defaults.model_cfg,
        help="Arquivo YAML de arquitetura YOLO Pose para iniciar do zero (ex: yolo11n-pose.yaml, yolo11s-pose.yaml, yolo11m-pose.yaml)",
    )
    grp_arch.add_argument(
        "--data",
        type=str,
        default=defaults.data_yaml,
        help="Caminho para o arquivo data.yaml do dataset",
    )
    grp_arch.add_argument(
        "--resume",
        type=str,
        default=defaults.resume,
        help="Caminho para o checkpoint last.pt para retomar um treinamento interrompido (ex: runs/pose/onevision-yolo/caprine-v5-head-clean/weights/last.pt)",
    )

    grp_ciclo = parser.add_argument_group("Ciclo de Treino e Resolução")
    grp_ciclo.add_argument("--epochs", type=int, default=defaults.epochs, help="Total de épocas de treinamento")
    grp_ciclo.add_argument("--patience", type=int, default=defaults.patience, help="Épocas sem melhora para early stopping")
    grp_ciclo.add_argument("--batch", type=int, default=defaults.batch, help="Tamanho do batch")
    grp_ciclo.add_argument("--imgsz", type=int, default=defaults.imgsz, help="Resolução de entrada (múltiplo de 32)")
    grp_ciclo.add_argument("--close-mosaic", type=int, default=defaults.close_mosaic, help="Últimas N épocas sem Mosaic")
    grp_ciclo.add_argument("--seed", type=int, default=defaults.seed, help="Semente aleatória para reprodutibilidade")

    grp_opt = parser.add_argument_group("Otimizador e Pesos de Perda")
    grp_opt.add_argument("--optimizer", type=str, default=defaults.optimizer, help="Otimizador (AdamW, SGD, auto)")
    grp_opt.add_argument("--lr0", type=float, default=defaults.lr0, help="Learning rate inicial")
    grp_opt.add_argument("--lrf", type=float, default=defaults.lrf, help="Fator final de LR (lr0 * lrf)")
    grp_opt.add_argument("--weight-decay", type=float, default=defaults.weight_decay, help="Decaimento de peso (L2)")
    grp_opt.add_argument("--warmup-epochs", type=float, default=defaults.warmup_epochs, help="Épocas de warmup")
    grp_opt.add_argument("--no-cos-lr", action="store_true", help="Desativa scheduler cosseno de LR")
    grp_opt.add_argument("--box", type=float, default=defaults.box, help="Peso da loss de bounding box")
    grp_opt.add_argument("--cls", type=float, default=defaults.cls, help="Peso da loss de classificação")
    grp_opt.add_argument("--pose", type=float, default=defaults.pose, help="Peso da loss de keypoints (pose)")
    grp_opt.add_argument("--kobj", type=float, default=defaults.kobj, help="Peso da loss de visibilidade dos keypoints")

    grp_aug = parser.add_argument_group("Data Augmentation")
    grp_aug.add_argument("--mosaic", type=float, default=defaults.mosaic, help="Probabilidade de Mosaic")
    grp_aug.add_argument("--fliplr", type=float, default=defaults.fliplr, help="Probabilidade de flip horizontal")
    grp_aug.add_argument("--degrees", type=float, default=defaults.degrees, help="Rotação aleatória máxima (graus)")
    grp_aug.add_argument("--translate", type=float, default=defaults.translate, help="Translação aleatória máxima")
    grp_aug.add_argument("--scale", type=float, default=defaults.scale, help="Variação de escala aleatória")
    grp_aug.add_argument("--shear", type=float, default=defaults.shear, help="Cisalhamento aleatório máximo (graus)")
    grp_aug.add_argument("--erasing", type=float, default=defaults.erasing, help="Probabilidade de Random Erasing")
    grp_aug.add_argument("--hsv-h", type=float, default=defaults.hsv_h, help="Variação de matiz (HSV-H)")
    grp_aug.add_argument("--hsv-s", type=float, default=defaults.hsv_s, help="Variação de saturação (HSV-S)")
    grp_aug.add_argument("--hsv-v", type=float, default=defaults.hsv_v, help="Variação de brilho/valor (HSV-V)")

    grp_hw = parser.add_argument_group("Hardware, Cache e Saída")
    grp_hw.add_argument("--device", type=str, default=defaults.device, help="Dispositivo de treino ('0', '0,1' ou 'cpu')")
    grp_hw.add_argument("--workers", type=int, default=defaults.workers, help="Número de workers do DataLoader")
    grp_hw.add_argument("--cache", action="store_true", help="Habilita cache de imagens em RAM/disco")
    grp_hw.add_argument("--no-amp", action="store_true", help="Desativa Automatic Mixed Precision (AMP)")
    grp_hw.add_argument("--project", type=str, default=defaults.project, help="Diretório do projeto em runs/pose/")
    grp_hw.add_argument("--name", type=str, default=defaults.name, help="Nome do experimento/run")
    grp_hw.add_argument("--exist-ok", action="store_true", help="Permite sobrescrever pasta de run existente")
    grp_hw.add_argument("--save-period", type=int, default=defaults.save_period, help="Salvar checkpoint a cada N épocas (-1 desativa)")

    return parser


def parse_args(argv: Optional[list[str]] = None) -> ConfiguracaoTreino:
    """Converte argumentos da CLI para uma instância validada de ConfiguracaoTreino."""
    args = construir_parser().parse_args(argv)
    return ConfiguracaoTreino(
        model_cfg=args.model_cfg,
        data_yaml=args.data,
        resume=args.resume,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        box=args.box,
        cls=args.cls,
        pose=args.pose,
        kobj=args.kobj,
        optimizer=args.optimizer,
        cos_lr=not args.no_cos_lr,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        fliplr=args.fliplr,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        erasing=args.erasing,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        amp=not args.no_amp,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save_period=args.save_period,
    )


if __name__ == "__main__":
    config = parse_args()
    treinar_do_zero(config)
