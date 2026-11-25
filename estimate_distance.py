import os
import re
import csv
import json
import shutil
import argparse
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ultralytics import YOLO
import main as pipeline
from evaluate import (
    MEASURES_CONFIG,
    VALID_IMAGE_EXTENSIONS,
    carregar_ground_truth,
    gerar_caminho_unico,
    resolver_medidas_selecionadas,
)


DEFAULT_IGNORE_MEASURES = ("grossura_pescoco",)

STATUS_LABELS = {
    "ok": "OK (Consistente)",
    "ajustada": "Ajustada (Nova Distância)",
    "sem_melhora": "Sem Melhora",
    "sem_dado": "Sem Dado / Sem Gabarito",
}

STATUS_COLORS = {
    "ok": "#2ca02c",
    "ajustada": "#1f77b4",
    "sem_melhora": "#ff7f0e",
    "sem_dado": "#7f7f7f",
}

CRITERIOS_DESCRICAO = {
    "min_sim": "Maior similaridade mínima entre medidas avaliadas (desempate: maior similaridade média)",
    "mean_sim": "Maior similaridade média entre medidas avaliadas (menor erro relativo médio)",
    "converged_then_mean": "Maior nº de medidas >= limiar (desempate: maior similaridade mínima e média)",
}


def extrair_distancia_anotada(nome_arquivo: str) -> Optional[float]:
    """
    Extrai a distância anotada em centímetros a partir do nome do arquivo via regex (ex: 'dist160' -> 160.0).
    Retorna None caso o padrão não exista no nome do arquivo.
    """
    match = re.search(r"dist(\d+(?:\.\d+)?)", os.path.basename(nome_arquivo), flags=re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return val if val > 0 else None
        except ValueError:
            return None
    return None


def extrair_id_animal(nome_arquivo: str) -> Optional[int]:
    """Extrai o prefixo numérico identificador do animal (ex: '007-dist160-v1.jpg' -> 7)."""
    match = re.match(r"^(\d+)", os.path.basename(nome_arquivo))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def resolver_medidas_ignoradas(ignore_arg: Optional[str]) -> list[str]:
    """
    Converte a string de medidas ignoradas (ex: 'pescoco,peso' ou 'none')
    em uma lista de chaves canônicas de MEASURES_CONFIG.
    """
    if ignore_arg is None:
        return list(DEFAULT_IGNORE_MEASURES)

    limpo = ignore_arg.strip().lower()
    if not limpo or limpo in ("none", "nenhuma", "false", "0"):
        return []

    return resolver_medidas_selecionadas(limpo)


def calcular_similaridade(pred: float, real: float) -> float:
    """
    Calcula a similaridade percentual no intervalo [0.0, 1.0] em relação ao valor real (ground truth):
        sim = max(0.0, 1.0 - abs(pred - real) / real)
    Onde 0.90 equivale a até 10% de erro relativo.
    """
    if real <= 0:
        return 0.0
    erro_rel = abs(pred - real) / real
    return float(max(0.0, 1.0 - erro_rel))


def criar_diretorio_execucao(base_out_dir: str, tag_modelo: str) -> tuple[str, str]:
    """
    Cria um subdiretório exclusivo dentro de base_out_dir (ex: outputs/estimativa-distancias/run_YYYYMMDD_HHMMSS_<tag>/)
    garantindo isolamento total entre execuções sem sobrescrever artefatos anteriores.
    """
    os.makedirs(base_out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_base = f"run_{timestamp}_{tag_modelo}"
    caminho_run = os.path.join(base_out_dir, run_id_base)

    contador = 1
    run_id = run_id_base
    while os.path.exists(caminho_run):
        run_id = f"{run_id_base}_{contador}"
        caminho_run = os.path.join(base_out_dir, run_id)
        contador += 1

    os.makedirs(caminho_run, exist_ok=True)
    return caminho_run, run_id


def extrair_keypoints_principais(
    image_path: str,
    modelo_yolo: YOLO,
    conf_thresh: float = 0.25,
    imgsz: int = 1280,
) -> Optional[tuple[np.ndarray, int, int]]:
    """
    Executa a detecção de pose seguindo exatamente a regra de filtragem de main.executar_inferencia
    e retorna (kpts_principal, largura_img, altura_img) ou None caso não haja detecção válida.
    """
    if not os.path.exists(image_path):
        return None

    nms_conf = min(float(conf_thresh), 0.25)
    results = modelo_yolo(image_path, conf=nms_conf, imgsz=imgsz, verbose=False)
    if not results or len(results[0]) == 0 or results[0].keypoints is None:
        return None

    result = results[0]
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    kpts_data = result.keypoints.data.cpu().numpy()
    boxes_conf = (
        result.boxes.conf.cpu().numpy()
        if (result.boxes is not None and result.boxes.conf is not None)
        else np.ones(len(kpts_data))
    )

    detecoes_validas = []
    for idx_det, kpts in enumerate(kpts_data):
        confs = [float(p[2]) for p in kpts]
        pts_bons = sum(1 for c in confs if c >= 0.35)
        if pts_bons >= 6:
            box_c = float(boxes_conf[idx_det]) if idx_det < len(boxes_conf) else 0.0
            top6_kpt_c = float(np.mean(sorted(confs, reverse=True)[:6]))
            if max(box_c, top6_kpt_c) >= conf_thresh:
                detecoes_validas.append((pts_bons, float(np.mean(confs)), kpts))

    if not detecoes_validas:
        return None

    detecoes_validas.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return detecoes_validas[0][2], w, h


def avaliar_estimativas_na_distancia(
    estimativas: dict[str, float],
    gt_item: dict[str, Any],
    medidas_todas: list[str],
    medidas_ignoradas: set[str],
    sim_threshold: float,
) -> dict[str, Any]:
    """
    Compara as medidas estimadas em uma dada distância com o ground truth (dados_biometria.json).
    Retorna métricas detalhadas por medida (incluindo as ignoradas marcadas com 'ignorada': True)
    e estatísticas agregadas apenas sobre as medidas avaliadas (não ignoradas).
    """
    detalhes_medidas: dict[str, dict[str, Any]] = {}
    similaridades_avaliadas: list[float] = []
    erros_rel_avaliados: list[float] = []
    n_abaixo_limiar = 0

    for m_key in medidas_todas:
        cfg = MEASURES_CONFIG[m_key]
        ignorada = m_key in medidas_ignoradas
        val_pred = estimativas.get(cfg["pred_key"])
        val_real = pipeline.extrair_valor_gabarito(gt_item, *cfg["gt_keys"])

        if val_pred is not None and val_real is not None and val_real > 0:
            erro_abs = abs(float(val_pred) - float(val_real))
            erro_rel = erro_abs / float(val_real)
            erro_pct = erro_rel * 100.0
            sim = calcular_similaridade(float(val_pred), float(val_real))
            atingiu_limiar = sim >= sim_threshold

            detalhes_medidas[m_key] = {
                "measure_key": m_key,
                "label": cfg["label"],
                "unit": cfg["unit"],
                "ignorada": ignorada,
                "real": float(val_real),
                "pred": float(val_pred),
                "erro_abs": float(erro_abs),
                "erro_pct": float(erro_pct),
                "similaridade": float(sim),
                "similaridade_pct": float(sim * 100.0),
                "atingiu_limiar": bool(atingiu_limiar),
            }

            if not ignorada:
                similaridades_avaliadas.append(sim)
                erros_rel_avaliados.append(erro_rel)
                if not atingiu_limiar:
                    n_abaixo_limiar += 1
        else:
            detalhes_medidas[m_key] = {
                "measure_key": m_key,
                "label": cfg["label"],
                "unit": cfg["unit"],
                "ignorada": ignorada,
                "real": float(val_real) if val_real is not None else None,
                "pred": float(val_pred) if val_pred is not None else None,
                "erro_abs": None,
                "erro_pct": None,
                "similaridade": None,
                "similaridade_pct": None,
                "atingiu_limiar": None,
            }

    if similaridades_avaliadas:
        min_sim = float(np.min(similaridades_avaliadas))
        mean_sim = float(np.mean(similaridades_avaliadas))
        mean_err_pct = float(np.mean(erros_rel_avaliados) * 100.0)
        max_err_pct = float(np.max(erros_rel_avaliados) * 100.0)
        todas_ok = (n_abaixo_limiar == 0)
        n_convergidas = len(similaridades_avaliadas) - n_abaixo_limiar
    else:
        min_sim = 0.0
        mean_sim = 0.0
        mean_err_pct = 100.0
        max_err_pct = 100.0
        todas_ok = False
        n_convergidas = 0

    return {
        "medidas": detalhes_medidas,
        "n_avaliadas": len(similaridades_avaliadas),
        "n_convergidas": n_convergidas,
        "n_abaixo_limiar": n_abaixo_limiar,
        "todas_acima_limiar": todas_ok,
        "min_sim": min_sim,
        "mean_sim": mean_sim,
        "mean_err_pct": mean_err_pct,
        "max_err_pct": max_err_pct,
    }


def gerar_distancias_candidatas(
    dist_anotada: float,
    search_range: float = 60.0,
    search_step: float = 5.0,
    min_dist: float = 30.0,
    max_dist: float = 500.0,
) -> list[float]:
    """
    Gera a lista ordenada de distâncias candidatas próximas à distância anotada,
    respeitando limites mínimo positivo e máximo configurados.
    """
    if search_step <= 0:
        raise ValueError("O passo de busca (--search-step) deve ser maior que zero.")

    inicio = max(min_dist, dist_anotada - abs(search_range))
    fim = min(max_dist, dist_anotada + abs(search_range))

    n_passos = int(round((fim - inicio) / search_step))
    candidatas = [round(inicio + i * search_step, 4) for i in range(n_passos + 1)]
    if round(dist_anotada, 4) not in candidatas and min_dist <= dist_anotada <= max_dist:
        candidatas.append(round(dist_anotada, 4))
    return sorted(set(candidatas))


def _chave_ordenacao_candidata(
    av: dict[str, Any],
    dist_cand: float,
    dist_anotada: float,
    criterion: str,
) -> tuple:
    """
    Define a tupla de pontuação para maximização segundo o critério configurado:
      - 'min_sim': Maximiza a menor similaridade entre as medidas avaliadas (critério minimax),
                   desempatando pelo número de medidas convergidas, maior similaridade média
                   e menor desvio em relação à distância anotada.
      - 'mean_sim': Maximiza a similaridade média (menor MAPE), desempatando pela similaridade mínima
                    e menor desvio em relação à distância anotada.
      - 'converged_then_mean': Maximiza o total de medidas >= limiar, desempatando pela similaridade
                               mínima e similaridade média.
    """
    prox_anotada = -abs(dist_cand - dist_anotada)
    if criterion == "mean_sim":
        return (
            round(av["mean_sim"], 6),
            round(av["min_sim"], 6),
            av["n_convergidas"],
            prox_anotada,
        )
    if criterion == "converged_then_mean":
        return (
            av["n_convergidas"],
            round(av["min_sim"], 6),
            round(av["mean_sim"], 6),
            prox_anotada,
        )
    # Padrão: 'min_sim'
    return (
        round(av["min_sim"], 6),
        av["n_convergidas"],
        round(av["mean_sim"], 6),
        prox_anotada,
    )


def buscar_melhor_distancia(
    kpts: np.ndarray,
    img_w: int,
    gt_item: dict[str, Any],
    dist_anotada: float,
    av_anotada: dict[str, Any],
    medidas_todas: list[str],
    medidas_ignoradas: set[str],
    sim_threshold: float,
    search_range: float,
    search_step: float,
    min_dist: float,
    max_dist: float,
    criterion: str = "min_sim",
    min_improvement: float = 0.005,
) -> dict[str, Any]:
    """
    Busca entre as distâncias candidatas aquela que maximiza o ajuste biométrico.
    Retorna a distância selecionada, a avaliação correspondente, a curva de busca
    e a classificação ('ajustada' ou 'sem_melhora').
    """
    candidatas = gerar_distancias_candidatas(
        dist_anotada=dist_anotada,
        search_range=search_range,
        search_step=search_step,
        min_dist=min_dist,
        max_dist=max_dist,
    )

    melhor_dist = dist_anotada
    melhor_av = av_anotada
    melhor_estimativas = None
    melhor_score = _chave_ordenacao_candidata(av_anotada, dist_anotada, dist_anotada, criterion)
    historico_busca = []

    for d_cand in candidatas:
        if abs(d_cand - dist_anotada) < 1e-6:
            av_cand = av_anotada
            est_cand = None
        else:
            est_cand = pipeline.estimar_biometria_autonoma(kpts, dist_cm=d_cand, img_w=img_w)
            av_cand = avaliar_estimativas_na_distancia(
                estimativas=est_cand,
                gt_item=gt_item,
                medidas_todas=medidas_todas,
                medidas_ignoradas=medidas_ignoradas,
                sim_threshold=sim_threshold,
            )

        historico_busca.append({
            "dist_cm": float(d_cand),
            "min_sim": float(av_cand["min_sim"]),
            "mean_sim": float(av_cand["mean_sim"]),
            "n_convergidas": int(av_cand["n_convergidas"]),
            "todas_acima_limiar": bool(av_cand["todas_acima_limiar"]),
        })

        score_cand = _chave_ordenacao_candidata(av_cand, d_cand, dist_anotada, criterion)
        if score_cand > melhor_score:
            melhor_score = score_cand
            melhor_dist = float(d_cand)
            melhor_av = av_cand
            melhor_estimativas = est_cand

    mudou_distancia = abs(melhor_dist - dist_anotada) >= 1e-4
    ganho_min_sim = melhor_av["min_sim"] - av_anotada["min_sim"]
    ganho_mean_sim = melhor_av["mean_sim"] - av_anotada["mean_sim"]
    ganho_convergidas = melhor_av["n_convergidas"] - av_anotada["n_convergidas"]

    # Classifica como 'ajustada' se encontrou uma distância diferente com convergência completa
    # ou ganho significativo na métrica principal sem degradar a similaridade mínima
    houve_melhora_valida = mudou_distancia and (
        melhor_av["todas_acima_limiar"]
        or (ganho_min_sim >= min_improvement)
        or (ganho_convergidas > 0 and ganho_min_sim >= 0.0)
        or (criterion == "mean_sim" and ganho_mean_sim >= min_improvement and ganho_min_sim >= -0.01)
    )

    status = "ajustada" if houve_melhora_valida else "sem_melhora"
    dist_final = melhor_dist if houve_melhora_valida else dist_anotada
    av_final = melhor_av if houve_melhora_valida else av_anotada

    return {
        "status": status,
        "dist_estimada": float(dist_final),
        "melhor_candidata_bruta": float(melhor_dist),
        "avaliacao_estimada": av_final,
        "avaliacao_melhor_candidata": melhor_av,
        "estimativas_melhor_candidata": melhor_estimativas,
        "ganho_min_sim": float(ganho_min_sim),
        "ganho_mean_sim": float(ganho_mean_sim),
        "ganho_convergidas": int(ganho_convergidas),
        "historico_busca": historico_busca,
    }


def copiar_imagem_distancia_estimada(
    image_path: str,
    dest_dir: str,
    dist_anotada: float,
    dist_estimada: float,
) -> dict[str, Any]:
    """
    Salva uma cópia da imagem original em inputs/distancia_estimada/ (ou diretório configurado)
    quando é identificada uma distância alternativa melhor (status 'ajustada'),
    renomeando o arquivo copiado para conter a distância estimada (ex: '007-dist160-v1.jpg' -> '007-dist180-v1.jpg').
    """
    os.makedirs(dest_dir, exist_ok=True)
    nome_original = os.path.basename(image_path)

    dist_int = int(round(dist_estimada))
    if re.search(r"dist\d+(?:\.\d+)?", nome_original, flags=re.IGNORECASE):
        nome_copia = re.sub(
            r"dist\d+(?:\.\d+)?",
            f"dist{dist_int}",
            nome_original,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        base, ext = os.path.splitext(nome_original)
        nome_copia = f"{base}-dist{dist_int}{ext}"

    caminho_copia = os.path.join(dest_dir, nome_copia)
    shutil.copy2(image_path, caminho_copia)

    return {
        "caminho_origem": image_path,
        "caminho_copia": caminho_copia,
        "nome_original": nome_original,
        "nome_copia": nome_copia,
        "nome_sugerido": nome_copia,
        "dist_anotada_cm": float(dist_anotada),
        "dist_estimada_cm": float(dist_estimada),
    }


def formatar_resumo_medidas_linha(
    medidas_dict: dict[str, dict[str, Any]],
    medidas_ordem: list[str],
) -> str:
    """Formata uma linha compacta com similaridade/erro de cada medida (destacando as ignoradas)."""
    abrev = {
        "peso": "Peso",
        "altura_cernelha": "Cern",
        "altura_garupa": "Gar",
        "comprimento": "Comp",
        "circ_torax": "Torax",
        "grossura_pescoco": "Pesc",
    }
    partes = []
    for m_key in medidas_ordem:
        info = medidas_dict.get(m_key)
        nome_curto = abrev.get(m_key, m_key[:5])
        if not info or info["similaridade"] is None:
            suff = "[IGN]" if (info and info.get("ignorada")) else ""
            partes.append(f"{nome_curto}:N/A{suff}")
        else:
            sim_pct = info["similaridade_pct"]
            err_pct = info["erro_pct"]
            if info["ignorada"]:
                partes.append(f"{nome_curto}:{sim_pct:4.0f}%(E:{err_pct:3.0f}%)[IGN]")
            else:
                flag = "ok" if info["atingiu_limiar"] else "!!"
                partes.append(f"{nome_curto}:{sim_pct:4.0f}%(E:{err_pct:3.0f}%){flag}")
    return " | ".join(partes)


def gerar_relatorio_png(
    registros: list[dict[str, Any]],
    resumo_status: dict[str, int],
    parametros: dict[str, Any],
    medidas_ordem: list[str],
    medidas_ignoradas: list[str],
    nome_modelo: str,
    tag_modelo: str,
    run_dir: str,
    run_id: str,
) -> str:
    """
    Gera o relatório visual consolidado em PNG dentro da pasta da execução,
    seguindo o estilo gráfico do evaluate.py (matplotlib/seaborn, grade, paleta e tipografia).
    Inclui:
      1. Cabeçalho completo com parâmetros da execução e resumo por status
      2. Gráfico de barras de contagem por status (ok, ajustada, sem_melhora, sem_dado)
      3. Gráfico comparativo de Distância Anotada vs Distância Estimada e ganho de similaridade
      4. Tabela detalhada por imagem com distâncias, métricas por medida (indicando ignoradas),
         status e indicação de cópia para inputs/distancia_estimada/
    """
    os.makedirs(run_dir, exist_ok=True)

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="deep")
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    registros_com_dist = [r for r in registros if r["dist_anotada"] is not None and r["status"] != "sem_dado"]
    n_linhas_tabela = max(1, len(registros))

    altura_tabela_in = max(4.8, 1.35 + n_linhas_tabela * 0.30)
    altura_total_in = 8.4 + altura_tabela_in

    fig = plt.figure(figsize=(21, altura_total_in), dpi=150)
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.2, 3.6, altura_tabela_in],
        width_ratios=[1.0, 1.45, 1.35],
        hspace=0.22,
        wspace=0.24,
    )

    sim_limiar_pct = parametros["sim_threshold"] * 100.0
    erro_max_pct = (1.0 - parametros["sim_threshold"]) * 100.0
    ignoradas_labels = [MEASURES_CONFIG[k]["label"] for k in medidas_ignoradas if k in MEASURES_CONFIG]
    ignoradas_str = ", ".join(ignoradas_labels) if ignoradas_labels else "Nenhuma"
    avaliadas_labels = [
        MEASURES_CONFIG[k]["label"] for k in medidas_ordem if k not in medidas_ignoradas and k in MEASURES_CONFIG
    ]

    fig.suptitle(
        f"ONEVISION — Estimativa de Distância da Câmera e Auditoria de Anotações\n"
        f"Modelo: {nome_modelo} [{tag_modelo}] | Execução: {run_id}",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )

    # 1. Painel de Cabeçalho de Parâmetros e Resumo Geral
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    texto_parametros = (
        f"PARÂMETROS DA EXECUÇÃO\n"
        f"• Modelo / Pesos: {parametros['model_path']}  (conf={parametros['conf']}, imgsz={parametros['imgsz']})\n"
        f"• Limiar de Proximidade (Similaridade): {parametros['sim_threshold']:.2f} ({sim_limiar_pct:.1f}% de acerto mín. / máx. {erro_max_pct:.1f}% erro relativo)\n"
        f"• Medidas Avaliadas ({len(avaliadas_labels)}): {', '.join(avaliadas_labels)}   |   Medidas Ignoradas ({len(ignoradas_labels)}): {ignoradas_str}\n"
        f"• Busca de Distância Alternativa: ±{parametros['search_range']:.0f} cm em torno da distância anotada, passo de {parametros['search_step']:.0f} cm "
        f"(limites: [{parametros['min_dist']:.0f}, {parametros['max_dist']:.0f}] cm)\n"
        f"• Critério de Seleção: {parametros['criterion']} — {CRITERIOS_DESCRICAO.get(parametros['criterion'], parametros['criterion'])}\n"
        f"• Cópia de Imagens Ajustadas: '{parametros['copy_dir']}/' ({resumo_status.get('copiadas', 0)} imagem(ns) copiada(s))"
    )

    ax_header.text(
        0.01,
        0.50,
        texto_parametros,
        transform=ax_header.transAxes,
        fontsize=9.3,
        va="center",
        ha="left",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f4f7fa", "edgecolor": "#2b8cbe", "linewidth": 1.3},
    )

    # 2A. Gráfico de Contagem por Status
    ax_status = fig.add_subplot(gs[1, 0])
    chaves_status = ["ok", "ajustada", "sem_melhora", "sem_dado"]
    nomes_status = ["OK\n(Consistente)", "Ajustada\n(Nova Dist.)", "Sem\nMelhora", "Sem Dado /\nSem GT"]
    contagens = [resumo_status.get(k, 0) for k in chaves_status]
    cores_status = [STATUS_COLORS[k] for k in chaves_status]

    x_st = np.arange(len(chaves_status))
    barras_st = ax_status.bar(x_st, contagens, color=cores_status, edgecolor="black", width=0.56, alpha=0.90)
    for bar, qtd in zip(barras_st, contagens):
        ax_status.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(0.25, max(contagens + [1]) * 0.03),
            f"{qtd}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_status.set_xticks(x_st)
    ax_status.set_xticklabels(nomes_status, fontsize=9)
    ax_status.set_ylim(0, max(contagens + [1]) * 1.24 + 1)
    ax_status.set_ylabel("Quantidade de Imagens", fontsize=9.5)
    ax_status.set_title(
        f"Resumo Geral por Status (Total: {len(registros)} imagens)",
        fontsize=11,
        fontweight="semibold",
    )
    ax_status.grid(True, axis="y", linestyle=":", alpha=0.6)

    # 2B. Gráfico de Barras Emparelhadas: Distância Anotada vs Distância Estimada
    ax_dist = fig.add_subplot(gs[1, 1])
    if registros_com_dist:
        labels_img = [os.path.splitext(r["imagem"])[0] for r in registros_com_dist]
        d_anot = np.array([r["dist_anotada"] for r in registros_com_dist], dtype=float)
        d_est = np.array(
            [
                r["dist_estimada"] if r["dist_estimada"] is not None else r["dist_anotada"]
                for r in registros_com_dist
            ],
            dtype=float,
        )
        x_d = np.arange(len(registros_com_dist))
        larg = 0.40

        ax_dist.bar(
            x_d - larg / 2,
            d_anot,
            width=larg,
            color="#7f7f7f",
            edgecolor="black",
            alpha=0.82,
            label="Distância Anotada (cm)",
        )
        cores_est = [STATUS_COLORS.get(r["status"], "#1f77b4") for r in registros_com_dist]
        ax_dist.bar(
            x_d + larg / 2,
            d_est,
            width=larg,
            color=cores_est,
            edgecolor="black",
            alpha=0.90,
            label="Distância Estimada / Final (cm)",
        )

        for idx_r, r in enumerate(registros_com_dist):
            if r["status"] == "ajustada" and r["dist_estimada"] is not None:
                delta = r["dist_estimada"] - r["dist_anotada"]
                sinal = "+" if delta > 0 else ""
                ax_dist.text(
                    x_d[idx_r] + larg / 2,
                    d_est[idx_r] + 3.0,
                    f"{int(round(d_est[idx_r]))}\n({sinal}{int(round(delta))})",
                    ha="center",
                    va="bottom",
                    fontsize=6.6,
                    fontweight="bold",
                    color="#0b4f8a",
                )

        ax_dist.set_xticks(x_d)
        ax_dist.set_xticklabels(labels_img, rotation=70, ha="right", fontsize=7.2)
        ax_dist.set_ylim(0, max(float(np.max(d_anot)), float(np.max(d_est))) * 1.22)
        ax_dist.set_ylabel("Distância da Câmera (cm)", fontsize=9.5)
        ax_dist.set_title(
            "Distância Anotada vs Distância Estimada por Imagem",
            fontsize=11,
            fontweight="semibold",
        )
        ax_dist.legend(loc="upper left", fontsize=8, frameon=True)
        ax_dist.grid(True, axis="y", linestyle=":", alpha=0.6)
    else:
        ax_dist.text(0.5, 0.5, "Sem imagens avaliadas com distância válida", ha="center", va="center")
        ax_dist.set_axis_off()

    # 2C. Gráfico de Similaridade Mínima (Antes vs Depois do Ajuste de Distância)
    ax_sim = fig.add_subplot(gs[1, 2])
    if registros_com_dist:
        labels_img = [os.path.splitext(r["imagem"])[0] for r in registros_com_dist]
        sim_antes = np.array(
            [r["avaliacao_anotada"]["min_sim"] * 100.0 for r in registros_com_dist],
            dtype=float,
        )
        sim_depois = np.array(
            [r["avaliacao_final"]["min_sim"] * 100.0 for r in registros_com_dist],
            dtype=float,
        )
        x_s = np.arange(len(registros_com_dist))
        larg = 0.40

        ax_sim.bar(
            x_s - larg / 2,
            sim_antes,
            width=larg,
            color="#d95f02",
            edgecolor="black",
            alpha=0.80,
            label="Sim. Mínima na Dist. Anotada (%)",
        )
        ax_sim.bar(
            x_s + larg / 2,
            sim_depois,
            width=larg,
            color="#1b9e77",
            edgecolor="black",
            alpha=0.88,
            label="Sim. Mínima na Dist. Estimada (%)",
        )
        ax_sim.axhline(
            sim_limiar_pct,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=f"Limiar Configurado ({sim_limiar_pct:.1f}%)",
        )

        ax_sim.set_xticks(x_s)
        ax_sim.set_xticklabels(labels_img, rotation=70, ha="right", fontsize=7.2)
        ax_sim.set_ylim(0, 112)
        ax_sim.set_ylabel("Similaridade Mínima entre Medidas (%)", fontsize=9.5)
        ax_sim.set_title(
            "Convergência Biométrica: Similaridade Mínima Antes vs Depois",
            fontsize=11,
            fontweight="semibold",
        )
        ax_sim.legend(loc="lower right", fontsize=8, frameon=True)
        ax_sim.grid(True, axis="y", linestyle=":", alpha=0.6)
    else:
        ax_sim.text(0.5, 0.5, "Sem dados de similaridade para exibir", ha="center", va="center")
        ax_sim.set_axis_off()

    # 3. Tabela Detalhada por Imagem
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    abrev_medida = {
        "peso": "Peso (kg)",
        "altura_cernelha": "Cernelha (cm)",
        "altura_garupa": "Garupa (cm)",
        "comprimento": "Compr. (cm)",
        "circ_torax": "Tórax (cm)",
        "grossura_pescoco": "Pescoço (cm)",
    }

    col_headers = [
        "Imagem",
        "ID",
        "Dist. Anot.",
        "Dist. Est.",
    ]
    for m_key in medidas_ordem:
        titulo_col = abrev_medida.get(m_key, m_key)
        if m_key in medidas_ignoradas:
            titulo_col += " [IGN]"
        col_headers.append(titulo_col)
    col_headers.extend(["Sim.Mín\n(Anot->Est)", "Sim.Méd\n(Anot->Est)", "Status", "Cópia Salva"])

    linhas_tabela = []
    cores_linhas = []

    paleta_fundo_status = {
        "ok": "#eaf7ea",
        "ajustada": "#e6f2ff",
        "sem_melhora": "#fff3e0",
        "sem_dado": "#f2f2f2",
    }

    for r in registros:
        st = r["status"]
        cor_fundo = paleta_fundo_status.get(st, "#ffffff")

        d_anot_str = f"{r['dist_anotada']:.0f} cm" if r["dist_anotada"] is not None else "N/A"
        if r["dist_estimada"] is not None:
            if r["dist_anotada"] is not None and abs(r["dist_estimada"] - r["dist_anotada"]) >= 1e-4:
                delta = r["dist_estimada"] - r["dist_anotada"]
                sinal = "+" if delta > 0 else ""
                d_est_str = f"{r['dist_estimada']:.0f} cm ({sinal}{delta:.0f})"
            else:
                d_est_str = f"{r['dist_estimada']:.0f} cm"
        else:
            d_est_str = "N/A"

        linha = [
            r["imagem"],
            str(r["id_animal"]) if r["id_animal"] is not None else "-",
            d_anot_str,
            d_est_str,
        ]

        av_ref = r.get("avaliacao_final") or r.get("avaliacao_anotada")
        medidas_info = av_ref["medidas"] if av_ref else {}

        for m_key in medidas_ordem:
            m_info = medidas_info.get(m_key)
            if not m_info or m_info["similaridade"] is None:
                tag_ign = " [IGN]" if m_key in medidas_ignoradas else ""
                linha.append(f"N/A{tag_ign}")
            else:
                sim_p = m_info["similaridade_pct"]
                err_p = m_info["erro_pct"]
                if m_info["ignorada"]:
                    linha.append(f"Sim {sim_p:4.1f}% (E:{err_p:3.1f}%) [IGN]")
                else:
                    flag = "OK" if m_info["atingiu_limiar"] else "BAIXA"
                    linha.append(f"Sim {sim_p:4.1f}% (E:{err_p:3.1f}%) [{flag}]")

        if r.get("avaliacao_anotada") and r.get("avaliacao_final"):
            s_min_a = r["avaliacao_anotada"]["min_sim"] * 100.0
            s_min_f = r["avaliacao_final"]["min_sim"] * 100.0
            s_med_a = r["avaliacao_anotada"]["mean_sim"] * 100.0
            s_med_f = r["avaliacao_final"]["mean_sim"] * 100.0
            sim_min_str = f"{s_min_a:4.1f}% -> {s_min_f:4.1f}%"
            sim_med_str = f"{s_med_a:4.1f}% -> {s_med_f:4.1f}%"
        else:
            sim_min_str = "-"
            sim_med_str = "-"

        copia_str = "-"
        if r.get("copia_info"):
            copia_str = r["copia_info"].get("nome_copia", f"SIM ({int(round(r['dist_estimada']))}cm)")

        status_cell = r["status"].upper()
        if r.get("motivo_status"):
            motivo_curto = r["motivo_status"]
            if len(motivo_curto) > 22:
                motivo_curto = motivo_curto[:20] + ".."
            status_cell = f"{status_cell}\n({motivo_curto})"

        linha.extend([sim_min_str, sim_med_str, status_cell, copia_str])
        linhas_tabela.append(linha)
        cores_linhas.append([cor_fundo] * len(col_headers))

    n_cols = len(col_headers)
    larguras_col = [0.10, 0.026, 0.050, 0.070] + [0.084] * len(medidas_ordem) + [0.070, 0.070, 0.095, 0.095]
    soma_larg = sum(larguras_col)
    larguras_col = [w / soma_larg for w in larguras_col]

    tabela = ax_table.table(
        cellText=linhas_tabela,
        colLabels=col_headers,
        cellColours=cores_linhas,
        colColours=["#2b5c8f"] * n_cols,
        colWidths=larguras_col,
        cellLoc="center",
        loc="upper center",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(6.9)
    tabela.scale(1.0, 1.28)

    for (row_idx, col_idx), cell in tabela.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=7.3)
            cell.set_facecolor("#1f4e78")
        elif col_idx == 0:
            cell.set_text_props(ha="left", weight="semibold")

    legenda_patches = [
        Patch(facecolor="#eaf7ea", edgecolor="#2ca02c", label="ok: Todas as medidas avaliadas >= limiar na distância anotada"),
        Patch(facecolor="#e6f2ff", edgecolor="#1f77b4", label="ajustada: Distância candidata melhorou o ajuste biométrico (cópia salva)"),
        Patch(facecolor="#fff3e0", edgecolor="#ff7f0e", label="sem_melhora: Medidas abaixo do limiar sem distância candidata superior"),
        Patch(facecolor="#f2f2f2", edgecolor="#7f7f7f", label="sem_dado: Sem padrão dist(\\d+), sem gabarito no JSON ou sem detecção"),
    ]
    ax_table.legend(
        handles=legenda_patches,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.035),
        ncol=2,
        fontsize=8.0,
        frameon=True,
    )

    caminho_png = gerar_caminho_unico(run_dir, f"relatorio_estimativa_distancias_{tag_modelo}", ".png")
    fig.savefig(caminho_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho_png


def salvar_artefatos_estruturados(
    run_dir: str,
    run_id: str,
    tag_modelo: str,
    nome_modelo: str,
    parametros: dict[str, Any],
    resumo_status: dict[str, int],
    registros: list[dict[str, Any]],
    medidas_ordem: list[str],
    medidas_ignoradas: list[str],
    caminho_png: str,
) -> tuple[str, str]:
    """
    Salva os resultados estruturados da execução em JSON e CSV dentro da pasta exclusiva da execução.
    """
    caminho_json = gerar_caminho_unico(run_dir, f"relatorio_estimativa_distancias_{tag_modelo}", ".json")
    caminho_csv = gerar_caminho_unico(run_dir, f"relatorio_estimativa_distancias_{tag_modelo}", ".csv")

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modelo_caminho": parametros["model_path"],
        "modelo_nome": nome_modelo,
        "modelo_tag": tag_modelo,
        "parametros": parametros,
        "criterio_descricao": CRITERIOS_DESCRICAO.get(parametros["criterion"], parametros["criterion"]),
        "medidas_avaliadas": [m for m in medidas_ordem if m not in medidas_ignoradas],
        "medidas_ignoradas": medidas_ignoradas,
        "resumo_status": resumo_status,
        "relatorio_png": caminho_png,
        "resultados_por_imagem": registros,
    }

    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    campos_csv = [
        "imagem",
        "id_animal",
        "identificacao",
        "status",
        "motivo_status",
        "dist_anotada_cm",
        "dist_estimada_cm",
        "delta_dist_cm",
        "sim_min_anotada",
        "sim_mean_anotada",
        "sim_min_final",
        "sim_mean_final",
        "copiada_para_distancia_estimada",
        "caminho_copia",
        "nome_sugerido",
    ]
    for m_key in medidas_ordem:
        campos_csv.extend([
            f"{m_key}_ignorada",
            f"{m_key}_real",
            f"{m_key}_pred_anotada",
            f"{m_key}_sim_anotada",
            f"{m_key}_pred_final",
            f"{m_key}_sim_final",
        ])

    with open(caminho_csv, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campos_csv)
        writer.writeheader()

        for r in registros:
            av_a = r.get("avaliacao_anotada")
            av_f = r.get("avaliacao_final")
            copia = r.get("copia_info")

            delta_d = (
                (r["dist_estimada"] - r["dist_anotada"])
                if (r["dist_anotada"] is not None and r["dist_estimada"] is not None)
                else None
            )

            row: dict[str, Any] = {
                "imagem": r["imagem"],
                "id_animal": r["id_animal"],
                "identificacao": r.get("identificacao", ""),
                "status": r["status"],
                "motivo_status": r.get("motivo_status", ""),
                "dist_anotada_cm": r["dist_anotada"],
                "dist_estimada_cm": r["dist_estimada"],
                "delta_dist_cm": round(delta_d, 2) if delta_d is not None else None,
                "sim_min_anotada": round(av_a["min_sim"], 4) if av_a else None,
                "sim_mean_anotada": round(av_a["mean_sim"], 4) if av_a else None,
                "sim_min_final": round(av_f["min_sim"], 4) if av_f else None,
                "sim_mean_final": round(av_f["mean_sim"], 4) if av_f else None,
                "copiada_para_distancia_estimada": bool(copia),
                "caminho_copia": copia["caminho_copia"] if copia else "",
                "nome_sugerido": copia["nome_sugerido"] if copia else "",
            }

            for m_key in medidas_ordem:
                m_a = av_a["medidas"].get(m_key) if av_a else None
                m_f = av_f["medidas"].get(m_key) if av_f else None
                row[f"{m_key}_ignorada"] = m_key in medidas_ignoradas
                row[f"{m_key}_real"] = round(m_a["real"], 2) if (m_a and m_a["real"] is not None) else None
                row[f"{m_key}_pred_anotada"] = round(m_a["pred"], 2) if (m_a and m_a["pred"] is not None) else None
                row[f"{m_key}_sim_anotada"] = (
                    round(m_a["similaridade"], 4) if (m_a and m_a["similaridade"] is not None) else None
                )
                row[f"{m_key}_pred_final"] = round(m_f["pred"], 2) if (m_f and m_f["pred"] is not None) else None
                row[f"{m_key}_sim_final"] = (
                    round(m_f["similaridade"], 4) if (m_f and m_f["similaridade"] is not None) else None
                )

            writer.writerow(row)

    return caminho_json, caminho_csv


def executar_estimativa_distancias(
    model_path: str = "./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt",
    conf_thresh: float = 0.25,
    imgsz: int = 1280,
    inputs_dir: str = "inputs",
    gt_file: str = "dados_biometria.json",
    out_dir: str = "outputs/estimativa-distancias",
    copy_dir: str = "inputs/distancia_estimada",
    sim_threshold: float = 0.90,
    ignore_measures: Optional[str] = "grossura_pescoco",
    search_range: float = 60.0,
    search_step: float = 5.0,
    min_dist: float = 30.0,
    max_dist: float = 500.0,
    criterion: str = "min_sim",
) -> dict[str, Any]:
    """
    Fluxo principal de auditoria e estimativa da distância da câmera para imagens em inputs/.
    """
    if not (0.0 < sim_threshold <= 1.0):
        raise ValueError("O limiar de similaridade (--sim-threshold) deve estar no intervalo (0.0, 1.0].")

    if not os.path.exists(inputs_dir):
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {inputs_dir}")

    banco_gt = carregar_ground_truth(gt_file)
    medidas_ordem = list(MEASURES_CONFIG.keys())
    lista_ignoradas = resolver_medidas_ignoradas(ignore_measures)
    set_ignoradas = set(lista_ignoradas)
    medidas_avaliadas = [m for m in medidas_ordem if m not in set_ignoradas]

    if not medidas_avaliadas:
        raise ValueError("Todas as medidas foram ignoradas. Mantenha ao menos uma medida ativa para avaliação.")

    model_resolvido = pipeline.resolver_modelo_padrao(model_path)
    tag_modelo, nome_run_modelo = pipeline.extrair_identificador_modelo(model_resolvido)
    run_dir, run_id = criar_diretorio_execucao(out_dir, tag_modelo)

    arquivos_imagem = [
        nome
        for nome in sorted(os.listdir(inputs_dir))
        if os.path.isfile(os.path.join(inputs_dir, nome)) and nome.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ]

    sim_pct_cfg = sim_threshold * 100.0
    erro_max_cfg = (1.0 - sim_threshold) * 100.0
    labels_ignoradas = [MEASURES_CONFIG[m]["label"] for m in lista_ignoradas]
    labels_avaliadas = [MEASURES_CONFIG[m]["label"] for m in medidas_avaliadas]

    print("=" * 100)
    print("         ONEVISION - ESTIMATIVA DE DISTÂNCIA DA CÂMERA E AUDITORIA DE ANOTAÇÕES (inputs/)         ")
    print("=" * 100)
    print(f" > Execução (Run ID)    : {run_id}")
    print(f" > Pasta da Execução    : {run_dir}")
    print(f" > Modelo / Pesos       : {model_resolvido} [{nome_run_modelo} | tag: {tag_modelo}]")
    print(f" > Parâmetros YOLO      : conf={conf_thresh} | imgsz={imgsz}")
    print(
        f" > Métrica Proximidade  : sim = max(0.0, 1.0 - |pred - real| / real) | "
        f"Limiar = {sim_threshold:.2f} ({sim_pct_cfg:.1f}% acerto / máx {erro_max_cfg:.1f}% erro)"
    )
    print(f" > Medidas Avaliadas    : {', '.join(labels_avaliadas)}")
    print(f" > Medidas Ignoradas    : {', '.join(labels_ignoradas) if labels_ignoradas else 'Nenhuma'}")
    print(
        f" > Busca de Distância   : dist_anotada ± {search_range:.0f} cm | passo = {search_step:.0f} cm | "
        f"limites = [{min_dist:.0f}, {max_dist:.0f}] cm"
    )
    print(f" > Critério de Seleção  : {criterion} ({CRITERIOS_DESCRICAO.get(criterion, criterion)})")
    print(f" > Destino de Cópias    : {copy_dir}/ (para imagens com status 'ajustada')")
    print("=" * 100)

    modelo_yolo = YOLO(model_resolvido)

    registros: list[dict[str, Any]] = []
    resumo_status = {"ok": 0, "ajustada": 0, "sem_melhora": 0, "sem_dado": 0, "copiadas": 0}
    avisos_lote: list[str] = []

    for idx, nome_arq in enumerate(arquivos_imagem, start=1):
        img_path = os.path.join(inputs_dir, nome_arq)
        id_animal = extrair_id_animal(nome_arq)
        dist_anotada = extrair_distancia_anotada(nome_arq)
        gt_item = banco_gt.get(id_animal) if id_animal is not None else None

        # 1. Validação de correspondência no JSON e padrão dist(\d+) no nome do arquivo
        if id_animal is None or gt_item is None or dist_anotada is None:
            motivos = []
            if id_animal is None:
                motivos.append("sem prefixo numérico de ID")
            elif gt_item is None:
                motivos.append(f"ID {id_animal:03d} ausente em {gt_file}")
            if dist_anotada is None:
                motivos.append("padrão dist(\\d+) ausente no nome")
            motivo_str = "; ".join(motivos)

            aviso = f"[AVISO] [{idx:02d}/{len(arquivos_imagem):02d}] {nome_arq}: {motivo_str} -> status='sem_dado'"
            print(aviso)
            avisos_lote.append(aviso)
            resumo_status["sem_dado"] += 1

            registros.append({
                "imagem": nome_arq,
                "image_path": img_path,
                "id_animal": id_animal,
                "identificacao": gt_item.get("identificacao", "") if gt_item else "",
                "dist_anotada": dist_anotada,
                "dist_estimada": None,
                "status": "sem_dado",
                "motivo_status": motivo_str,
                "avaliacao_anotada": None,
                "avaliacao_final": None,
                "copia_info": None,
            })
            continue

        # 2. Executa a inferência YOLO com a distância anotada (reutilizando main.executar_inferencia)
        print(
            f"[{idx:02d}/{len(arquivos_imagem):02d}] Processando {nome_arq} "
            f"(ID: {id_animal:03d} | Dist. Anotada: {dist_anotada:.0f} cm)...",
            end=" ",
        )

        resultado_inf = pipeline.executar_inferencia(
            image_path=img_path,
            model_path=model_resolvido,
            dist_cm=dist_anotada,
            conf_thresh=conf_thresh,
            ground_truth_file=gt_file,
            imgsz=imgsz,
            model=modelo_yolo,
            save_image=False,
            verbose=False,
        )

        det_kpts = extrair_keypoints_principais(
            image_path=img_path,
            modelo_yolo=modelo_yolo,
            conf_thresh=conf_thresh,
            imgsz=imgsz,
        )

        if not resultado_inf or not resultado_inf.get("estimativas") or det_kpts is None:
            motivo_str = "falha na detecção de keypoints/pose"
            aviso = f"AVISO ({motivo_str}) -> status='sem_dado'"
            print(aviso)
            avisos_lote.append(f"[AVISO] {nome_arq}: {motivo_str}")
            resumo_status["sem_dado"] += 1

            registros.append({
                "imagem": nome_arq,
                "image_path": img_path,
                "id_animal": id_animal,
                "identificacao": gt_item.get("identificacao", f"ID {id_animal}"),
                "dist_anotada": dist_anotada,
                "dist_estimada": None,
                "status": "sem_dado",
                "motivo_status": "sem detecção",
                "avaliacao_anotada": None,
                "avaliacao_final": None,
                "copia_info": None,
            })
            continue

        kpts_principal, img_w, _ = det_kpts
        estimativas_anotada = resultado_inf["estimativas"]

        av_anotada = avaliar_estimativas_na_distancia(
            estimativas=estimativas_anotada,
            gt_item=gt_item,
            medidas_todas=medidas_ordem,
            medidas_ignoradas=set_ignoradas,
            sim_threshold=sim_threshold,
        )

        # 3. Verifica se todas as medidas avaliadas atingiram o limiar mínimo configurado
        if av_anotada["todas_acima_limiar"]:
            status = "ok"
            dist_estimada = dist_anotada
            av_final = av_anotada
            copia_info = None
            resumo_status["ok"] += 1
            print(
                f"OK (Consistente em {dist_anotada:.0f} cm | "
                f"Sim.Mín={av_anotada['min_sim']*100:.1f}% | Sim.Méd={av_anotada['mean_sim']*100:.1f}%)"
            )
        else:
            # 4. Busca distância candidata alternativa
            busca = buscar_melhor_distancia(
                kpts=kpts_principal,
                img_w=img_w,
                gt_item=gt_item,
                dist_anotada=dist_anotada,
                av_anotada=av_anotada,
                medidas_todas=medidas_ordem,
                medidas_ignoradas=set_ignoradas,
                sim_threshold=sim_threshold,
                search_range=search_range,
                search_step=search_step,
                min_dist=min_dist,
                max_dist=max_dist,
                criterion=criterion,
            )

            status = busca["status"]
            dist_estimada = busca["dist_estimada"]
            av_final = busca["avaliacao_estimada"]
            resumo_status[status] += 1

            if status == "ajustada":
                copia_info = copiar_imagem_distancia_estimada(
                    image_path=img_path,
                    dest_dir=copy_dir,
                    dist_anotada=dist_anotada,
                    dist_estimada=dist_estimada,
                )
                resumo_status["copiadas"] += 1
                delta = dist_estimada - dist_anotada
                sinal = "+" if delta > 0 else ""
                print(
                    f"AJUSTADA ({dist_anotada:.0f} -> {dist_estimada:.0f} cm [{sinal}{delta:.0f} cm] | "
                    f"Sim.Mín: {av_anotada['min_sim']*100:.1f}% -> {av_final['min_sim']*100:.1f}% | "
                    f"Sim.Méd: {av_anotada['mean_sim']*100:.1f}% -> {av_final['mean_sim']*100:.1f}% | "
                    f"Cópia: '{copia_info['caminho_copia']}')"
                )
            else:
                copia_info = None
                print(
                    f"SEM_MELHORA (Mantida {dist_anotada:.0f} cm | "
                    f"Sim.Mín={av_anotada['min_sim']*100:.1f}% | Sim.Méd={av_anotada['mean_sim']*100:.1f}%)"
                )

        registros.append({
            "imagem": nome_arq,
            "image_path": img_path,
            "id_animal": id_animal,
            "identificacao": gt_item.get("identificacao", f"ID {id_animal}"),
            "dist_anotada": float(dist_anotada),
            "dist_estimada": float(dist_estimada),
            "status": status,
            "motivo_status": "",
            "avaliacao_anotada": av_anotada,
            "avaliacao_final": av_final,
            "copia_info": copia_info,
        })

    # 5. Relatório Detalhado no Terminal
    print("\n" + "=" * 100)
    print(f"                      RESUMO GERAL DA EXECUÇÃO ({run_id})                      ")
    print("=" * 100)
    print(f" > Total de imagens inspecionadas em '{inputs_dir}/' : {len(registros)}")
    print(f" > Status 'ok' (distância anotada consistente)       : {resumo_status['ok']}")
    print(f" > Status 'ajustada' (nova distância estimada)       : {resumo_status['ajustada']}")
    print(f" > Status 'sem_melhora' (abaixo do limiar s/ ajuste) : {resumo_status['sem_melhora']}")
    print(f" > Status 'sem_dado' (sem padrão/gabarito/detecção)  : {resumo_status['sem_dado']}")
    print(f" > Imagens copiadas para '{copy_dir}/'      : {resumo_status['copiadas']}")
    print("-" * 100)
    print("DETALHE POR IMAGEM (Métricas na distância final selecionada; [IGN] = medida ignorada na decisão):")
    print("-" * 100)
    print(
        f"{'Imagem':<24} | {'Status':<12} | {'Anot.':<7} | {'Est.':<7} | "
        f"{'Sim.Mín (A->E)':<16} | {'Métricas por Medida (Sim% e Erro%)'}"
    )
    print("-" * 100)

    for r in registros:
        d_anot_s = f"{r['dist_anotada']:.0f}cm" if r["dist_anotada"] is not None else "N/A"
        d_est_s = f"{r['dist_estimada']:.0f}cm" if r["dist_estimada"] is not None else "N/A"

        if r["avaliacao_anotada"] and r["avaliacao_final"]:
            s_min_a = r["avaliacao_anotada"]["min_sim"] * 100.0
            s_min_f = r["avaliacao_final"]["min_sim"] * 100.0
            sim_trans = f"{s_min_a:4.1f}% -> {s_min_f:4.1f}%"
            medidas_str = formatar_resumo_medidas_linha(r["avaliacao_final"]["medidas"], medidas_ordem)
        else:
            sim_trans = "N/A"
            medidas_str = f"Motivo: {r.get('motivo_status', 'sem dados')}"

        print(
            f"{r['imagem']:<24} | {r['status']:<12} | {d_anot_s:<7} | {d_est_s:<7} | "
            f"{sim_trans:<16} | {medidas_str}"
        )

    if resumo_status["copiadas"] > 0:
        print("-" * 100)
        print(f"IMAGENS COPIADAS PARA '{copy_dir}/' COM DISTÂNCIA ESTIMADA NO NOME:")
        for r in registros:
            if r.get("copia_info"):
                ci = r["copia_info"]
                print(
                    f" -> {ci['nome_original']} | Dist. Anotada: {ci['dist_anotada_cm']:.0f} cm -> "
                    f"Dist. Estimada: {ci['dist_estimada_cm']:.0f} cm | "
                    f"Salva como: {ci['caminho_copia']}"
                )

    print("=" * 100)

    parametros_exec = {
        "model_path": model_resolvido,
        "conf": conf_thresh,
        "imgsz": imgsz,
        "inputs_dir": inputs_dir,
        "gt_file": gt_file,
        "out_dir": out_dir,
        "run_dir": run_dir,
        "copy_dir": copy_dir,
        "sim_threshold": sim_threshold,
        "ignore_measures": lista_ignoradas,
        "search_range": search_range,
        "search_step": search_step,
        "min_dist": min_dist,
        "max_dist": max_dist,
        "criterion": criterion,
    }

    caminho_png = gerar_relatorio_png(
        registros=registros,
        resumo_status=resumo_status,
        parametros=parametros_exec,
        medidas_ordem=medidas_ordem,
        medidas_ignoradas=lista_ignoradas,
        nome_modelo=nome_run_modelo,
        tag_modelo=tag_modelo,
        run_dir=run_dir,
        run_id=run_id,
    )

    caminho_json, caminho_csv = salvar_artefatos_estruturados(
        run_dir=run_dir,
        run_id=run_id,
        tag_modelo=tag_modelo,
        nome_modelo=nome_run_modelo,
        parametros=parametros_exec,
        resumo_status=resumo_status,
        registros=registros,
        medidas_ordem=medidas_ordem,
        medidas_ignoradas=lista_ignoradas,
        caminho_png=caminho_png,
    )

    print("\nArtefatos da execução salvos em:")
    print(f" -> [Relatório PNG]  {caminho_png}")
    print(f" -> [Relatório JSON] {caminho_json}")
    print(f" -> [Relatório CSV]  {caminho_csv}")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "relatorio_png": caminho_png,
        "relatorio_json": caminho_json,
        "relatorio_csv": caminho_csv,
        "resumo_status": resumo_status,
        "registros": registros,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "OneVision - Estimativa de Distância da Câmera e Auditoria de Anotações de Distância em inputs/ "
            "(YOLO Pose + AP-10K vs Ground Truth)"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt",
        help="Caminho para os pesos do modelo YOLO (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar de confiança para a inferência do YOLO (padrão: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Resolução da imagem na inferência (padrão: 1280)",
    )
    parser.add_argument(
        "--inputs-dir",
        type=str,
        default="inputs",
        help="Pasta contendo as imagens de entrada (padrão: inputs)",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="dados_biometria.json",
        help="Arquivo JSON com os dados biométricos reais / ground truth (padrão: dados_biometria.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/estimativa-distancias",
        help="Diretório base onde cada execução criará sua subpasta exclusiva (padrão: outputs/estimativa-distancias)",
    )
    parser.add_argument(
        "--copy-dir",
        type=str,
        default="inputs/distancia_estimada",
        help="Diretório onde serão salvas cópias das imagens com distância ajustada (padrão: inputs/distancia_estimada)",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.90,
        help="Limiar mínimo de similaridade (0 a 1) para considerar a medida consistente (padrão: 0.90 = até 10%% de erro)",
    )
    parser.add_argument(
        "--ignore-measures",
        type=str,
        default="grossura_pescoco",
        help=(
            "Medidas a ignorar na decisão separadas por vírgula (ex: 'grossura_pescoco', 'pescoco,peso' ou 'none' para nenhuma). "
            "Padrão: grossura_pescoco"
        ),
    )
    parser.add_argument(
        "--search-range",
        type=float,
        default=60.0,
        help="Alcance de busca em cm para menos e para mais a partir da distância anotada (padrão: 60.0)",
    )
    parser.add_argument(
        "--search-step",
        type=float,
        default=5.0,
        help="Passo em cm entre as distâncias candidatas na busca (padrão: 5.0)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=30.0,
        help="Distância mínima positiva permitida na busca em cm (padrão: 30.0)",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=500.0,
        help="Distância máxima permitida na busca em cm (padrão: 500.0)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=["min_sim", "mean_sim", "converged_then_mean"],
        default="min_sim",
        help=(
            "Critério de seleção da melhor distância candidata: "
            "'min_sim' (maior similaridade mínima entre medidas avaliadas - padrão), "
            "'mean_sim' (maior similaridade média / menor erro relativo médio) ou "
            "'converged_then_mean' (maior número de medidas >= limiar, desempatando pela similaridade mínima e média)"
        ),
    )

    args = parser.parse_args()

    executar_estimativa_distancias(
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        inputs_dir=args.inputs_dir,
        gt_file=args.gt,
        out_dir=args.out_dir,
        copy_dir=args.copy_dir,
        sim_threshold=args.sim_threshold,
        ignore_measures=args.ignore_measures,
        search_range=args.search_range,
        search_step=args.search_step,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        criterion=args.criterion,
    )
