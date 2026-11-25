import os
import re
import csv
import json
import argparse
from datetime import datetime
from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ultralytics import YOLO
import main as pipeline


# Mapeamento das medidas suportadas pelo pipeline e suas respectivas chaves no JSON (ground truth)
MEASURES_CONFIG = {
    "peso": {
        "label": "Peso Vivo",
        "unit": "kg",
        "pred_key": "peso_estimado_kg",
        "gt_keys": ("peso_kg", "peso", "peso_real_kg"),
        "aliases": ("peso", "peso_kg", "weight"),
    },
    "altura_cernelha": {
        "label": "Altura da Cernelha",
        "unit": "cm",
        "pred_key": "altura_cernelha_cm",
        "gt_keys": ("altura_cernelha", "altura_cernelha_cm", "cernelha_cm"),
        "aliases": ("altura_cernelha", "cernelha", "altura"),
    },
    "altura_garupa": {
        "label": "Altura da Garupa",
        "unit": "cm",
        "pred_key": "altura_garupa_cm",
        "gt_keys": ("altura_garupa", "altura_garupa_cm", "garupa_cm"),
        "aliases": ("altura_garupa", "garupa"),
    },
    "comprimento": {
        "label": "Comprimento Corporal",
        "unit": "cm",
        "pred_key": "comprimento_cm",
        "gt_keys": ("comprimento", "comprimento_cm", "comprimento_corpo_cm"),
        "aliases": ("comprimento", "comprimento_cm", "length"),
    },
    "circ_torax": {
        "label": "Perímetro Torácico",
        "unit": "cm",
        "pred_key": "circ_torax_cm",
        "gt_keys": ("circ_do_torax", "circ_torax_cm", "perimetro_torax_cm", "perimetro_toracico_cm"),
        "aliases": ("circ_torax", "circ_do_torax", "torax", "perimetro_toracico"),
    },
    "grossura_pescoco": {
        "label": "Grossura do Pescoço",
        "unit": "cm",
        "pred_key": "grossura_pescoco_cm",
        "gt_keys": ("pescoco_circ_medio", "grossura_pescoco_cm", "pescoco_cm"),
        "aliases": ("grossura_pescoco", "pescoco", "pescoco_circ_medio"),
    },
}

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def gerar_caminho_unico(out_dir: str, nome_base: str, ext: str) -> str:
    """
    Gera um caminho de arquivo único dentro de out_dir sem sobrescrever arquivos existentes.
    Se '<nome_base><ext>' já existir, incrementa para '<nome_base>-1<ext>', '<nome_base>-2<ext>', etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not ext.startswith("."):
        ext = f".{ext}"

    caminho_inicial = os.path.join(out_dir, f"{nome_base}{ext}")
    if not os.path.exists(caminho_inicial):
        return caminho_inicial

    contador = 1
    while True:
        caminho_candidato = os.path.join(out_dir, f"{nome_base}-{contador}{ext}")
        if not os.path.exists(caminho_candidato):
            return caminho_candidato
        contador += 1


def criar_diretorio_execucao(base_out_dir: str, tag_modelo: str) -> tuple[str, str]:
    """
    Cria um subdiretório exclusivo dentro de base_out_dir (ex: outputs/evaluation/run_YYYYMMDD_HHMMSS_<tag>/)
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


def resolver_medidas_selecionadas(measure_arg: str) -> list[str]:
    """Resolve o argumento --measure (ex: 'peso', 'altura', 'all') para chaves canônicas de MEASURES_CONFIG."""
    arg_limpo = measure_arg.strip().lower()
    if arg_limpo == "all":
        return list(MEASURES_CONFIG.keys())

    escolhidas = []
    for parte in [p.strip() for p in arg_limpo.split(",") if p.strip()]:
        encontrada = None
        for canon_key, cfg in MEASURES_CONFIG.items():
            if parte == canon_key or parte in cfg["aliases"]:
                encontrada = canon_key
                break
        if not encontrada:
            opcoes = ", ".join(list(MEASURES_CONFIG.keys()) + ["all"])
            raise ValueError(f"Medida '{parte}' não reconhecida. Opções válidas: {opcoes}")
        if encontrada not in escolhidas:
            escolhidas.append(encontrada)

    return escolhidas


def carregar_ground_truth(gt_path: str) -> dict[int, dict]:
    """Carrega o arquivo dados_biometria.json indexado pelo id_array."""
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Arquivo de ground truth não encontrado: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    return {int(item["id_array"]): item for item in dados if "id_array" in item}


def filtrar_imagens_com_gabarito(inputs_dir: str, banco_gt: dict[int, dict]) -> list[tuple[str, int, dict]]:
    """
    Lê a pasta inputs/ e retorna apenas as imagens cujo prefixo numérico (ex: '006-...')
    possui registro correspondente no dados_biometria.json.
    """
    if not os.path.exists(inputs_dir):
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {inputs_dir}")

    imagens_validas = []
    for nome_arq in sorted(os.listdir(inputs_dir)):
        caminho_completo = os.path.join(inputs_dir, nome_arq)
        if not os.path.isfile(caminho_completo):
            continue
        if not nome_arq.lower().endswith(VALID_IMAGE_EXTENSIONS):
            continue

        match_id = re.match(r"^(\d+)", nome_arq)
        if not match_id:
            continue

        id_animal = int(match_id.group(1))
        if id_animal in banco_gt:
            imagens_validas.append((caminho_completo, id_animal, banco_gt[id_animal]))

    return imagens_validas


def calcular_metricas_medida(registros: list[dict], measure_key: str) -> dict:
    """
    Calcula MAPE, Acurácia (100% - MAPE), MAE e RMSE para uma medida biométrica específica.
    """
    cfg = MEASURES_CONFIG[measure_key]
    amostras = []

    for reg in registros:
        est_dict = reg["estimativas"]
        gt_dict = reg["gt"]

        val_pred = est_dict.get(cfg["pred_key"])
        val_real = pipeline.extrair_valor_gabarito(gt_dict, *cfg["gt_keys"])

        if val_pred is not None and val_real is not None and val_real > 0:
            erro_abs = abs(val_pred - val_real)
            erro_pct = (erro_abs / val_real) * 100.0
            acerto_pct = max(0.0, 100.0 - erro_pct)
            amostras.append({
                "imagem": os.path.basename(reg["image_path"]),
                "label_curto": os.path.splitext(os.path.basename(reg["image_path"]))[0],
                "id_animal": reg["id_animal"],
                "identificacao": gt_dict.get("identificacao", f"ID {reg['id_animal']}"),
                "distancia_usada_cm": reg.get("distancia_cm"),
                "real": float(val_real),
                "inferido": float(val_pred),
                "erro_abs": float(erro_abs),
                "erro_pct": float(erro_pct),
                "acerto_pct": float(acerto_pct),
            })

    if not amostras:
        return {
            "measure_key": measure_key,
            "label": cfg["label"],
            "unit": cfg["unit"],
            "n_amostras": 0,
            "mape": None,
            "acuracia": None,
            "mae": None,
            "rmse": None,
            "amostras": [],
        }

    erros_pct = np.array([a["erro_pct"] for a in amostras], dtype=float)
    erros_abs = np.array([a["erro_abs"] for a in amostras], dtype=float)

    mape = float(np.mean(erros_pct))
    acuracia = float(max(0.0, 100.0 - mape))
    mae = float(np.mean(erros_abs))
    rmse = float(np.sqrt(np.mean(erros_abs ** 2)))

    return {
        "measure_key": measure_key,
        "label": cfg["label"],
        "unit": cfg["unit"],
        "n_amostras": len(amostras),
        "mape": mape,
        "acuracia": acuracia,
        "mae": mae,
        "rmse": rmse,
        "amostras": amostras,
    }


def _formatar_texto_parametros(parametros: dict[str, Any]) -> str:
    """Formata o bloco padronizado de parâmetros da execução para exibição no cabeçalho/rodapé dos gráficos PNG."""
    dist_info = (
        f"Fixa ({parametros['dist_cm']:.0f} cm)"
        if parametros.get("dist_cm") is not None
        else "Automática (extraída de 'dist<N>' no nome da imagem; fallback 160 cm)"
    )
    medidas_labels = [
        MEASURES_CONFIG[m]["label"] if m in MEASURES_CONFIG else m
        for m in parametros.get("medidas_selecionadas", [])
    ]
    poses_info = (
        f"Sim ({parametros.get('poses_dir', parametros.get('run_dir'))})"
        if parametros.get("save_poses")
        else "Não"
    )

    return (
        f"PARÂMETROS DA EXECUÇÃO ({parametros.get('run_id', '-')})   |   Data/Hora: {parametros.get('timestamp', '-')}\n"
        f"• Modelo / Pesos : {parametros.get('model_path')}  [run: {parametros.get('modelo_nome')} | tag: {parametros.get('modelo_tag')}]\n"
        f"• Inferência YOLO: conf={parametros.get('conf')} | imgsz={parametros.get('imgsz')} | Distância Câmera: {dist_info} | Calibração Focal: {pipeline.FOCAL_CALIBRATION_CONSTANT:.0f} (Base W={pipeline.BASE_IMAGE_WIDTH:.0f}px)\n"
        f"• Dados & Filtros: inputs='{parametros.get('inputs_dir')}/' | gabarito='{parametros.get('gt_file')}' | Imagens válidas: {parametros.get('n_inferencias_validas', 0)}/{parametros.get('n_imagens_com_gt', 0)} "
        f"| plot_type='{parametros.get('plot_type')}' | save_poses={poses_info}\n"
        f"• Medidas (--measure='{parametros.get('measure_arg')}'): {', '.join(medidas_labels)}"
    )


def _adicionar_painel_parametros_figura(fig: plt.Figure, parametros: Optional[dict[str, Any]]) -> None:
    """Adiciona uma caixa estilizada com todos os parâmetros da execução na base da figura."""
    if not parametros:
        return

    texto_param = _formatar_texto_parametros(parametros)
    fig.text(
        0.012,
        0.012,
        texto_param,
        fontsize=8.2,
        va="bottom",
        ha="left",
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.50",
            "facecolor": "#f4f7fa",
            "edgecolor": "#2b8cbe",
            "linewidth": 1.2,
        },
    )


def gerar_grafico_medida(
    metrica_info: dict,
    nome_modelo: str,
    tag_modelo: str,
    out_dir: str,
    plot_type: str = "both",
    parametros: Optional[dict[str, Any]] = None,
) -> str:
    """
    Gera e salva o gráfico comparando Valor Real vs Valor Inferido (Scatter Plot e/ou Barras Emparelhadas),
    incluindo obrigatoriamente no título, legenda e painel de metadados:
      - Nome do modelo utilizado e identificador da run
      - Nome da medida avaliada
      - Porcentagem de acerto final (Acurácia = 100% - MAPE), MAPE, MAE e RMSE
      - Todos os parâmetros de configuração utilizados na execução
    """
    os.makedirs(out_dir, exist_ok=True)

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="deep")
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    amostras = metrica_info["amostras"]
    label_medida = metrica_info["label"]
    unidade = metrica_info["unit"]
    acuracia = metrica_info["acuracia"]
    mape = metrica_info["mape"]
    mae = metrica_info["mae"]
    rmse = metrica_info["rmse"]
    measure_key = metrica_info["measure_key"]
    n_amostras = metrica_info["n_amostras"]

    reais = np.array([a["real"] for a in amostras])
    inferidos = np.array([a["inferido"] for a in amostras])
    labels_x = [a["label_curto"] for a in amostras]

    run_id_str = f" | Run: {parametros['run_id']}" if (parametros and parametros.get("run_id")) else ""
    titulo_principal = (
        f"Avaliação Biométrica: {label_medida} ({unidade}){run_id_str}\n"
        f"Modelo: {nome_modelo} [{tag_modelo}] | Acurácia Final: {acuracia:.2f}% "
        f"(MAPE: {mape:.2f}% | MAE: {mae:.2f} {unidade} | RMSE: {rmse:.2f} {unidade} | N={n_amostras})"
    )
    legenda_identificacao = f"{nome_modelo} | {label_medida} | Acerto: {acuracia:.2f}%"

    if plot_type == "scatter":
        fig, ax_scatter = plt.subplots(figsize=(10.5, 8.2), dpi=150)
    elif plot_type == "bars":
        fig, ax_bars = plt.subplots(figsize=(max(12.5, len(amostras) * 0.45), 8.2), dpi=150)
    else:
        fig, (ax_scatter, ax_bars) = plt.subplots(
            1, 2, figsize=(18.5, 8.6), dpi=150, gridspec_kw={"width_ratios": [1.0, 1.35]}
        )

    fig.suptitle(titulo_principal, fontsize=12.5, fontweight="bold", y=0.985)

    # 1. Gráfico de Dispersão (Scatter Plot: Valor Real vs Valor Inferido)
    if plot_type in ("scatter", "both"):
        if HAS_SEABORN:
            sns.scatterplot(
                x=reais,
                y=inferidos,
                ax=ax_scatter,
                s=90,
                color="#1f77b4",
                edgecolor="black",
                alpha=0.85,
                label=f"Inferências ({legenda_identificacao})",
            )
        else:
            ax_scatter.scatter(
                reais,
                inferidos,
                s=80,
                color="#1f77b4",
                edgecolors="black",
                alpha=0.85,
                label=f"Inferências ({legenda_identificacao})",
            )

        min_v = min(float(np.min(reais)), float(np.min(inferidos))) * 0.92
        max_v = max(float(np.max(reais)), float(np.max(inferidos))) * 1.08
        ax_scatter.plot(
            [min_v, max_v],
            [min_v, max_v],
            linestyle="--",
            color="#d62728",
            linewidth=2.0,
            label="Linha Ideal (Valor Real = Inferido)",
        )

        ax_scatter.set_xlim(min_v, max_v)
        ax_scatter.set_ylim(min_v, max_v)
        ax_scatter.set_title(
            f"Dispersão: Real vs Inferido — {label_medida}\nModelo: {tag_modelo} | Acerto: {acuracia:.2f}% (MAE: {mae:.2f} {unidade})",
            fontsize=11,
            fontweight="semibold",
        )
        ax_scatter.set_xlabel(f"Valor Real - Ground Truth ({unidade})", fontsize=10)
        ax_scatter.set_ylabel(f"Valor Inferido pelo Modelo ({unidade})", fontsize=10)
        ax_scatter.legend(loc="upper left", fontsize=8.5, frameon=True)
        ax_scatter.grid(True, linestyle=":", alpha=0.6)

    # 2. Gráfico de Barras Emparelhadas (Valor Real vs Valor Inferido por Imagem)
    if plot_type in ("bars", "both"):
        x_idx = np.arange(len(amostras))
        largura = 0.38

        ax_bars.bar(
            x_idx - largura / 2,
            reais,
            width=largura,
            color="#2ca02c",
            edgecolor="black",
            alpha=0.88,
            label=f"Valor Real (Gabarito) [{unidade}]",
        )
        ax_bars.bar(
            x_idx + largura / 2,
            inferidos,
            width=largura,
            color="#1f77b4",
            edgecolor="black",
            alpha=0.88,
            label=f"Valor Inferido ({legenda_identificacao})",
        )

        ax_bars.set_title(
            f"Barras Emparelhadas por Imagem — {label_medida}\nModelo: {tag_modelo} | Acerto: {acuracia:.2f}% (MAPE: {mape:.2f}%)",
            fontsize=11,
            fontweight="semibold",
        )
        ax_bars.set_xlabel("Imagens Avaliadas (inputs/)", fontsize=10)
        ax_bars.set_ylabel(f"{label_medida} ({unidade})", fontsize=10)
        ax_bars.set_xticks(x_idx)
        ax_bars.set_xticklabels(labels_x, rotation=65, ha="right", fontsize=8)
        ax_bars.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax_bars.grid(True, axis="y", linestyle=":", alpha=0.6)

    _adicionar_painel_parametros_figura(fig, parametros)
    rect_bottom = 0.145 if parametros else 0.0
    plt.tight_layout(rect=[0, rect_bottom, 1, 0.915])

    caminho_grafico = gerar_caminho_unico(out_dir, f"eval_{tag_modelo}_{measure_key}", ".png")
    fig.savefig(caminho_grafico, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return caminho_grafico


def gerar_grafico_resumo_geral(
    resultados_medidas: list[dict],
    nome_modelo: str,
    tag_modelo: str,
    out_dir: str,
    parametros: Optional[dict[str, Any]] = None,
) -> str:
    """
    Gera um gráfico de resumo com a Porcentagem de Acerto (Acurácia), MAPE, MAE e RMSE
    de todas as medidas avaliadas, acompanhado de todos os parâmetros da execução.
    """
    validas = [m for m in resultados_medidas if m["acuracia"] is not None]
    if len(validas) < 2:
        return ""

    os.makedirs(out_dir, exist_ok=True)
    nomes = [m["label"] for m in validas]
    acuracias = [m["acuracia"] for m in validas]
    mapes = [m["mape"] for m in validas]
    maes = [m["mae"] for m in validas]
    unidades = [m["unit"] for m in validas]
    media_global = float(np.mean(acuracias))
    mape_medio_global = float(np.mean(mapes))

    fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=150)
    x_idx = np.arange(len(validas))
    cores = ["#2b8cbe" if ac >= 80 else "#fdae6b" if ac >= 65 else "#de2d26" for ac in acuracias]

    barras = ax.bar(
        x_idx,
        acuracias,
        color=cores,
        edgecolor="black",
        width=0.55,
        label=(
            f"Modelo: {nome_modelo} ({tag_modelo}) | "
            f"Acurácia Média Global: {media_global:.2f}% (MAPE Médio: {mape_medio_global:.2f}%)"
        ),
    )

    for bar, ac, mp, mae_v, und in zip(barras, acuracias, mapes, maes, unidades):
        altura = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            altura + 1.2,
            f"{ac:.1f}%\n(MAPE {mp:.1f}% | MAE {mae_v:.1f}{und})",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )

    run_id_str = f" | Run: {parametros['run_id']}" if (parametros and parametros.get("run_id")) else ""
    ax.set_ylim(0, 115)
    ax.axhline(media_global, color="#d62728", linestyle="--", linewidth=1.8, label=f"Média Global: {media_global:.2f}%")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(nomes, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylabel("Porcentagem de Acerto / Acurácia (%)", fontsize=10)
    ax.set_title(
        f"Resumo Geral de Acurácia por Medida Biométrica{run_id_str}\n"
        f"Modelo: {nome_modelo} [{tag_modelo}] | Acerto Médio Global: {media_global:.2f}% (MAPE Médio: {mape_medio_global:.2f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    _adicionar_painel_parametros_figura(fig, parametros)
    rect_bottom = 0.175 if parametros else 0.0
    plt.tight_layout(rect=[0, rect_bottom, 1, 0.96])

    caminho_resumo = gerar_caminho_unico(out_dir, f"eval_{tag_modelo}_resumo_todas_medidas", ".png")
    fig.savefig(caminho_resumo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho_resumo


def salvar_csv_avaliacao(
    run_dir: str,
    tag_modelo: str,
    sufixo_medida: str,
    parametros: dict[str, Any],
    resultados_finais: list[dict],
) -> tuple[str, str]:
    """
    Salva dois arquivos CSV na pasta da execução contendo todos os parâmetros e métricas:
      1. CSV de resumo por medida (com colunas de parâmetros da run + MAE, RMSE, MAPE, Acurácia)
      2. CSV detalhado por amostra/imagem e medida avaliada (com parâmetros da run + Real, Inferido, Erros)
    """
    caminho_csv_resumo = gerar_caminho_unico(run_dir, f"resumo_metricas_{tag_modelo}_{sufixo_medida}", ".csv")
    caminho_csv_amostras = gerar_caminho_unico(run_dir, f"detalhe_amostras_{tag_modelo}_{sufixo_medida}", ".csv")

    campos_resumo = [
        "run_id",
        "timestamp",
        "modelo_nome",
        "modelo_tag",
        "model_path",
        "conf",
        "imgsz",
        "dist_cm_fixa",
        "inputs_dir",
        "gt_file",
        "plot_type",
        "save_poses",
        "measure_arg",
        "measure_key",
        "measure_label",
        "unit",
        "n_amostras",
        "mae",
        "rmse",
        "mape_pct",
        "acuracia_pct",
    ]

    with open(caminho_csv_resumo, "w", encoding="utf-8", newline="") as f_res:
        writer = csv.DictWriter(f_res, fieldnames=campos_resumo)
        writer.writeheader()
        for info in resultados_finais:
            writer.writerow({
                "run_id": parametros["run_id"],
                "timestamp": parametros["timestamp"],
                "modelo_nome": parametros["modelo_nome"],
                "modelo_tag": parametros["modelo_tag"],
                "model_path": parametros["model_path"],
                "conf": parametros["conf"],
                "imgsz": parametros["imgsz"],
                "dist_cm_fixa": parametros["dist_cm"] if parametros["dist_cm"] is not None else "auto_filename",
                "inputs_dir": parametros["inputs_dir"],
                "gt_file": parametros["gt_file"],
                "plot_type": parametros["plot_type"],
                "save_poses": parametros["save_poses"],
                "measure_arg": parametros["measure_arg"],
                "measure_key": info["measure_key"],
                "measure_label": info["label"],
                "unit": info["unit"],
                "n_amostras": info["n_amostras"],
                "mae": round(info["mae"], 4) if info["mae"] is not None else None,
                "rmse": round(info["rmse"], 4) if info["rmse"] is not None else None,
                "mape_pct": round(info["mape"], 4) if info["mape"] is not None else None,
                "acuracia_pct": round(info["acuracia"], 4) if info["acuracia"] is not None else None,
            })

    campos_amostras = [
        "run_id",
        "timestamp",
        "modelo_nome",
        "modelo_tag",
        "conf",
        "imgsz",
        "dist_cm_config",
        "imagem",
        "id_animal",
        "identificacao",
        "distancia_usada_cm",
        "measure_key",
        "measure_label",
        "unit",
        "real",
        "inferido",
        "erro_abs",
        "erro_pct",
        "acerto_pct",
    ]

    with open(caminho_csv_amostras, "w", encoding="utf-8", newline="") as f_am:
        writer = csv.DictWriter(f_am, fieldnames=campos_amostras)
        writer.writeheader()
        for info in resultados_finais:
            for am in info["amostras"]:
                writer.writerow({
                    "run_id": parametros["run_id"],
                    "timestamp": parametros["timestamp"],
                    "modelo_nome": parametros["modelo_nome"],
                    "modelo_tag": parametros["modelo_tag"],
                    "conf": parametros["conf"],
                    "imgsz": parametros["imgsz"],
                    "dist_cm_config": parametros["dist_cm"] if parametros["dist_cm"] is not None else "auto_filename",
                    "imagem": am["imagem"],
                    "id_animal": am["id_animal"],
                    "identificacao": am["identificacao"],
                    "distancia_usada_cm": am.get("distancia_usada_cm"),
                    "measure_key": info["measure_key"],
                    "measure_label": info["label"],
                    "unit": info["unit"],
                    "real": round(am["real"], 4),
                    "inferido": round(am["inferido"], 4),
                    "erro_abs": round(am["erro_abs"], 4),
                    "erro_pct": round(am["erro_pct"], 4),
                    "acerto_pct": round(am["acerto_pct"], 4),
                })

    return caminho_csv_resumo, caminho_csv_amostras


def executar_avaliacao(
    model_path: str,
    conf_thresh: float = 0.25,
    imgsz: int = 1280,
    measure: str = "all",
    inputs_dir: str = "inputs",
    gt_file: str = "dados_biometria.json",
    out_dir: str = "outputs/evaluation",
    dist_cm: Optional[float] = None,
    save_annotated_images: bool = False,
    plot_type: str = "both",
) -> Optional[dict[str, Any]]:
    medidas_selecionadas = resolver_medidas_selecionadas(measure)
    banco_gt = carregar_ground_truth(gt_file)
    imagens_filtradas = filtrar_imagens_com_gabarito(inputs_dir, banco_gt)

    if not imagens_filtradas:
        print(f"Nenhuma imagem na pasta '{inputs_dir}' possui correspondência no arquivo '{gt_file}'.")
        return None

    model_resolvido = pipeline.resolver_modelo_padrao(model_path)
    tag_modelo, nome_run_modelo = pipeline.extrair_identificador_modelo(model_resolvido)

    # Cria subpasta exclusiva da execução dentro de outputs/evaluation/
    run_dir, run_id = criar_diretorio_execucao(out_dir, tag_modelo)
    timestamp_exec = datetime.now().isoformat(timespec="seconds")
    poses_dir = os.path.join(run_dir, "poses") if save_annotated_images else None

    dist_desc = f"{dist_cm:.0f} cm (fixa via --dist)" if dist_cm is not None else "Automática (extraída do nome do arquivo)"

    print("=" * 88)
    print("             ONEVISION - AVALIAÇÃO E VALIDAÇÃO DE MODELOS BIOMÉTRICOS             ")
    print("=" * 88)
    print(f" > Execução (Run ID)  : {run_id}")
    print(f" > Pasta da Execução  : {run_dir}")
    print(f" > Data / Hora        : {timestamp_exec}")
    print(f" > Modelo Selecionado : {model_resolvido}")
    print(f" > Identificador Run  : {nome_run_modelo} [tag: {tag_modelo}]")
    print(f" > Parâmetros YOLO    : conf={conf_thresh} | imgsz={imgsz}")
    print(f" > Distância Câmera   : {dist_desc}")
    print(f" > Fonte de Dados     : inputs='{inputs_dir}/' | gabarito='{gt_file}' ({len(imagens_filtradas)} imagens c/ GT)")
    print(f" > Medidas Avaliadas  : {', '.join(medidas_selecionadas)} (--measure='{measure}')")
    print(f" > Opções Gráficas    : plot_type='{plot_type}' | save_poses={save_annotated_images}")
    print("=" * 88)

    # Carrega o modelo YOLO uma única vez para otimizar a inferência em lote
    modelo_yolo = YOLO(model_resolvido)

    registros_inferencia = []
    falhas_deteccao = []

    for idx, (img_path, id_animal, gt_item) in enumerate(imagens_filtradas, start=1):
        nome_img = os.path.basename(img_path)
        print(f"[{idx:02d}/{len(imagens_filtradas):02d}] Inferindo: {nome_img} (ID Animal: {id_animal})...", end=" ")

        resultado = pipeline.executar_inferencia(
            image_path=img_path,
            model_path=model_resolvido,
            dist_cm=dist_cm,
            conf_thresh=conf_thresh,
            ground_truth_file=gt_file,
            imgsz=imgsz,
            model=modelo_yolo,
            save_image=save_annotated_images,
            verbose=False,
        )

        # Caso --save-poses esteja ativo, move a imagem gerada para a subpasta poses/ da própria run
        if save_annotated_images and resultado and resultado.get("caminho_saida") and poses_dir:
            os.makedirs(poses_dir, exist_ok=True)
            origem_pose = resultado["caminho_saida"]
            if os.path.exists(origem_pose):
                destino_pose = os.path.join(poses_dir, os.path.basename(origem_pose))
                os.replace(origem_pose, destino_pose)
                resultado["caminho_saida"] = destino_pose

        if resultado and resultado.get("estimativas"):
            dist_usada = resultado.get("distancia_cm")
            print(f"OK (dist={dist_usada:.0f}cm)" if dist_usada is not None else "OK")
            registros_inferencia.append({
                "image_path": img_path,
                "id_animal": id_animal,
                "distancia_cm": dist_usada,
                "caminho_pose": resultado.get("caminho_saida"),
                "estimativas": resultado["estimativas"],
                "gt": gt_item,
            })
        else:
            print("FALHA (animal/keypoints insuficientes)")
            falhas_deteccao.append(nome_img)

    parametros_exec = {
        "run_id": run_id,
        "timestamp": timestamp_exec,
        "model_path": model_resolvido,
        "modelo_nome": nome_run_modelo,
        "modelo_tag": tag_modelo,
        "conf": conf_thresh,
        "imgsz": imgsz,
        "dist_cm": dist_cm,
        "measure_arg": measure,
        "medidas_selecionadas": medidas_selecionadas,
        "inputs_dir": inputs_dir,
        "gt_file": gt_file,
        "base_out_dir": out_dir,
        "run_dir": run_dir,
        "poses_dir": poses_dir,
        "plot_type": plot_type,
        "save_poses": save_annotated_images,
        "focal_calibration_constant": pipeline.FOCAL_CALIBRATION_CONSTANT,
        "base_image_width": pipeline.BASE_IMAGE_WIDTH,
        "n_imagens_com_gt": len(imagens_filtradas),
        "n_inferencias_validas": len(registros_inferencia),
        "n_falhas_deteccao": len(falhas_deteccao),
    }

    print("\n" + "=" * 88)
    print(f"               RESULTADOS CONSOLIDADOS ({nome_run_modelo} | {run_id})               ")
    print("=" * 88)
    print(f" Imagens processadas com sucesso: {len(registros_inferencia)}/{len(imagens_filtradas)}")
    if falhas_deteccao:
        print(f" Imagens sem detecção válida ({len(falhas_deteccao)}): {', '.join(falhas_deteccao)}")
    print("-" * 88)
    print(f"{'Medida Biométrica':<24} | {'Amostras':<8} | {'MAE':<11} | {'RMSE':<11} | {'MAPE (%)':<10} | {'Acerto (%)'}")
    print("-" * 88)

    resultados_finais = []
    graficos_salvos = []

    for m_key in medidas_selecionadas:
        info = calcular_metricas_medida(registros_inferencia, m_key)
        resultados_finais.append(info)

        if info["n_amostras"] > 0:
            und = info["unit"]
            print(
                f"{info['label']:<24} | {info['n_amostras']:<8d} | "
                f"{info['mae']:6.2f} {und:<4} | {info['rmse']:6.2f} {und:<4} | "
                f"{info['mape']:8.2f}%  | {info['acuracia']:8.2f}%"
            )
            caminho_fig = gerar_grafico_medida(
                metrica_info=info,
                nome_modelo=nome_run_modelo,
                tag_modelo=tag_modelo,
                out_dir=run_dir,
                plot_type=plot_type,
                parametros=parametros_exec,
            )
            graficos_salvos.append(caminho_fig)
        else:
            print(f"{info['label']:<24} | {'0':<8} | {'N/A':<11} | {'N/A':<11} | {'N/A':<10} | {'N/A'}")

    print("=" * 88)

    if len(medidas_selecionadas) > 1:
        caminho_resumo = gerar_grafico_resumo_geral(
            resultados_medidas=resultados_finais,
            nome_modelo=nome_run_modelo,
            tag_modelo=tag_modelo,
            out_dir=run_dir,
            parametros=parametros_exec,
        )
        if caminho_resumo:
            graficos_salvos.append(caminho_resumo)

    sufixo_medida = "all" if len(medidas_selecionadas) > 1 else medidas_selecionadas[0]

    # Salva CSVs de resumo e de amostras detalhadas dentro da pasta exclusiva da run
    csv_resumo_path, csv_amostras_path = salvar_csv_avaliacao(
        run_dir=run_dir,
        tag_modelo=tag_modelo,
        sufixo_medida=sufixo_medida,
        parametros=parametros_exec,
        resultados_finais=resultados_finais,
    )

    # Salva relatório JSON detalhado na pasta exclusiva da run com todos os parâmetros utilizados
    acuracias_validas = [m["acuracia"] for m in resultados_finais if m["acuracia"] is not None]
    mapes_validos = [m["mape"] for m in resultados_finais if m["mape"] is not None]

    relatorio_json_path = gerar_caminho_unico(run_dir, f"relatorio_{tag_modelo}_{sufixo_medida}", ".json")
    with open(relatorio_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": timestamp_exec,
                "run_dir": run_dir,
                "modelo_caminho": model_resolvido,
                "modelo_nome": nome_run_modelo,
                "modelo_tag": tag_modelo,
                "parametros": parametros_exec,
                "resumo_global": {
                    "total_imagens_com_gt": len(imagens_filtradas),
                    "total_inferencias_validas": len(registros_inferencia),
                    "total_falhas_deteccao": len(falhas_deteccao),
                    "falhas_deteccao": falhas_deteccao,
                    "acuracia_media_global_pct": float(np.mean(acuracias_validas)) if acuracias_validas else None,
                    "mape_medio_global_pct": float(np.mean(mapes_validos)) if mapes_validos else None,
                },
                "arquivos_gerados": {
                    "graficos_png": graficos_salvos,
                    "csv_resumo_metricas": csv_resumo_path,
                    "csv_detalhe_amostras": csv_amostras_path,
                },
                "metricas_por_medida": resultados_finais,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nArtefatos da execução ({run_id}) salvos em '{run_dir}':")
    for g in graficos_salvos:
        print(f" -> [Gráfico PNG]      {g}")
    print(f" -> [Relatório JSON]   {relatorio_json_path}")
    print(f" -> [Resumo CSV]       {csv_resumo_path}")
    print(f" -> [Amostras CSV]     {csv_amostras_path}")
    if poses_dir and os.path.exists(poses_dir):
        print(f" -> [Poses Anotadas]   {poses_dir}")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "graficos_png": graficos_salvos,
        "relatorio_json": relatorio_json_path,
        "csv_resumo": csv_resumo_path,
        "csv_amostras": csv_amostras_path,
        "parametros": parametros_exec,
        "metricas_por_medida": resultados_finais,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avaliação e Validação de Modelos Biométricos (YOLO Pose + AP-10K vs Ground Truth)"
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
        "--measure",
        type=str,
        default="all",
        help=(
            "Medida biométrica para avaliar: 'peso', 'altura_cernelha' (ou 'altura'), "
            "'altura_garupa', 'comprimento', 'circ_torax', 'grossura_pescoco' ou 'all' para todas (padrão: all)"
        ),
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
        help="Arquivo JSON com os dados reais / ground truth (padrão: dados_biometria.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/evaluation",
        help="Diretório base onde cada execução criará sua subpasta exclusiva (padrão: outputs/evaluation)",
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=None,
        help="Distância fixa da câmera em cm (opcional; se omitida, deduz do nome do arquivo)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["both", "scatter", "bars"],
        default="both",
        help="Tipo de gráfico a gerar: 'scatter' (dispersão), 'bars' (barras emparelhadas) ou 'both' (ambos lado a lado)",
    )
    parser.add_argument(
        "--save-poses",
        action="store_true",
        help="Se ativado, salva também as imagens individuais anotadas com o esqueleto na subpasta poses/ da run",
    )

    args = parser.parse_args()

    executar_avaliacao(
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        measure=args.measure,
        inputs_dir=args.inputs_dir,
        gt_file=args.gt,
        out_dir=args.out_dir,
        dist_cm=args.dist,
        save_annotated_images=args.save_poses,
        plot_type=args.plot_type,
    )
