import os
import re
import json
import argparse
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
        if not nome_arq.lower().endswith(VALID_IMAGE_EXTENSIONS):
            continue

        match_id = re.match(r"^(\d+)", nome_arq)
        if not match_id:
            continue

        id_animal = int(match_id.group(1))
        if id_animal in banco_gt:
            caminho_completo = os.path.join(inputs_dir, nome_arq)
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


def gerar_grafico_medida(metrica_info: dict, nome_modelo: str, tag_modelo: str, out_dir: str, plot_type: str = "both") -> str:
    """
    Gera e salva o gráfico comparando Valor Real vs Valor Inferido (Scatter Plot e/ou Barras Emparelhadas),
    incluindo obrigatoriamente no título e na legenda:
      - Nome do modelo utilizado
      - Nome da medida avaliada
      - Porcentagem de acerto final (Acurácia = 100% - MAPE)
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
    measure_key = metrica_info["measure_key"]

    reais = np.array([a["real"] for a in amostras])
    inferidos = np.array([a["inferido"] for a in amostras])
    labels_x = [a["label_curto"] for a in amostras]

    titulo_principal = (
        f"Avaliação Biométrica: {label_medida} ({unidade})\n"
        f"Modelo: {nome_modelo} [{tag_modelo}] | Acurácia Final: {acuracia:.2f}% (MAPE: {mape:.2f}% | MAE: {mae:.2f} {unidade})"
    )
    legenda_identificacao = f"{nome_modelo} | {label_medida} | Acerto: {acuracia:.2f}%"

    if plot_type == "scatter":
        fig, ax_scatter = plt.subplots(figsize=(9, 7), dpi=150)
        axes_list = [ax_scatter]
    elif plot_type == "bars":
        fig, ax_bars = plt.subplots(figsize=(max(12, len(amostras) * 0.45), 7), dpi=150)
        axes_list = [ax_bars]
    else:
        fig, (ax_scatter, ax_bars) = plt.subplots(
            1, 2, figsize=(18, 7.5), dpi=150, gridspec_kw={"width_ratios": [1.0, 1.35]}
        )
        axes_list = [ax_scatter, ax_bars]

    fig.suptitle(titulo_principal, fontsize=13, fontweight="bold", y=0.98)

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
            f"Dispersão: Real vs Inferido — {label_medida}\nModelo: {tag_modelo} | Acerto: {acuracia:.2f}%",
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
            f"Barras Emparelhadas por Imagem — {label_medida}\nModelo: {tag_modelo} | Acerto: {acuracia:.2f}%",
            fontsize=11,
            fontweight="semibold",
        )
        ax_bars.set_xlabel("Imagens Avaliadas (inputs/)", fontsize=10)
        ax_bars.set_ylabel(f"{label_medida} ({unidade})", fontsize=10)
        ax_bars.set_xticks(x_idx)
        ax_bars.set_xticklabels(labels_x, rotation=65, ha="right", fontsize=8)
        ax_bars.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax_bars.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    caminho_grafico = gerar_caminho_unico(out_dir, f"eval_{tag_modelo}_{measure_key}", ".png")
    fig.savefig(caminho_grafico, dpi=150)
    plt.close(fig)

    return caminho_grafico


def gerar_grafico_resumo_geral(resultados_medidas: list[dict], nome_modelo: str, tag_modelo: str, out_dir: str) -> str:
    """Gera um gráfico de resumo com a Porcentagem de Acerto (Acurácia) de todas as medidas avaliadas."""
    validas = [m for m in resultados_medidas if m["acuracia"] is not None]
    if len(validas) < 2:
        return ""

    os.makedirs(out_dir, exist_ok=True)
    nomes = [m["label"] for m in validas]
    acuracias = [m["acuracia"] for m in validas]
    mapes = [m["mape"] for m in validas]
    media_global = float(np.mean(acuracias))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x_idx = np.arange(len(validas))
    cores = ["#2b8cbe" if ac >= 80 else "#fdae6b" if ac >= 65 else "#de2d26" for ac in acuracias]

    barras = ax.bar(
        x_idx,
        acuracias,
        color=cores,
        edgecolor="black",
        width=0.55,
        label=f"Modelo: {nome_modelo} ({tag_modelo}) | Acurácia Média Global: {media_global:.2f}%",
    )

    for bar, ac, mp in zip(barras, acuracias, mapes):
        altura = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            altura + 1.2,
            f"{ac:.1f}%\n(MAPE {mp:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_ylim(0, 112)
    ax.axhline(media_global, color="#d62728", linestyle="--", linewidth=1.8, label=f"Média Global: {media_global:.2f}%")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(nomes, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylabel("Porcentagem de Acerto / Acurácia (%)", fontsize=10)
    ax.set_title(
        f"Resumo Geral de Acurácia por Medida Biométrica\n"
        f"Modelo: {nome_modelo} [{tag_modelo}] | Acerto Médio Global: {media_global:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    caminho_resumo = gerar_caminho_unico(out_dir, f"eval_{tag_modelo}_resumo_todas_medidas", ".png")
    fig.savefig(caminho_resumo, dpi=150)
    plt.close(fig)
    return caminho_resumo


def executar_avaliacao(
    model_path: str,
    conf_thresh: float = 0.25,
    imgsz: int = 1280,
    measure: str = "all",
    inputs_dir: str = "inputs",
    gt_file: str = "dados_biometria.json",
    out_dir: str = "outputs/evaluation",
    dist_cm: float = None,
    save_annotated_images: bool = False,
    plot_type: str = "both",
):
    medidas_selecionadas = resolver_medidas_selecionadas(measure)
    banco_gt = carregar_ground_truth(gt_file)
    imagens_filtradas = filtrar_imagens_com_gabarito(inputs_dir, banco_gt)

    if not imagens_filtradas:
        print(f"Nenhuma imagem na pasta '{inputs_dir}' possui correspondência no arquivo '{gt_file}'.")
        return

    model_resolvido = pipeline.resolver_modelo_padrao(model_path)
    tag_modelo, nome_run_modelo = pipeline.extrair_identificador_modelo(model_resolvido)

    print("=" * 72)
    print("         ONEVISION - AVALIAÇÃO E VALIDAÇÃO DE MODELOS BIOMÉTRICOS       ")
    print("=" * 72)
    print(f" > Modelo Selecionado : {model_resolvido}")
    print(f" > Identificador Run  : {nome_run_modelo} [tag: {tag_modelo}]")
    print(f" > Parâmetros YOLO    : conf={conf_thresh} | imgsz={imgsz}")
    print(f" > Imagens com GT     : {len(imagens_filtradas)} encontradas em '{inputs_dir}/'")
    print(f" > Medidas Avaliadas  : {', '.join(medidas_selecionadas)}")
    print("=" * 72)

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

        if resultado and resultado.get("estimativas"):
            print("OK")
            registros_inferencia.append({
                "image_path": img_path,
                "id_animal": id_animal,
                "estimativas": resultado["estimativas"],
                "gt": gt_item,
            })
        else:
            print("FALHA (animal/keypoints insuficientes)")
            falhas_deteccao.append(nome_img)

    print("\n" + "=" * 72)
    print(f"               RESULTADOS CONSOLIDADOS ({nome_run_modelo})               ")
    print("=" * 72)
    print(f" Imagens processadas com sucesso: {len(registros_inferencia)}/{len(imagens_filtradas)}")
    if falhas_deteccao:
        print(f" Imagens sem detecção válida ({len(falhas_deteccao)}): {', '.join(falhas_deteccao)}")
    print("-" * 72)
    print(f"{'Medida Biométrica':<24} | {'Amostras':<8} | {'MAE':<10} | {'MAPE (%)':<10} | {'Acerto (%)'}")
    print("-" * 72)

    os.makedirs(out_dir, exist_ok=True)
    resultados_finais = []
    graficos_salvos = []

    for m_key in medidas_selecionadas:
        info = calcular_metricas_medida(registros_inferencia, m_key)
        resultados_finais.append(info)

        if info["n_amostras"] > 0:
            und = info["unit"]
            print(
                f"{info['label']:<24} | {info['n_amostras']:<8d} | "
                f"{info['mae']:6.2f} {und:<3} | {info['mape']:8.2f}%  | {info['acuracia']:8.2f}%"
            )
            caminho_fig = gerar_grafico_medida(
                metrica_info=info,
                nome_modelo=nome_run_modelo,
                tag_modelo=tag_modelo,
                out_dir=out_dir,
                plot_type=plot_type,
            )
            graficos_salvos.append(caminho_fig)
        else:
            print(f"{info['label']:<24} | {'0':<8} | {'N/A':<10} | {'N/A':<10} | {'N/A'}")

    print("=" * 72)

    if len(medidas_selecionadas) > 1:
        caminho_resumo = gerar_grafico_resumo_geral(resultados_finais, nome_run_modelo, tag_modelo, out_dir)
        if caminho_resumo:
            graficos_salvos.append(caminho_resumo)

    # Salva também um relatório JSON detalhado na pasta de avaliação sem sobrescrever existentes
    sufixo_medida = "all" if len(medidas_selecionadas) > 1 else medidas_selecionadas[0]
    relatorio_json_path = gerar_caminho_unico(out_dir, f"relatorio_{tag_modelo}_{sufixo_medida}", ".json")
    with open(relatorio_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo_caminho": model_resolvido,
                "modelo_nome": nome_run_modelo,
                "modelo_tag": tag_modelo,
                "parametros": {"conf": conf_thresh, "imgsz": imgsz, "measure": measure},
                "total_imagens_com_gt": len(imagens_filtradas),
                "total_inferencias_validas": len(registros_inferencia),
                "falhas_deteccao": falhas_deteccao,
                "metricas_por_medida": resultados_finais,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nGráficos e relatórios salvos em:")
    for g in graficos_salvos:
        print(f" -> [Gráfico]   {g}")
    print(f" -> [Relatório] {relatorio_json_path}")


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
        help="Pasta onde os gráficos e relatórios de avaliação serão salvos (padrão: outputs/evaluation)",
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
        help="Se ativado, salva também as imagens individuais anotadas com o esqueleto na pasta outputs/",
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
