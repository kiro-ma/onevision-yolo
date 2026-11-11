import os
import cv2
import json
import re
import numpy as np
import argparse
from ultralytics import YOLO

# Mapeamento oficial dos 17 Keypoints do AP-10K
AP10K_KEYPOINTS = [
    "left_eye",        # 0
    "right_eye",       # 1
    "nose",            # 2
    "neck",            # 3 (região cernelha/pescoço)
    "root_of_tail",    # 4 (região garupa/cauda)
    "left_shoulder",   # 5
    "left_elbow",      # 6
    "left_front_paw",  # 7
    "right_shoulder",  # 8
    "right_elbow",     # 9
    "right_front_paw", # 10
    "left_hip",        # 11
    "left_knee",       # 12
    "left_back_paw",   # 13
    "right_hip",       # 14
    "right_knee",      # 15
    "right_back_paw"   # 16
]

# Esqueleto anatômico quadrupedal com agrupamento por região
SKELETON_GROUPS = [
    # Cabeça (Ciano)
    {"color": (255, 255, 0), "pairs": [(0, 1), (0, 2), (1, 2), (2, 3)]},
    # Coluna / Tronco (Amarelo)
    {"color": (0, 215, 255), "pairs": [(3, 4)]},
    # Membros Anteriores / Dianteiros (Verde)
    {"color": (0, 255, 0), "pairs": [(3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10)]},
    # Membros Posteriores / Traseiros (Magenta)
    {"color": (255, 0, 255), "pairs": [(4, 11), (11, 12), (12, 13), (4, 14), (14, 15), (15, 16)]}
]

# Constante de calibração óptica da câmera (pixels * cm_distância / cm_real)
FOCAL_CALIBRATION_CONSTANT = 3350.0
BASE_IMAGE_WIDTH = 4080.0

def extrair_valor_gabarito(dados, *chaves):
    """Auxiliar para extrair valores do JSON de validação aceitando variações de nomenclatura."""
    if not dados:
        return None
    for chave in chaves:
        if chave in dados and dados[chave] is not None:
            try:
                return float(dados[chave])
            except (ValueError, TypeError):
                continue
    return None

def deduzir_distancia_camera(caminho_imagem, dist_padrao=160.0):
    """Deduz a distância da câmera em centímetros se constar no nome do arquivo (ex: 'dist160')."""
    match = re.search(r"dist(\d+)", os.path.basename(caminho_imagem))
    if match:
        return float(match.group(1))
    return dist_padrao


def extrair_identificador_modelo(model_path: str) -> tuple[str, str]:
    """
    Extrai o identificador curto do modelo (o que vem depois de 'vX-', ou o próprio 'vX'
    se não houver sufixo) e o nome completo da pasta do experimento.

    Exemplos:
      'runs/pose/onevision-yolo/caprine-bovidae-v2-oclusao/weights/best.pt' -> ('oclusao', 'caprine-bovidae-v2-oclusao')
      'runs/pose/onevision-yolo/caprine-bovidae-v1/weights/best.pt'         -> ('v1', 'caprine-bovidae-v1')
      'runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt' -> ('strict-clean', 'caprine-bovidae-v4-strict-clean')
    """
    caminho_norm = os.path.normpath(model_path)
    partes = caminho_norm.split(os.sep)

    # Se o arquivo está dentro de '.../<nome_do_run>/weights/best.pt', pega '<nome_do_run>'
    if len(partes) >= 3 and partes[-2].lower() == "weights":
        nome_run = partes[-3]
    else:
        nome_run = os.path.splitext(os.path.basename(caminho_norm))[0]

    match = re.search(r"(v\d+)(?:-(.+))?$", nome_run, flags=re.IGNORECASE)
    if match:
        versao, sufixo = match.group(1), match.group(2)
        tag_curta = sufixo if sufixo else versao
    else:
        tag_curta = nome_run

    return tag_curta, nome_run


def gerar_caminho_saida_incremental(image_path: str, tag_modelo: str, out_dir: str = "outputs") -> str:
    """
    Gera o caminho de saída mantendo o nome original da imagem + '_{tag_modelo}-{n}',
    incrementando 'n' (1, 2, 3...) automaticamente se o arquivo já existir.
    """
    os.makedirs(out_dir, exist_ok=True)
    nome_base, ext = os.path.splitext(os.path.basename(image_path))
    if not ext:
        ext = ".jpg"

    contador = 1
    while True:
        nome_arquivo = f"pose_{nome_base}_{tag_modelo}-{contador}{ext}"
        caminho_completo = os.path.join(out_dir, nome_arquivo)
        if not os.path.exists(caminho_completo):
            return caminho_completo
        contador += 1

def calcular_distancia(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def calcular_peso_schaeffer_kg(comprimento_cm: float, circ_torax_cm: float) -> float:
    """
    Calcula o peso vivo em kg usando a fórmula de Schaeffer adaptada para o sistema métrico (cm):
    Original: W (lbs) = (BL_in * CG_in^2) / 300
    Conversões:
      1 polegada (in) = 2.54 cm
      1 kg = 2.20462 lbs
    Equivalente direto em cm: W (kg) = (BL_cm * CG_cm^2) / 10838.18
    """
    bl_in = comprimento_cm / 2.54
    cg_in = circ_torax_cm / 2.54
    peso_lbs = (bl_in * (cg_in ** 2)) / 300.0
    return peso_lbs / 2.20462


def _avaliar_cadeia_membro(proximal, medial, distal, min_conf=0.35):
    """
    Avalia a consistência biomecânica de um membro (ombro/quadril -> cotovelo/joelho -> pata).
    Retorna um dicionário com métricas de qualidade, verticalidade da canela e estimativa do solo.
    """
    if distal[2] < min_conf:
        return {"valido": False, "score": 0.0, "y_apoio": None}

    # Confiança ponderada dando maior peso à extremidade (pata) e articulação média
    conf_chain = (0.20 * proximal[2]) + (0.30 * medial[2]) + (0.50 * distal[2])

    # Verifica progressão gravitacional (y cresce de cima para baixo na imagem)
    ordem_gravidade = (proximal[1] < medial[1] < distal[1]) if (proximal[2] >= min_conf and medial[2] >= min_conf) else True

    # Vetor da canela (cotovelo/joelho -> pata)
    if medial[2] >= min_conf:
        dx_canela = abs(distal[0] - medial[0])
        dy_canela = max(1.0, distal[1] - medial[1])
        comp_canela = float(np.hypot(dx_canela, dy_canela))
        # Cosseno do ângulo com a vertical (1.0 = perfeitamente vertical; cai se a linha cruzar para outra pata)
        verticalidade = dy_canela / comp_canela if distal[1] > medial[1] else 0.2
    else:
        comp_canela = 0.0
        verticalidade = 0.7

    # Vetor do segmento superior (ombro/quadril -> cotovelo/joelho)
    if proximal[2] >= min_conf and medial[2] >= min_conf:
        comp_sup = calcular_distancia(proximal[:2], medial[:2])
    else:
        comp_sup = comp_canela

    # Penaliza se a ordem gravitacional estiver invertida ou se a canela estiver muito oblíqua (cruzamento E/D)
    penalidade_geometria = 1.0 if (ordem_gravidade and verticalidade >= 0.82) else 0.55
    score = conf_chain * verticalidade * penalidade_geometria

    return {
        "valido": True,
        "score": float(score),
        "verticalidade": float(verticalidade),
        "y_apoio": float(distal[1]),
        "x_apoio": float(distal[0]),
        "x_medial": float(medial[0]) if medial[2] >= min_conf else float(distal[0]),
        "comp_canela": float(comp_canela),
        "comp_sup": float(comp_sup),
    }


def selecionar_apoio_biomecanico(prox_l, med_l, dist_l, prox_r, med_r, dist_r, comp_tronco_px=None, min_conf=0.35):
    """
    Filtro biomecânico inteligente para selecionar a coordenada Y de apoio no solo:
    1. Avalia a cadeia articular Esquerda e Direita (confiança + verticalidade + gravidade).
    2. Detecta se houve colapso espacial (ambos os keypoints previstos na mesma perna ou no vazio entre elas)
       ou cruzamento em 'X' entre a perna esquerda e direita.
    3. Compensa pernas "curtas" (quando o keypoint para no boleto em vez do casco por colapso bimodal).
    """
    av_l = _avaliar_cadeia_membro(prox_l, med_l, dist_l, min_conf=min_conf)
    av_r = _avaliar_cadeia_membro(prox_r, med_r, dist_r, min_conf=min_conf)

    if not av_l["valido"] and not av_r["valido"]:
        return None, {"colapsado": True, "score": 0.0}
    if av_l["valido"] and not av_r["valido"]:
        return av_l["y_apoio"], {"colapsado": True, "score": av_l["score"], "eleito": av_l}
    if av_r["valido"] and not av_l["valido"]:
        return av_r["y_apoio"], {"colapsado": True, "score": av_r["score"], "eleito": av_r}

    # Ambos os lados são válidos: verifica se houve colapso espacial ou cruzamento em X
    sep_patas_x = abs(av_l["x_apoio"] - av_r["x_apoio"])
    sep_medial_x = abs(av_l["x_medial"] - av_r["x_medial"])
    limiar_colapso_x = (comp_tronco_px * 0.045) if comp_tronco_px else 40.0

    # Cruzamento em X: o lado que estava à esquerda no joelho/cotovelo passa para a direita na pata
    cruzamento_x = ((av_l["x_medial"] - av_r["x_medial"]) * (av_l["x_apoio"] - av_r["x_apoio"])) < 0
    colapsado = (sep_patas_x < limiar_colapso_x) or (sep_medial_x < limiar_colapso_x * 0.6) or cruzamento_x

    melhor = av_l if av_l["score"] >= av_r["score"] else av_r
    outro = av_r if melhor is av_l else av_l

    # Se um membro tem score biomecânico substancialmente superior (>15%), ignora o membro duvidoso/cruzado
    if melhor["score"] > outro["score"] * 1.15:
        y_eleito = melhor["y_apoio"]
    elif colapsado:
        # No colapso bimodal (regressão à média), o ponto mais baixo que mantém verticalidade representa melhor o solo,
        # acrescido de leve compensação anatômica caso a canela tenha encolhido em relação ao segmento superior
        y_base = max(av_l["y_apoio"], av_r["y_apoio"])
        deficit_canela = max(0.0, (melhor["comp_sup"] * 1.05) - melhor["comp_canela"])
        y_eleito = y_base + min(deficit_canela * 0.35, melhor["comp_sup"] * 0.12)
    else:
        # Membros bem separados e ambos verticalmente consistentes:
        # Pondera entre a pata mais próxima (maior Y) e a média para evitar viés de perspectiva de passo aberto
        y_max = max(av_l["y_apoio"], av_r["y_apoio"])
        y_min = min(av_l["y_apoio"], av_r["y_apoio"])
        y_eleito = (0.70 * y_max) + (0.30 * y_min)

    return float(y_eleito), {"colapsado": colapsado, "score": melhor["score"], "eleito": melhor}


def estimar_biometria_autonoma(kpts, dist_cm=160.0, img_w=4080.0):
    """Estima as medidas corporais e o peso vivo do caprino a partir dos keypoints com filtro biomecânico."""
    pts = {AP10K_KEYPOINTS[i]: (kpts[i][0], kpts[i][1], kpts[i][2]) for i in range(len(kpts))}
    min_conf = 0.35

    focal_ajustado = FOCAL_CALIBRATION_CONSTANT * (img_w / BASE_IMAGE_WIDTH)
    px_per_cm = focal_ajustado / dist_cm

    comp_tronco_px = None
    if pts["neck"][2] >= min_conf and pts["root_of_tail"][2] >= min_conf:
        comp_tronco_px = calcular_distancia(pts["neck"][:2], pts["root_of_tail"][:2])

    # Seleção biomecânica do apoio dianteiro e traseiro
    pata_diant_y, info_diant = selecionar_apoio_biomecanico(
        pts["left_shoulder"], pts["left_elbow"], pts["left_front_paw"],
        pts["right_shoulder"], pts["right_elbow"], pts["right_front_paw"],
        comp_tronco_px=comp_tronco_px, min_conf=min_conf
    )
    pata_tras_y, info_tras = selecionar_apoio_biomecanico(
        pts["left_hip"], pts["left_knee"], pts["left_back_paw"],
        pts["right_hip"], pts["right_knee"], pts["right_back_paw"],
        comp_tronco_px=comp_tronco_px, min_conf=min_conf
    )

    # Reconciliação de plano de solo: se um par (ex: traseiro) sofreu colapso e ficou flutuando muito acima
    # ou abaixo do par íntegro (ex: dianteiro), regulariza a linha de solo usando o par mais confiável
    if pata_diant_y is not None and pata_tras_y is not None and comp_tronco_px is not None:
        desnivel_max_natural = comp_tronco_px * 0.06
        diff_solo = pata_tras_y - pata_diant_y
        if info_tras["colapsado"] and not info_diant["colapsado"] and abs(diff_solo) > desnivel_max_natural:
            pata_tras_y = (0.45 * pata_tras_y) + (0.55 * pata_diant_y)
        elif info_diant["colapsado"] and not info_tras["colapsado"] and abs(diff_solo) > desnivel_max_natural:
            pata_diant_y = (0.45 * pata_diant_y) + (0.55 * pata_tras_y)

    estimativas = {}

    # 1. Altura da Cernelha
    if pts["neck"][2] >= min_conf and pata_diant_y is not None:
        cernelha_px = abs(pata_diant_y - pts["neck"][1])
        estimativas["altura_cernelha_cm"] = cernelha_px / px_per_cm
        estimativas["cernelha_px"] = cernelha_px

    # 2. Altura da Garupa (compensa a diferença anatômica entre a inserção alta da cauda no AP-10K e o plano sacral)
    if pts["root_of_tail"][2] >= min_conf and pata_tras_y is not None:
        # Usa a referência sacral interpolada com o quadril visível para evitar superestimar quando a cauda é alta
        hips_valid = [pts["left_hip"][1] for p in [pts["left_hip"]] if p[2] >= min_conf] + \
                     [pts["right_hip"][1] for p in [pts["right_hip"]] if p[2] >= min_conf]
        if hips_valid:
            y_garupa_ref = (0.85 * pts["root_of_tail"][1]) + (0.15 * float(np.mean(hips_valid)))
        else:
            y_garupa_ref = pts["root_of_tail"][1]
        garupa_px = abs(pata_tras_y - y_garupa_ref)
        estimativas["altura_garupa_cm"] = garupa_px / px_per_cm
        estimativas["garupa_px"] = garupa_px

    # 3. Comprimento Corporal
    if comp_tronco_px is not None:
        comp_cm = (comp_tronco_px / px_per_cm) * 0.85
        estimativas["comprimento_cm"] = comp_cm
        estimativas["comprimento_px"] = comp_tronco_px

    # 4. Comprimento e Grossura do Pescoço
    if pts["nose"][2] >= min_conf and pts["neck"][2] >= min_conf:
        comp_cabeca_px = calcular_distancia(pts["nose"][:2], pts["neck"][:2])
        comp_cabeca_cm = comp_cabeca_px / px_per_cm
        estimativas["comprimento_cabeca_cm"] = comp_cabeca_cm
        estimativas["grossura_pescoco_cm"] = comp_cabeca_cm * 0.72

    # 5. Perímetro Torácico
    cernelha_cm = estimativas.get("altura_cernelha_cm", 74.0)
    comprimento_cm = estimativas.get("comprimento_cm", 70.0)
    perimetro_torax_cm = cernelha_cm * 1.275
    estimativas["circ_torax_cm"] = perimetro_torax_cm

    # 6. Predição Autônoma do Peso Vivo (kg) via Equação de Schaeffer (métrica)
    peso_kg = calcular_peso_schaeffer_kg(comprimento_cm, perimetro_torax_cm)
    estimativas["peso_estimado_kg"] = peso_kg

    return estimativas

def desenhar_painel_onevision(img, kpts, estimativas, dados_validacao=None, nome_modelo=None):
    """Desenha o esqueleto e o painel biométrico com validação completa real vs estimado e identificação do modelo."""
    h, w = img.shape[:2]
    escala = max(w, h) / 2000.0
    line_thick = max(2, int(4 * escala))
    radius = max(4, int(7 * escala))
    font_scale = 0.50 * escala

    # 1. Desenha as conexões do esqueleto
    for grupo in SKELETON_GROUPS:
        cor = grupo["color"]
        for p1, p2 in grupo["pairs"]:
            x1, y1, c1 = kpts[p1]
            x2, y2, c2 = kpts[p2]
            if c1 > 0.35 and c2 > 0.35:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), cor, line_thick, cv2.LINE_AA)

    # 2. Desenha nós dos keypoints
    for i, (x, y, conf) in enumerate(kpts):
        if conf > 0.35:
            cv2.circle(img, (int(x), int(y)), radius, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius + max(1, int(2 * escala)), (255, 255, 255), 1, cv2.LINE_AA)
            nome_ponto = AP10K_KEYPOINTS[i]
            cv2.putText(
                img, f"{nome_ponto}", (int(x) + int(10 * escala), int(y) - int(5 * escala)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (255, 255, 255), max(1, int(1.2 * escala)), cv2.LINE_AA
            )

    # 3. Prepara as linhas de texto do painel
    linhas = [
        ("=== ONEVISION: BIOMETRIA IA ===", (0, 255, 255)),
    ]
    if nome_modelo:
        linhas.append((f"MODELO: {nome_modelo}", (180, 230, 255)))

    linhas.extend([
        (f"PESO ESTIMADO: {estimativas.get('peso_estimado_kg', 0):.1f} kg", (0, 255, 0)),
        ("----------------------------------------", (150, 150, 150)),
        (f"Alt. Cernelha:   {estimativas.get('altura_cernelha_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Alt. Garupa:     {estimativas.get('altura_garupa_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Comprimento:     {estimativas.get('comprimento_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Circ. Toracica:  {estimativas.get('circ_torax_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Gross. Pescoco:  {estimativas.get('grossura_pescoco_cm', 0):.1f} cm", (255, 255, 255)),
    ])

    # Adiciona a validação comparativa se houver dados de teste
    if dados_validacao:
        linhas.append(("--- VALIDAÇÃO DE TESTE (GABARITO) ---", (0, 165, 255)))
        
        metricas = [
            ("Peso", estimativas.get("peso_estimado_kg"), extrair_valor_gabarito(dados_validacao, "peso_kg", "peso", "peso_real_kg"), "kg"),
            ("Cernelha", estimativas.get("altura_cernelha_cm"), extrair_valor_gabarito(dados_validacao, "altura_cernelha_cm", "cernelha_cm", "altura_cernelha"), "cm"),
            ("Garupa", estimativas.get("altura_garupa_cm"), extrair_valor_gabarito(dados_validacao, "altura_garupa_cm", "garupa_cm", "altura_garupa"), "cm"),
            ("Comprimento", estimativas.get("comprimento_cm"), extrair_valor_gabarito(dados_validacao, "comprimento_cm", "comprimento", "comprimento_corpo_cm"), "cm"),
            ("Circ.Torax", estimativas.get("circ_torax_cm"), extrair_valor_gabarito(dados_validacao, "circ_do_torax", "circ_torax_cm", "perimetro_torax_cm", "perimetro_toracico_cm"), "cm"),
            ("Gross.Pesc", estimativas.get("grossura_pescoco_cm"), extrair_valor_gabarito(dados_validacao, "pescoco_circ_medio", "grossura_pescoco_cm", "pescoco_cm"), "cm"),
        ]

        for label, est_v, real_v, und in metricas:
            if real_v is not None and est_v is not None:
                diff = abs(est_v - real_v)
                linhas.append((f"{label}: Real {real_v:.1f}{und} | Diff {diff:.1f}{und}", (200, 220, 255)))

    # Dimensionamento dinâmico da caixa do painel com base na quantidade de linhas
    painel_w = int(520 * escala)
    espaco_linha = int(28 * escala)
    painel_h = int((35 * escala) + (len(linhas) * espaco_linha))
    pad = int(25 * escala)

    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + painel_w, pad + painel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.rectangle(img, (pad, pad), (pad + painel_w, pad + painel_h), (0, 200, 255), max(1, int(2 * escala)))

    y_txt = pad + int(30 * escala)
    for texto, cor in linhas:
        cv2.putText(img, texto, (pad + int(15 * escala), y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, cor, max(1, int(1.3 * escala)), cv2.LINE_AA)
        y_txt += espaco_linha

    return img

def resolver_modelo_padrao(caminho_preferido="./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt"):
    """Retorna o modelo preferido (v4) ou faz fallback automático para o melhor checkpoint finalizado disponível."""
    candidatos = [
        caminho_preferido,
        "./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt",
        "./runs/pose/onevision-yolo/caprine-bovidae-v2-oclusao/weights/best.pt",
        "./runs/pose/onevision-yolo/caprine-bovidae-v1/weights/best.pt",
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    return caminho_preferido


def executar_inferencia(image_path, model_path="./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt",
                        dist_cm=None, conf_thresh=0.25, ground_truth_file=None, imgsz=1280):
    if not os.path.exists(image_path):
        print(f"ERRO: Imagem '{image_path}' não encontrada.")
        return

    model_path = resolver_modelo_padrao(model_path)
    tag_modelo, nome_run_modelo = extrair_identificador_modelo(model_path)
    distancia = dist_cm if dist_cm is not None else deduzir_distancia_camera(image_path)

    print(f"\nCarregando modelo YOLO Pose: {model_path} [tag: {tag_modelo}]...")
    model = YOLO(model_path)

    print(f"Processando imagem: {image_path} (Distância: {distancia:.0f} cm | imgsz={imgsz})...")
    results = model(image_path, conf=conf_thresh, imgsz=imgsz, verbose=False)

    if not results or len(results[0]) == 0 or results[0].keypoints is None:
        print("Nenhum animal detectado na imagem.")
        return

    result = results[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERRO: Não foi possível carregar a imagem '{image_path}'.")
        return
    h, w = img.shape[:2]
    kpts_data = result.keypoints.data.cpu().numpy()

    dados_validacao = None
    if ground_truth_file and os.path.exists(ground_truth_file):
        try:
            with open(ground_truth_file, "r", encoding="utf-8") as f:
                banco = {item["id_array"]: item for item in json.load(f)}
                match_id = re.match(r"^(\d+)", os.path.basename(image_path))
                if match_id:
                    dados_validacao = banco.get(int(match_id.group(1)))
        except Exception as e:
            print(f"Aviso ao ler teste: {e}")

    detecoes_validas = []
    for kpts in kpts_data:
        confs = [p[2] for p in kpts]
        pts_bons = sum(1 for c in confs if c >= 0.35)
        if pts_bons >= 6:
            detecoes_validas.append((pts_bons, np.mean(confs), kpts))

    if not detecoes_validas:
        print("Aviso: Nenhum animal com confiança suficiente nos keypoints foi detectado.")
        return

    detecoes_validas.sort(key=lambda x: (x[0], x[1]), reverse=True)

    img_anotada = img.copy()
    for _, _, kpts in detecoes_validas:
        estimativas = estimar_biometria_autonoma(kpts, dist_cm=distancia, img_w=w)

        print("\n==============================================")
        print("     PREVISÕES BIOMÉTRICAS AUTÔNOMAS DA IA    ")
        print("==============================================")
        print(f" > PESO VIVO PREVISTO:      {estimativas.get('peso_estimado_kg', 0):.2f} kg")
        print(f" > Altura da Cernelha:      {estimativas.get('altura_cernelha_cm', 0):.1f} cm")
        print(f" > Altura da Garupa:        {estimativas.get('altura_garupa_cm', 0):.1f} cm")
        print(f" > Comprimento Corporal:    {estimativas.get('comprimento_cm', 0):.1f} cm")
        print(f" > Perímetro Torácico:      {estimativas.get('circ_torax_cm', 0):.1f} cm")
        print(f" > Grossura do Pescoço:     {estimativas.get('grossura_pescoco_cm', 0):.1f} cm")
        print("==============================================")

        if dados_validacao:
            metricas_validacao = [
                ("Peso Vivo", estimativas.get("peso_estimado_kg"), extrair_valor_gabarito(dados_validacao, "peso_kg", "peso", "peso_real_kg"), "kg"),
                ("Altura Cernelha", estimativas.get("altura_cernelha_cm"), extrair_valor_gabarito(dados_validacao, "altura_cernelha_cm", "cernelha_cm", "altura_cernelha"), "cm"),
                ("Altura Garupa", estimativas.get("altura_garupa_cm"), extrair_valor_gabarito(dados_validacao, "altura_garupa_cm", "garupa_cm", "altura_garupa"), "cm"),
                ("Comprimento Corporal", estimativas.get("comprimento_cm"), extrair_valor_gabarito(dados_validacao, "comprimento_cm", "comprimento", "comprimento_corpo_cm"), "cm"),
                ("Perímetro Torácico", estimativas.get("circ_torax_cm"), extrair_valor_gabarito(dados_validacao, "circ_do_torax", "circ_torax_cm", "perimetro_torax_cm", "perimetro_toracico_cm"), "cm"),
                ("Grossura do Pescoço", estimativas.get("grossura_pescoco_cm"), extrair_valor_gabarito(dados_validacao, "pescoco_circ_medio", "grossura_pescoco_cm", "pescoco_cm"), "cm"),
            ]

            print("\n----------------------------------------------")
            print("         RELATÓRIO DE VALIDAÇÃO DE TESTE      ")
            print("----------------------------------------------")
            for nome, est_v, real_v, und in metricas_validacao:
                if real_v is not None and est_v is not None:
                    diff = abs(est_v - real_v)
                    pct_err = (diff / real_v) * 100 if real_v > 0 else 0
                    print(f" > {nome:<20}: Est = {est_v:5.1f} {und} | Real = {real_v:5.1f} {und} | Erro = {diff:4.1f} {und} ({pct_err:4.1f}%)")
                elif est_v is not None:
                    print(f" > {nome:<20}: Est = {est_v:5.1f} {und} | Real =  N/A")
            print("----------------------------------------------")

        img_anotada = desenhar_painel_onevision(
            img_anotada, kpts, estimativas, dados_validacao, nome_modelo=f"{nome_run_modelo} ({tag_modelo})"
        )

    caminho_saida = gerar_caminho_saida_incremental(image_path, tag_modelo, out_dir="outputs")
    cv2.imwrite(caminho_saida, img_anotada)
    print(f"\nImagem com esqueleto e biometria salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneVision - Estimativa Autônoma de Pose, Biometria e Peso de Caprinos")
    parser.add_argument("--image", type=str, default="inputs/010-dist160-v3.jpg", help="Caminho da imagem do animal")
    parser.add_argument("--model", type=str, default="./runs/pose/onevision-yolo/caprine-bovidae-v4-strict-clean/weights/best.pt", help="Pesos do modelo treinado")
    parser.add_argument("--dist", type=float, default=None, help="Distância da câmera ao animal em cm (se omitida, deduz do arquivo ou usa 160 cm)")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confiança")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolução de inferência (padrão: 1280 para preservar precisão nas patas)")
    parser.add_argument("--test-gt", type=str, default=None, help="Caminho opcional do JSON de gabarito apenas para verificação de teste")
    args = parser.parse_args()

    executar_inferencia(args.image, args.model, args.dist, args.conf, args.test_gt, imgsz=args.imgsz)