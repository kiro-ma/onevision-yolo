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

def calcular_distancia(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def estimar_biometria_autonoma(kpts, dist_cm=160.0, img_w=4080.0):
    """Estima as medidas corporais e o peso vivo do caprino a partir dos keypoints."""
    pts = {AP10K_KEYPOINTS[i]: (kpts[i][0], kpts[i][1], kpts[i][2]) for i in range(len(kpts))}
    min_conf = 0.35

    focal_ajustado = FOCAL_CALIBRATION_CONSTANT * (img_w / BASE_IMAGE_WIDTH)
    px_per_cm = focal_ajustado / dist_cm

    front_paws = [pts["left_front_paw"], pts["right_front_paw"]]
    front_valid = [p for p in front_paws if p[2] >= min_conf]
    pata_diant_y = max([p[1] for p in front_valid]) if front_valid else None

    back_paws = [pts["left_back_paw"], pts["right_back_paw"]]
    back_valid = [p for p in back_paws if p[2] >= min_conf]
    pata_tras_y = max([p[1] for p in back_valid]) if back_valid else None

    estimativas = {}

    # 1. Altura da Cernelha
    if pts["neck"][2] >= min_conf and pata_diant_y is not None:
        cernelha_px = abs(pata_diant_y - pts["neck"][1])
        estimativas["altura_cernelha_cm"] = cernelha_px / px_per_cm
        estimativas["cernelha_px"] = cernelha_px

    # 2. Altura da Garupa
    if pts["root_of_tail"][2] >= min_conf and pata_tras_y is not None:
        garupa_px = abs(pata_tras_y - pts["root_of_tail"][1])
        estimativas["altura_garupa_cm"] = garupa_px / px_per_cm
        estimativas["garupa_px"] = garupa_px

    # 3. Comprimento Corporal
    if pts["neck"][2] >= min_conf and pts["root_of_tail"][2] >= min_conf:
        comp_px = calcular_distancia(pts["neck"][:2], pts["root_of_tail"][:2])
        comp_cm = (comp_px / px_per_cm) * 0.85
        estimativas["comprimento_cm"] = comp_cm
        estimativas["comprimento_px"] = comp_px

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

    # 6. Predição Autônoma do Peso Vivo (kg)
    peso_kg = (pow(perimetro_torax_cm, 2) * comprimento_cm) / 10800.0
    estimativas["peso_estimado_kg"] = peso_kg

    return estimativas

def desenhar_painel_onevision(img, kpts, estimativas, dados_validacao=None):
    """Desenha o esqueleto e o painel biométrico com validação completa real vs estimado."""
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
        (f"PESO ESTIMADO: {estimativas.get('peso_estimado_kg', 0):.1f} kg", (0, 255, 0)),
        ("----------------------------------------", (150, 150, 150)),
        (f"Alt. Cernelha:   {estimativas.get('altura_cernelha_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Alt. Garupa:     {estimativas.get('altura_garupa_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Comprimento:     {estimativas.get('comprimento_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Circ. Toracica:  {estimativas.get('circ_torax_cm', 0):.1f} cm", (255, 255, 255)),
        (f"Gross. Pescoco:  {estimativas.get('grossura_pescoco_cm', 0):.1f} cm", (255, 255, 255)),
    ]

    # Adiciona a validação comparativa se houver dados de teste
    if dados_validacao:
        linhas.append(("--- VALIDAÇÃO DE TESTE (GABARITO) ---", (0, 165, 255)))
        
        metricas = [
            ("Peso", estimativas.get("peso_estimado_kg"), extrair_valor_gabarito(dados_validacao, "peso_kg", "peso", "peso_real_kg"), "kg"),
            ("Cernelha", estimativas.get("altura_cernelha_cm"), extrair_valor_gabarito(dados_validacao, "altura_cernelha_cm", "cernelha_cm", "altura_cernelha"), "cm"),
            ("Garupa", estimativas.get("altura_garupa_cm"), extrair_valor_gabarito(dados_validacao, "altura_garupa_cm", "garupa_cm", "altura_garupa"), "cm"),
            ("Comprimento", estimativas.get("comprimento_cm"), extrair_valor_gabarito(dados_validacao, "comprimento_cm", "comprimento", "comprimento_corpo_cm"), "cm"),
            ("Circ.Torax", estimativas.get("circ_torax_cm"), extrair_valor_gabarito(dados_validacao, "circ_torax_cm", "perimetro_torax_cm", "perimetro_toracico_cm"), "cm"),
            ("Gross.Pesc", estimativas.get("grossura_pescoco_cm"), extrair_valor_gabarito(dados_validacao, "grossura_pescoco_cm", "pescoco_cm"), "cm"),
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

def executar_inferencia(image_path, model_path="./runs/pose/onevision-yolo/treino_base_ap10k-4/weights/best.pt",
                        dist_cm=None, conf_thresh=0.25, ground_truth_file=None):
    if not os.path.exists(image_path):
        print(f"ERRO: Imagem '{image_path}' não encontrada.")
        return

    distancia = dist_cm if dist_cm is not None else deduzir_distancia_camera(image_path)

    print(f"\nCarregando modelo YOLO Pose: {model_path}...")
    model = YOLO(model_path)

    print(f"Processando imagem: {image_path} (Distância da câmera: {distancia:.0f} cm)...")
    results = model(image_path, conf=conf_thresh, verbose=False)

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
                ("Perímetro Torácico", estimativas.get("circ_torax_cm"), extrair_valor_gabarito(dados_validacao, "circ_torax_cm", "perimetro_torax_cm", "perimetro_toracico_cm"), "cm"),
                ("Grossura do Pescoço", estimativas.get("grossura_pescoco_cm"), extrair_valor_gabarito(dados_validacao, "grossura_pescoco_cm", "pescoco_cm"), "cm"),
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

        img_anotada = desenhar_painel_onevision(img_anotada, kpts, estimativas, dados_validacao)

    os.makedirs("outputs", exist_ok=True)
    caminho_saida = os.path.join("outputs", f"pose_{os.path.basename(image_path)}")
    cv2.imwrite(caminho_saida, img_anotada)
    print(f"\nImagem com esqueleto e biometria salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneVision - Estimativa Autônoma de Pose, Biometria e Peso de Caprinos")
    parser.add_argument("--image", type=str, default="inputs/010-dist160-v3.jpg", help="Caminho da imagem do animal")
    parser.add_argument("--model", type=str, default="./runs/pose/onevision-yolo/caprine-bovidae-v1/weights/best.pt", help="Pesos do modelo treinado")
    parser.add_argument("--dist", type=float, default=None, help="Distância da câmera ao animal em cm (se omitida, deduz do arquivo ou usa 160 cm)")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confiança")
    parser.add_argument("--test-gt", type=str, default=None, help="Caminho opcional do JSON de gabarito apenas para verificação de teste")
    args = parser.parse_args()

    executar_inferencia(args.image, args.model, args.dist, args.conf, args.test_gt)