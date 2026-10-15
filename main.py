from ultralytics import YOLO
import os

# app.roboflow.com

def run_poc():
    print("Carregando o modelo YOLOv11 Pose...")
    model = YOLO("./runs/pose/onevision-yolo/treino_base_ap10k-4/weights/best.pt")

    image_path = "./inputs/010-dist160-v3.jpg"
    
    # VALIDAÇÃO: Verifica se o arquivo realmente existe antes de rodar a IA
    if not os.path.exists(image_path):
        print(f"ERRO: O arquivo '{image_path}' não foi encontrado na pasta do projeto!")
        print("Verifique o nome e a extensão do arquivo.")
        return

    # Garante que a pasta outputs exista no projeto
    os.makedirs("outputs", exist_ok=True)

    print(f"Executando a inferência na imagem: {image_path}...")
    
    # ALTERAÇÃO AQUI: Força o salvamento na pasta ./outputs
    results = model(
        image_path, 
        save=True, 
        project="outputs",  # Define a pasta raiz de destino
        name="",            # Deixa o nome da subpasta vazio para salvar direto na raiz do project
        exist_ok=True       # Sobrescreve resultados antigos em vez de criar novas pastas
    )
    
    # Prevenção extra de erro
    if not results:
        print("Erro interno: A IA não retornou resultados.")
        return

    result = results[0]
    
    print("--- Resultados da Inferência ---")
    if len(result) > 0 and result.keypoints is not None:
        pontos = result.keypoints.xy[0]
        print(f"Foram detectados {len(pontos)} pontos anatômicos.")
        # Mostra o caminho exato onde a imagem anotada foi salva
        caminho_salvo = os.path.join(result.save_dir, os.path.basename(image_path))
        print(f"Imagem com esqueleto salva em: {caminho_salvo}")
    else:
        print("Nenhum ponto de pose detectado. O modelo não reconhece o esqueleto nesta imagem.")

if __name__ == "__main__":
    run_poc()