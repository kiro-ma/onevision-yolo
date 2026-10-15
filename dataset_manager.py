import json
import os
import glob

def carregar_dados_e_imagens():
    # 1. Carrega o arquivo JSON
    with open('dados_biometria.json', 'r', encoding='utf-8') as f:
        dados_reais = json.load(f)

    # Cria um dicionário para busca rápida pelo ID (1 a 14)
    banco_de_dados = {animal["id_array"]: animal for animal in dados_reais}

    # 2. Varre a pasta inputs em busca das fotos
    # Assumindo que o script roda na raiz e a pasta inputs existe
    caminho_imagens = os.path.join("inputs", "*.jpg")
    fotos = glob.glob(caminho_imagens)

    animais_processados = 0

    print(f"Encontradas {len(fotos)} fotos. Cruzando dados...\n")

    for foto in fotos:
        nome_arquivo = os.path.basename(foto)
        
        # Ignora as fotos de teste que não tem o padrão numérico
        if nome_arquivo in ['bode-preto.jpg', 'dog.jpg', 'man.jpg']:
            continue
            
        # Extrai os 3 primeiros caracteres numéricos do nome do arquivo (ex: "006" -> 6)
        try:
            id_animal = int(nome_arquivo[:3])
        except ValueError:
            continue
            
        # Busca no banco de dados JSON o peso correspondente a esse ID
        dados_do_animal = banco_de_dados.get(id_animal)
        
        if dados_do_animal:
            peso = dados_do_animal["peso_kg"]
            nome = dados_do_animal["identificacao"]
            print(f"Foto: {nome_arquivo} | ID: {id_animal} | Nome: {nome} | Peso Real: {peso}kg")
            animais_processados += 1
            
            # FUTURO: Aqui é onde você passaria a variável 'foto' para o YOLOv11 processar,
            # extrair os pontos anatômicos e usar como features para uma Regressão!

    print(f"\nCruzamento concluído. {animais_processados} imagens mapeadas com sucesso.")

if __name__ == "__main__":
    carregar_dados_e_imagens()