from ultralytics import YOLO

def treinar_bovidae_finetune():
    """
    Fine-tuning do AP-10K especializado em Bovidae (antelope, sheep, bison, buffalo, cow).
    Partindo do checkpoint do treino geral do AP-10K, o modelo já reconhece
    quadrúpedes. Aqui ele é especializado na morfologia Bovidae/Capridae.

    Melhorias aplicadas sobre o treino anterior:
    - Dataset filtrado: apenas 862 imagens Bovidae (removidas as 54 espécies irrelevantes)
    - imgsz=1280: maior resolução para capturar melhor as articulações das patas
    - freeze=5: congela as primeiras 5 camadas do backbone (preserva features gerais)
    - epochs=80: mais épocas para convergência com dataset menor
    - data augmentation: mais mosaic e flipud para variabilidade de pose
    - lr0 menor: fine-tuning mais suave (não destruir pesos já treinados)
    """
    # Parte do checkpoint do treino geral do AP-10K (não do zero)
    model = YOLO('./runs/pose/onevision-yolo/treino_base_ap10k-4/weights/best.pt')

    results = model.train(
        data='./dataset_versions/ap-10k-bovidae/data.yaml',

        # Treino
        epochs=80,
        patience=20,         # para cedo se não melhorar (evita overfitting com dataset pequeno)
        batch=8,             # menor batch pela maior resolução
        imgsz=1280,          # resolução alta para capturar articulações finas

        # Transfer Learning
        freeze=5,            # congela primeiras 5 camadas do backbone

        # Otimizador
        optimizer='AdamW',
        lr0=0.0005,          # LR inicial baixo para fine-tuning suave
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentação
        mosaic=0.5,
        flipud=0.1,
        fliplr=0.5,
        degrees=5.0,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        # Saída
        device=0,
        project='onevision-yolo',
        name='caprine-bovidae-v1',
        exist_ok=False,
        save_period=10,      # salva checkpoint a cada 10 épocas
        plots=True,
    )

if __name__ == '__main__':
    treinar_bovidae_finetune()
