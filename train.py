from ultralytics import YOLO

def train_custom_pose_model():
    # Carrega o modelo base pré-treinado (para transfer learning)
    model = YOLO('yolo11n-pose.pt')

    # Inicia o treinamento
    # Atualize o caminho do 'data' para apontar exatamente para o seu data.yaml
    results = model.train(
        data='./dataset_versions/onevision-beta-02.yolov8/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='onevision-yolo',
        name='treino_animais_pose'
    )

if __name__ == '__main__':
    train_custom_pose_model()