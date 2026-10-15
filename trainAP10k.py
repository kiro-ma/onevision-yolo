from ultralytics import YOLO

def train_ap10k_model():
    model = YOLO('yolo11n-pose.pt')

    results = model.train(
        data='./dataset_versions/ap-10k/data.yaml',
        epochs=50, 
        imgsz=640,
        batch=16,
        project='onevision-yolo',
        name='treino_base_ap10k',
        device=0
    )

if __name__ == '__main__':
    train_ap10k_model()