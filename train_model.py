from ultralytics import YOLO
import os

# Path to your dataset YAML file
YAML_PATH = './processed_dataset/data.yaml'

# Training parameters
EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 416

def train_model():
    # Load a pre-trained YOLO model (YOLOv8 nano version)
    model = YOLO('yolov8n.pt')
    
    # Train the model on your dataset
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=10,  # Early stopping
        verbose=True
    )
    
    print(f"Training complete! Model saved to {os.path.join(model.export_dir)}")
    
    return model

if __name__ == "__main__":
    trained_model = train_model()