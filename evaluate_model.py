from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import random
from glob import glob

# Path to your best model
MODEL_PATH = './runs/detect/train3/weights/best.pt'

def evaluate_model():
    # Load your trained model
    model = YOLO(MODEL_PATH)
    
    # Validate the model on the test set
    results = model.val(data='./processed_dataset/data.yaml')
    
    # Print mAP@0.5 (as required in the assignment)
    map50 = results.box.map50
    print(f"Model Performance - mAP@0.5: {map50:.4f}")
    
    return model, map50

def visualize_examples(model, num_samples=5):
    # Get some random test images
    test_images = glob(os.path.join('./processed_dataset/test/images', '*'))
    samples = random.sample(test_images, min(num_samples, len(test_images)))
    
    # Create figure for visualization
    plt.figure(figsize=(15, 12))
    
    for i, img_path in enumerate(samples):
        # Run inference
        results = model.predict(img_path, conf=0.25)
        
        # Plot results
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(results[0].plot())
        plt.title(f"Detected {len(results[0].boxes)} LEGO pieces")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_examples.png')
    plt.show()

if __name__ == "__main__":
    model, map50 = evaluate_model()
    visualize_examples(model)