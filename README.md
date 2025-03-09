# LEGO Piece Detection

This project implements a deep learning model to detect and count LEGO pieces in images using YOLOv8.

## Dataset

The model is trained on the "Biggest LEGO Dataset" from Kaggle, which contains 168,000 annotated synthetic images with PASCAL VOC annotations. I used a reduced dataset of 10,000 images for efficiency, converting all 600 unique LEGO part types to a single "lego" class.

## Model

- Architecture: YOLOv8n (nano variant)
- Input resolution: 416×416 pixels
- Training: 10 epochs on Apple M2 Pro CPU
- Performance: 99.01% mAP@0.5 on test set

## Project Structure

```
├── prepare_dataset.py   # Dataset preparation script
├── train_model.py       # Model training script
├── evaluate_model.py    # Evaluation script
├── app.py               # Interactive Gradio interface for testing on new images
├── processed_dataset/   # Processed dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── runs/                # Training outputs directory
│   └── detect/
│       └── train3/
│           └── weights/
│               ├── best.pt  # Best model weights
│               └── last.pt  # Final model weights
└── examples/            # Example images for testing
```

## Usage

### Prepare Dataset

```bash
python prepare_dataset.py
```

### Train Model

```bash
python train_model.py
```

### Evaluate Model

```bash
python evaluate_model.py
```

### Interactive Demo

```bash
python app.py
```

This launches a Gradio web interface where you can upload images and test the model interactively.

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics
- OpenCV
- Matplotlib
- tqdm
- scikit-learn
- PyYAML
- gradio (for interactive demo)

## Results

The model achieves excellent performance with:
- mAP@0.5: 0.9901 (99.01%)
- Precision: 0.983
- Recall: 0.960

This demonstrates that YOLOv8, even in its smallest configuration (nano), is highly effective for LEGO piece detection.