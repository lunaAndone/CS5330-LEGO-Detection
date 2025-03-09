# CS5330-LAB3-LEGO Piece Detection
*Jun Liu*

## Methods

### Dataset Preparation

For this lab, I worked with the "Biggest LEGO Dataset" from Kaggle, which contains 168,000 annotated synthetic images with PASCAL VOC format annotations. Due to computational constraints, I implemented a reduced dataset approach:

1. **Data Reduction**: I limited the dataset to 10,000 images to make training feasible on my hardware (Apple M2 Pro).

2. **Data Verification**: I implemented a verification process to filter out potentially mislabeled images by checking:
   - Image file integrity
   - Bounding box validity (coordinates within image boundaries)
   - Presence of at least one annotation per image

3. **Label Simplification**: Although the original dataset contains 600 unique LEGO part types, I simplified this to a single "lego" class since the lab only required detecting and counting pieces regardless of type.

4. **Dataset Splitting**: I divided the verified images into:
   - Training set: 70% of the data
   - Validation set: 15% of the data
   - Testing set: 15% of the data

5. **Format Conversion**: I converted PASCAL VOC annotations to YOLO format (normalized center coordinates, width, and height) using the following code:

```python
def convert_to_yolo_format(image_path, xml_path, output_path):
    # Read image dimensions
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create YOLO txt file
    txt_path = os.path.join(output_path, os.path.basename(image_path).rsplit('.', 1)[0] + '.txt')
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            # Convert all object types to single class 'lego' (class_id = 0)
            class_id = 0
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
```

### Model Architecture and Training

I implemented YOLOv8 (You Only Look Once version 8) for this object detection task. YOLOv8 is a state-of-the-art, real-time object detection system that offers a good balance between speed and accuracy.

**Model Configuration:**
- Base network: YOLOv8n (nano variant - smallest and fastest YOLO version)
- Input resolution: 416×416 pixels (reduced from standard 640×640 to speed up training)
- Batch size: 16
- Optimizer: AdamW (automatically selected by YOLOv8)
- Learning rate: 0.002 (automatically determined)
- Confidence threshold for detection: 0.25
- IoU threshold for NMS: 0.7
- Early stopping patience: 10 epochs

**Training Strategy:**
- Transfer learning from a pre-trained YOLOv8n model (initialized with COCO dataset weights)
- Training for 10 epochs on an Apple M2 Pro CPU
- Total training time: approximately 5 hours

The training code using the Ultralytics framework:

```python
def train_model():
    # Load a pre-trained YOLO model (YOLOv8 nano version)
    model = YOLO('yolov8n.pt')
    
    # Train the model on your dataset
    results = model.train(
        data=YAML_PATH,
        epochs=10,
        imgsz=416,
        batch=16,
        patience=10,  # Early stopping
        verbose=True
    )
    
    return model
```

## Results and Discussion

### Performance Metrics

The model achieved excellent performance on the test set:

- **mAP@0.5**: 0.9901 (99.01%)
- **Precision**: 0.983
- **Recall**: 0.960

These metrics indicate that the model is extremely effective at detecting LEGO pieces in the test images. The high mAP@0.5 score of 99.01% suggests that almost all LEGO pieces were correctly detected with accurate bounding boxes.

### Visualization

The model successfully detected LEGO pieces in test images, as shown in detection_examples.png. The visualizations demonstrate that the model can accurately detect and count LEGO pieces of various shapes and sizes, regardless of their position in the image.

### Training Progress

The training loss and validation metrics improved consistently over the 10 epochs:

| Epoch | Box Loss | Class Loss | DFL Loss | mAP@0.5 | 
|-------|----------|------------|----------|---------|
| 1     | 0.7671   | 1.034      | 1.027    | 0.932   |
| 5     | 0.6933   | 0.5937     | 1.003    | 0.976   |
| 10    | 0.5487   | 0.4273     | 0.9431   | 0.990   |

This progression shows a consistent improvement in the model's performance throughout training, with particular gains in classification accuracy.

### Limitations

Despite the excellent results, several limitations should be acknowledged:

1. **Synthetic Data**: The model was trained exclusively on synthetic images, which might not fully represent real-world conditions with varying lighting, backgrounds, and camera angles.

2. **Single Class Detection**: The model treats all LEGO pieces as a single class, losing the ability to differentiate between the 600 different part types in the original dataset.

3. **Reduced Dataset**: Using only a subset of the available data (10,000 out of 168,000 images) might have limited the model's exposure to the full variety of LEGO pieces.

4. **Computational Constraints**: Training on CPU rather than GPU resulted in longer training times and limited the experimentation with larger models or higher resolution inputs.

5. **Simple Backgrounds**: The synthetic images likely have simple backgrounds, making detection easier than in cluttered real-world scenarios.

## Conclusion

This project successfully implemented a deep learning solution for detecting and counting LEGO pieces in images. The YOLOv8n model achieved an impressive mAP@0.5 score of 99.01%, demonstrating its effectiveness for this task.

### Key Findings:

1. YOLOv8, even in its smallest configuration (nano), is powerful enough to achieve excellent results on the LEGO piece detection task.

2. Transfer learning from pre-trained weights significantly accelerates training and improves performance, even when the target domain (LEGO pieces) differs from the source domain (COCO dataset objects).

3. A reduced dataset of 10,000 images was sufficient to achieve high accuracy, suggesting that the full 168,000 image dataset may not be necessary for this specific task.

4. Reducing the input resolution to 416×416 provided a good balance between detection accuracy and computational efficiency.

### Future Work:

1. **Multi-class Detection**: Extending the model to classify different types of LEGO pieces would make it more useful for inventory management or building instructions.

2. **Real-world Testing**: Evaluating the model on real photographs of LEGO pieces to assess how well it generalizes beyond synthetic data.

3. **Mobile Deployment**: Optimizing the model for mobile devices could enable applications for real-time LEGO piece counting.

4. **Occlusion Handling**: Improving the model's ability to detect partially occluded LEGO pieces in complex arrangements.

This lab demonstrates the effectiveness of modern object detection algorithms for specialized domains and provides a foundation for more advanced LEGO-related computer vision applications.

## References

1. Ultralytics. (2023). YOLOv8 Documentation. https://docs.ultralytics.com/

2. Kaggle. (2021). Biggest LEGO Dataset - 600 Parts. https://www.kaggle.com/datasets/dreamfactor/biggest-lego-dataset-600-parts

3. Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

4. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.

5. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 88(2), 303-338.