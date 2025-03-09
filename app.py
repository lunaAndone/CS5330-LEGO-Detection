from ultralytics import YOLO
import gradio as gr
import os

# Load your model
model = YOLO('./runs/detect/train3/weights/best.pt')

def detect_lego(image, confidence=0.25):
    # Run inference
    results = model.predict(image, conf=confidence)
    result = results[0]
    
    # Get the output image with detections
    output_img = result.plot()
    
    # Count the number of LEGO pieces
    num_pieces = len(result.boxes)
    
    return output_img, f"Detected {num_pieces} LEGO pieces"

# Find example images
example_dir = "./examples"
example_images = [os.path.join(example_dir, f) for f in os.listdir(example_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create examples list for Gradio
examples = [[img, 0.25] for img in example_images]

# Create Gradio interface
demo = gr.Interface(
    fn=detect_lego,
    inputs=[
        gr.Image(type="numpy"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detections"),
        gr.Textbox(label="Result")
    ],
    title="LEGO Piece Detector",
    description="Upload an image to detect LEGO pieces.",
    examples=examples
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)