import torch
import torchvision
import numpy as np
import gradio as gr
from PIL import Image

class ERF:
    def __init__(self):
        self.model = None
        self.list_models = torchvision.models.list_models()
        self.app = gr.Interface(
            fn=self.deep_visualization,
            inputs = [
                "image",
                gr.Slider(minimum=0, maximum=10, label="Thresholding"),
                gr.Slider(minimum=0, maximum=10, label="Opacity Controller"),
                gr.Dropdown(choices=self.list_models, label="Select models")
            ],
            outputs=["image"]
        )
        
    def get_model(self, model_name):
        model = torchvision.models.resnet50(weights=model_name)
        return model

    def choose_option(self, option):
        dataset = "IMAGENET1K_V1"
        model_name = f"ResNet50_Weights.{dataset}"
        self.model = self.get_model(model_name)

    def deep_visualization(self, image, threshold, opacity, models):
        print(self.model)
    
    def run(self):
        self.app.launch(debug=True)

    
if __name__ == "__main__":
    erf = ERF()
    erf.run()