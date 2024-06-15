import torchvision.models as models

dataset = "IMAGENET1K_V1"
selection = "resnet50"

if hasattr(models, selection):
    model_class = getattr(models, selection)
    print(model_class)
    model = model_class(weights=f"{selection}_Weights.{dataset}", progress=True)
    