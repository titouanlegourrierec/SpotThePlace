import torch
from torchvision import transforms
from PIL import Image
from spottheplace.ml.trainer import Trainer

def load_model(model_path="spottheplace/trainings/pretrained_model.pth"):
    model = torch.load(model_path)
    model.eval()
    return model

def predict_country(image_path):
    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ["France", "Mexico", "South Africa", "Japan", "Italy"]
    return classes[predicted.item()]
