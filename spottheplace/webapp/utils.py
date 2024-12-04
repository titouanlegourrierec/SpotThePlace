
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download


def load_model():
    model_path = hf_hub_download(
        repo_id="titouanlegourrierec/SpotThePlace",
        filename="Classification_ResNet50_4countries.pth"
    )
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 countries
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_country(image_path):
    model = load_model()

    class_labels = {0: 'France', 1: 'Japan', 2: 'Mexico', 3: 'South Africa'}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_labels[predicted.item()]
