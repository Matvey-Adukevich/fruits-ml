import torch
from PIL import Image
import json
from torchvision import transforms
import torchvision.models.resnet

torch.serialization.add_safe_globals([torchvision.models.resnet.ResNet])
model = torch.load("fruit_classifier_full.pth", map_location="cpu", weights_only=False)
model.eval()#запрет на изменение модели
with open("class_names.json", "r") as f:
    classes = json.load(f)["classes"]
print("Модель и классы загружены")



data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_fruit(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = data_transforms(image).unsqueeze(0)#создаем батч из картинки

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output[0], dim=0)#используем функцию активации, чтобы сумма значений была = 1
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    fruit_name = classes[predicted_class]
    return fruit_name, confidence


fruit, conf = predict_fruit("examples/watermelon.jpg")
print(f"Предсказание: {fruit} (уверенность: {conf:.1%})")

fruit, conf = predict_fruit("examples/apple.jpg")
print(f"Предсказание: {fruit} (уверенность: {conf:.1%})")