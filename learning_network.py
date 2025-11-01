import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = ImageFolder("fruit-classification10-class/versions/1/MY_data/train", transform=data_transforms["train"])
val_dataset = ImageFolder("fruit-classification10-class/versions/1/MY_data/test", transform=data_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)


def create_model(num_classes=10):
    model = models.resnet18(weights=True)
    for param in model.parameters():
        param.requires_grad = False #запретили переобучение всех слоев

    num_features = model.fc.in_features #сколько входов у последнего слоя
    model.fc = nn.Linear(num_features, num_classes) #заменили последний слой(его и будем обучать)
    return model

model = create_model(num_classes=10)
model = model.to(device)

criterian = nn.CrossEntropyLoss()#считает ошибку
optimizer = optim.Adam(model.parameters(), lr=0.001)#меняет веса

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterian(outputs, labels)

        optimizer.zero_grad()#удалили старые градиенты
        loss.backward()#считаем градиенты
        optimizer.step()#обновляем веса

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Эпоха {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

print("Обучение завершено")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Точность на тестовых данных: {accuracy:.2f}%")


torch.save(model, "fruit_classifier_full.pth")
print("Полная модель сохранена как 'fruit_classifier_full.pth'")
# torch.save(model.state_dict(), "fruit_classifier.pth")
# print("Модель сохранена как 'fruit_classifier.pth'")

# Сохраняем информацию о классах
class_info = {
    'classes': train_dataset.classes,
    'class_to_idx': train_dataset.class_to_idx
}
with open('class_names.json', 'w') as f:
    json.dump(class_info, f)
print("Информация о классах сохранена в 'class_names.json'")