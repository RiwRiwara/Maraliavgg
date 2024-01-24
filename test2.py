import torch
import torchvision  
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

device = torch.device('cpu')

class VGGModel(torch.nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()

    def forward(self, x):
        pass

# Load the model
model = VGGModel()
model_path = './Maraliavgg16.pt'
model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.to(device)
model.eval()
print("Model loaded successfully!")

single_image_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

Classes = ['Uninfected_Patients', 'Plasmodium_falciparum', 'Plasmodium_Vivax']

image_path = './UNI.jpg'  
image = Image.open(image_path)

transformed_image = single_image_transform(image)

transformed_image = transformed_image.unsqueeze(0)

outputs = model(transformed_image)

print("Raw Scores:", outputs)

_, predicted = torch.max(outputs, 1)

print("Predicted Class Index:", predicted.item())

image = transformed_image.squeeze().cpu().permute(1, 2, 0).numpy()
predicted_label = predicted.item()
predicted_class_name = Classes[predicted_label]
print(f"Predicted Class: {predicted_class_name}")
