import torch
import torchvision  
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Classes = [ 'Uninfected_Patients', 'Plasmodium_falciparum', 'Plasmodium_Vivax']
class VGGModel(torch.nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()

    def forward(self, x):
        pass

model = VGGModel()
model_path = './Maraliavgg16.pt'  
model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.eval()
print("Model loaded successfully!")

# Define the transformation for a single image
single_image_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load a single image
image_path = './UNI.jpg'  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')

# Apply the transformation
input_tensor = single_image_transform(image)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

# Display the image with the predicted label
plt.imshow(np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0)))
plt.axis('off')
plt.title(f"Predicted Class: {Classes[prediction]}")
plt.show()