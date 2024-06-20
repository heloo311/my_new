import torch
from torchvision import transforms
from PIL import Image
import model_mae

# Assume the previous code is defined above or imported appropriately
# Ensure you have the following imports as well
import timm  # if timm is not installed, you can install it using `pip install timm`

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img_path = r"C:\Users\Acer\PycharmProjects\mae-main\dew\2208.jpg"  # replace with your image path
img = Image.open(img_path)
img = transform(img).unsqueeze(0)  # add batch dimension

# Instantiate the model
model = model_vit.mae_vit_base_patch16()
model.eval()

# Forward the image through the model
with torch.no_grad():
    loss, pred, mask = model(img)

# Print the output (loss, prediction, and mask)
print(f"Loss: {loss.item()}")
print(f"Prediction shape: {pred.shape}")
print(f"Mask shape: {mask.shape}")
