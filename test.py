from CNN import ConvNet, device
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define the transform to match the training preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# Define the class names in CIFAR-10 (for visualization purposes)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def test_single_image(image_path, model):
    # Load the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    # Preprocess the image: Resize (if needed), apply transformations

    # Move the image to the device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations (not needed for inference)
    with torch.no_grad():
        # Get the model's output for the image
        output = model(image)

        # Get the predicted class (index of the max value)
        _, predicted = torch.max(output, 1)

    # Return the predicted class name
    predicted_class = class_names[predicted.item()]
    return predicted_class

# Example usage:
image_path = 'images.jpg'  # Replace with your image path
model = ConvNet().to(device)  # Load your model
model.load_state_dict(torch.load('cnn.pth'))  # Load the saved model weights
predicted_class = test_single_image(image_path, model)
print(f"Predicted class: {predicted_class}")