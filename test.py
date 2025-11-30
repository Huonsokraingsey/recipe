import torch
from torchvision import transforms
from PIL import Image
from tkinter import Tk, filedialog

# ==============================
# 1. Load your saved model
# ==============================

model_path = "C:/Users/ASUS/Documents/KOLNEY/MACHINE LEARNING/Final Project/project/fruit_veg_classifier_2.pth"  
model = torch.load(model_path, weights_only=False)

model.eval()

print("Model loaded successfully!")

# ==============================
# 2. Class names (EDIT THIS)
# ==============================

classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
     'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
     'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'not ingredient', 
     'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 
     'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']



# ==============================
# 3. Image preprocessing
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# 4. Function to upload and predict
# ==============================

def choose_image():
    Tk().withdraw()  # close tkinter main window
    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    return file_path

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    class_idx = predicted.item()
    return class_idx, classes[class_idx]

# ==============================
# 5. Run the prediction
# ==============================

img_path = choose_image()
print("Selected image:", img_path)

idx, name = predict(img_path)
print(f"\nPredicted class index: {idx}")
print(f"Predicted class name : {name}")
