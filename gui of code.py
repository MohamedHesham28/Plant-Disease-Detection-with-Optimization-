import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import get_model
import os

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection")
        self.root.geometry("800x600")
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(device=self.device)
        
        # Load the trained model
        model_path = 'plant_disease_model.pt'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            messagebox.showerror("Error", "Trained model not found. Please train the model first.")
            return
        
        # Class names
        self.class_names = [
            'Pepper Bell Bacterial Spot',
            'Pepper Bell healthy',
            'Potato Early Blight',
            'Potato Healthy',
            'Potato Late Blight',
            'Tomato Bacterial Spot',
            'Tomato Early Blight',
            'Tomato Healthy',
            'Tomato Late Blight',
            'Tomato Leaf Mold',
            'Tomato Mosaic Virus',
            'Tomato Septoria Leaf Spot',
            'Tomato Target Spot',
            'Tomato Yellow Leaf Curl Virus',
            'Two spotted Spider Mite'
        ]
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Upload button
        self.upload_btn = tk.Button(
            self.root, 
            text="Upload Image", 
            command=self.upload_image,
            font=('Arial', 12)
        )
        self.upload_btn.pack(pady=20)
        
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Prediction label
        self.prediction_label = tk.Label(
            self.root, 
            text="", 
            font=('Arial', 12)
        )
        self.prediction_label.pack(pady=10)
        
        # Confidence label
        self.confidence_label = tk.Label(
            self.root, 
            text="", 
            font=('Arial', 12)
        )
        self.confidence_label.pack(pady=10)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            # Load and display image
            image = Image.open(file_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # Make prediction
            prediction, confidence = self.predict_image(image)
            
            # Update labels
            self.prediction_label.config(
                text=f"Prediction: {self.class_names[prediction]}"
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%"
            )
    
    def predict_image(self, image):
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item() * 100

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop() 