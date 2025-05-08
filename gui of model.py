import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import get_model
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
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
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel for image upload and display
        left_panel = tk.Frame(main_frame, bg='#f0f0f0')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Upload button
        self.upload_btn = tk.Button(
            left_panel, 
            text="Upload Image", 
            command=self.upload_image,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            relief=tk.RAISED
        )
        self.upload_btn.pack(pady=20)
        
        # Image display frame
        image_frame = tk.Frame(left_panel, bg='white', relief=tk.SUNKEN, borderwidth=2)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_label = tk.Label(image_frame, bg='white')
        self.image_label.pack(padx=10, pady=10)
        
        # Right panel for results
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Results frame
        results_frame = tk.LabelFrame(right_panel, text="Detection Results", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Prediction label
        self.prediction_label = tk.Label(
            results_frame, 
            text="No image uploaded", 
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0'
        )
        self.prediction_label.pack(pady=20)
        
        # Confidence label
        self.confidence_label = tk.Label(
            results_frame, 
            text="", 
            font=('Arial', 12),
            bg='#f0f0f0'
        )
        self.confidence_label.pack(pady=10)
        
        # Confidence bar
        self.confidence_bar = ttk.Progressbar(
            results_frame,
            orient='horizontal',
            length=300,
            mode='determinate'
        )
        self.confidence_bar.pack(pady=10)
        
        # Top predictions frame
        top_predictions_frame = tk.LabelFrame(right_panel, text="Top Predictions", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        top_predictions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a figure for the bar chart
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_predictions_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                image = image.resize((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                
                # Make prediction
                prediction, confidence, top_predictions = self.predict_image(image)
                
                # Update prediction label
                self.prediction_label.config(
                    text=f"Prediction: {self.class_names[prediction]}",
                    fg='#2196F3'
                )
                
                # Update confidence label and bar
                confidence_percentage = confidence * 100
                self.confidence_label.config(
                    text=f"Confidence: {confidence_percentage:.2f}%",
                    fg='#4CAF50'
                )
                self.confidence_bar['value'] = confidence_percentage
                
                # Update bar chart
                self.update_bar_chart(top_predictions)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
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
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            top_predictions = list(zip(top5_idx[0].cpu().numpy(), top5_prob[0].cpu().numpy()))
            
        return predicted.item(), confidence.item(), top_predictions
    
    def update_bar_chart(self, top_predictions):
        # Clear previous plot
        self.ax.clear()
        
        # Prepare data for plotting
        indices = [idx for idx, _ in top_predictions]
        probabilities = [prob for _, prob in top_predictions]
        labels = [self.class_names[idx] for idx in indices]
        
        # Create bar chart
        bars = self.ax.barh(range(len(labels)), probabilities, color='#4CAF50')
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_yticklabels(labels)
        self.ax.set_xlabel('Probability')
        self.ax.set_title('Top 5 Predictions')
        
        # Add probability values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            self.ax.text(width, i, f'{width:.2f}', ha='left', va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Update canvas
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()