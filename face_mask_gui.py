import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

# Configure matplotlib for better GUI integration
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
import joblib
import os
import glob
from typing import List, Tuple
import threading
import pandas as pd
from datetime import datetime

class FaceMaskClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Classification System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.classifier = None
        self.pca_model = None
        self.scaler = None
        self.dataset_path = r"C:\Users\ROG\Desktop\FaceMaskImage"
        self.test_results = []
        self.current_image_index = 0
        self.test_images = []
        self.test_labels = []
        
        # Create GUI components
        self.create_widgets()
        
        # Load model if exists
        self.load_model()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Mask Classification System", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Model Training & Metrics
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Training & Metrics")
        self.create_metrics_tab()
        
        # Tab 2: Individual Image Testing
        self.testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.testing_frame, text="Image Testing")
        self.create_testing_tab()
        
        # Tab 3: Batch Testing
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Batch Testing")
        self.create_batch_tab()
        
    def create_metrics_tab(self):
        # Left panel - Controls
        left_panel = ttk.Frame(self.metrics_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Training controls
        training_group = ttk.LabelFrame(left_panel, text="Model Training", padding=10)
        training_group.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(training_group, text="Dataset Path:").pack(anchor=tk.W)
        self.dataset_var = tk.StringVar(value=self.dataset_path)
        dataset_entry = ttk.Entry(training_group, textvariable=self.dataset_var, width=40)
        dataset_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(training_group, text="Browse", command=self.browse_dataset).pack(fill=tk.X)
        
        # Training parameters
        params_frame = ttk.Frame(training_group)
        params_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(params_frame, text="Augmentation Factor:").grid(row=0, column=0, sticky=tk.W)
        self.aug_factor_var = tk.StringVar(value="2")
        ttk.Entry(params_frame, textvariable=self.aug_factor_var, width=10).grid(row=0, column=1, padx=(5, 0))
        
        ttk.Label(params_frame, text="PCA Variance:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.pca_var_var = tk.StringVar(value="0.95")
        ttk.Entry(params_frame, textvariable=self.pca_var_var, width=10).grid(row=1, column=1, padx=(5, 0), pady=(5, 0))
        
        ttk.Button(training_group, text="Train Model", command=self.train_model).pack(fill=tk.X, pady=(10, 0))
        
        # Model status
        status_group = ttk.LabelFrame(left_panel, text="Model Status", padding=10)
        status_group.pack(fill=tk.X, pady=(0, 10))
        
        self.model_status_var = tk.StringVar(value="No model loaded")
        ttk.Label(status_group, textvariable=self.model_status_var).pack()
        
        ttk.Button(status_group, text="Load Model", command=self.load_model).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(status_group, text="Save Model", command=self.save_model).pack(fill=tk.X, pady=(5, 0))
        
        # Right panel - Metrics and Plots
        right_panel = ttk.Frame(self.metrics_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for metrics
        metrics_notebook = ttk.Notebook(right_panel)
        metrics_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        self.metrics_display = ttk.Frame(metrics_notebook)
        metrics_notebook.add(self.metrics_display, text="Metrics")
        
        # Create text widget for metrics
        self.metrics_text = tk.Text(self.metrics_display, wrap=tk.WORD, height=20)
        metrics_scrollbar = ttk.Scrollbar(self.metrics_display, orient=tk.VERTICAL, command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=metrics_scrollbar.set)
        
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metrics_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Plots tab
        self.plots_frame = ttk.Frame(metrics_notebook)
        metrics_notebook.add(self.plots_frame, text="Plots")
        
    def create_testing_tab(self):
        # Top controls
        controls_frame = ttk.Frame(self.testing_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(controls_frame, text="Load Single Image", command=self.load_single_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="Load Multiple Images", command=self.load_multiple_images).pack(side=tk.LEFT, padx=(0, 10))
        
        # Image count selector
        ttk.Label(controls_frame, text="Number of images:").pack(side=tk.LEFT, padx=(20, 5))
        self.image_count_var = tk.StringVar(value="10")
        image_count_combo = ttk.Combobox(controls_frame, textvariable=self.image_count_var, 
                                        values=["5", "10", "20", "50", "100"], width=10)
        image_count_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(controls_frame, text="Test Images", command=self.test_images).pack(side=tk.LEFT)
        
        # Main content area
        content_frame = ttk.Frame(self.testing_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Left - Image display
        image_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding=10)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="No image loaded")
        self.image_label.pack(expand=True)
        
        # Right - Results
        results_frame = ttk.LabelFrame(content_frame, text="Classification Results", padding=10)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Navigation controls
        nav_frame = ttk.Frame(results_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT)
        
        self.image_info_var = tk.StringVar(value="No image selected")
        ttk.Label(nav_frame, textvariable=self.image_info_var).pack(side=tk.RIGHT)
        
        # Results display
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=15)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_batch_tab(self):
        # Controls
        controls_frame = ttk.Frame(self.batch_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(controls_frame, text="Run Batch Test", command=self.run_batch_test).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Results area
        results_frame = ttk.Frame(self.batch_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create notebook for batch results
        batch_notebook = ttk.Notebook(results_frame)
        batch_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        self.summary_frame = ttk.Frame(batch_notebook)
        batch_notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD)
        summary_scrollbar = ttk.Scrollbar(self.summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Detailed results tab
        self.detailed_frame = ttk.Frame(batch_notebook)
        batch_notebook.add(self.detailed_frame, text="Detailed Results")
        
        # Create treeview for detailed results
        columns = ('Image', 'Prediction', 'Confidence', 'Actual', 'Correct')
        self.results_tree = ttk.Treeview(self.detailed_frame, columns=columns, show='headings')
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        
        results_scrollbar = ttk.Scrollbar(self.detailed_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Directory")
        if path:
            self.dataset_var.set(path)
            self.dataset_path = path
            
    def train_model(self):
        def train_thread():
            try:
                self.root.config(cursor="wait")
                
                # Import training functions
                from algo import complete_preprocessing_pipeline
                
                # Get parameters
                augmentation_factor = int(self.aug_factor_var.get())
                pca_variance = float(self.pca_var_var.get())
                
                # Update status
                self.model_status_var.set("Training in progress...")
                self.root.update()
                
                # Train model
                self.classifier, metrics, self.pca_model = complete_preprocessing_pipeline(
                    dataset_path=self.dataset_path,
                    augmentation_factor=augmentation_factor,
                    pca_variance=pca_variance
                )
                
                # Store metrics for later use
                self.metrics = metrics
                
                # Update GUI
                self.root.after(0, lambda: self.update_metrics_display(metrics))
                self.root.after(0, lambda: self.model_status_var.set("Model trained successfully"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
                self.root.after(0, lambda: self.model_status_var.set("Training failed"))
            finally:
                self.root.after(0, lambda: self.root.config(cursor=""))
        
        threading.Thread(target=train_thread, daemon=True).start()
        
    def update_metrics_display(self, metrics):
        # Clear previous content
        self.metrics_text.delete(1.0, tk.END)
        
        # Display metrics
        self.metrics_text.insert(tk.END, "=== MODEL PERFORMANCE METRICS ===\n\n")
        self.metrics_text.insert(tk.END, f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        self.metrics_text.insert(tk.END, f"Cross-validation Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})\n\n")
        
        self.metrics_text.insert(tk.END, "=== CLASSIFICATION REPORT ===\n")
        self.metrics_text.insert(tk.END, metrics['classification_report'])
        self.metrics_text.insert(tk.END, "\n")
        
        self.metrics_text.insert(tk.END, "=== CONFUSION MATRIX ===\n")
        self.metrics_text.insert(tk.END, str(metrics['confusion_matrix']))
        
        # Create plots
        self.create_performance_plots(metrics)
        
    def create_performance_plots(self, metrics):
        # Clear previous plots
        for widget in self.plots_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots - use smaller size that fits GUI
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle('Model Performance Analysis', fontsize=14)
        
        # ROC Curve
        y_test = metrics['y_test']
        y_pred_proba = metrics['y_pred_proba'][:, 1]  # Probability of positive class
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Confusion Matrix Heatmap
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # Prediction Distribution
        ax3.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='No Mask', color='red')
        ax3.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Mask', color='green')
        ax3.set_xlabel('Predicted Probability')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prediction Distribution')
        ax3.legend()
        ax3.grid(True)
        
        # Performance Metrics Bar Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            metrics['accuracy'],
            metrics['classification_report'].split('\n')[3].split()[1:5][1],  # Extract precision
            metrics['classification_report'].split('\n')[3].split()[1:5][2],  # Extract recall
            metrics['classification_report'].split('\n')[3].split()[1:5][3]   # Extract f1-score
        ]
        metrics_values = [float(x) for x in metrics_values]
        
        bars = ax4.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Use constrained_layout instead of tight_layout for better GUI compatibility
        fig.set_constrained_layout(True)
        
        # Embed plot in GUI with proper sizing
        canvas = FigureCanvasTkAgg(fig, self.plots_frame)
        canvas.draw()
        
        # Pack the canvas with proper constraints
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_model(self):
        try:
            if os.path.exists('face_mask_classifier.pkl'):
                self.classifier = joblib.load('face_mask_classifier.pkl')
                self.model_status_var.set("Model loaded successfully")
                messagebox.showinfo("Success", "Model loaded successfully!")
            else:
                self.model_status_var.set("No saved model found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def save_model(self):
        if self.classifier is not None:
            try:
                joblib.dump(self.classifier, 'face_mask_classifier.pkl')
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No model to save. Please train a model first.")
            
    def load_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            self.test_images = [file_path]
            self.test_labels = [None]  # Unknown label
            self.current_image_index = 0
            self.display_current_image()
            
    def load_multiple_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_paths:
            self.test_images = list(file_paths)
            self.test_labels = [None] * len(file_paths)
            self.current_image_index = 0
            self.display_current_image()
            
    def test_images(self):
        if not self.test_images:
            messagebox.showwarning("Warning", "No images loaded for testing.")
            return
            
        if self.classifier is None:
            messagebox.showwarning("Warning", "No model loaded. Please train or load a model first.")
            return
            
        def test_thread():
            try:
                self.root.config(cursor="wait")
                
                # Import testing functions
                from algo import extract_hog_features_20bins, apply_pca_dimensionality_reduction
                
                results = []
                
                for i, image_path in enumerate(self.test_images):
                    try:
                        # Extract features
                        features = extract_hog_features_20bins(image_path, target_size=(64, 128))
                        
                        # Apply PCA (if available)
                        if hasattr(self, 'pca_model') and self.pca_model is not None:
                            features = self.pca_model.transform(features.reshape(1, -1))
                        else:
                            # If no PCA model, use the original features
                            features = features.reshape(1, -1)
                        
                        # Make prediction
                        prediction = self.classifier.predict(features)[0]
                        confidence = self.classifier.predict_proba(features)[0]
                        
                        results.append({
                            'path': image_path,
                            'prediction': prediction,
                            'confidence': confidence,
                            'label': self.test_labels[i]
                        })
                        
                        # Update progress
                        progress = (i + 1) / len(self.test_images) * 100
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue
                
                # Update GUI with results
                self.root.after(0, lambda: self.update_test_results(results))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Testing failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.root.config(cursor=""))
                
        threading.Thread(target=test_thread, daemon=True).start()
        
    def update_test_results(self, results):
        self.test_results = results
        self.current_image_index = 0
        self.display_current_image()
        self.display_current_results()
        
    def display_current_image(self):
        if not self.test_images or self.current_image_index >= len(self.test_images):
            return
            
        try:
            # Load and resize image
            image_path = self.test_images[self.current_image_index]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = image.shape[:2]
            max_size = 400
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Convert to PIL and display
            pil_image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Update image info
            filename = os.path.basename(image_path)
            self.image_info_var.set(f"Image {self.current_image_index + 1}/{len(self.test_images)}: {filename}")
            
        except Exception as e:
            self.image_label.configure(image="", text=f"Error loading image: {str(e)}")
            
    def display_current_results(self):
        if not self.test_results or self.current_image_index >= len(self.test_results):
            return
            
        result = self.test_results[self.current_image_index]
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Display results
        self.results_text.insert(tk.END, "=== CLASSIFICATION RESULTS ===\n\n")
        
        filename = os.path.basename(result['path'])
        self.results_text.insert(tk.END, f"Image: {filename}\n\n")
        
        prediction = result['prediction']
        confidence = result['confidence']
        
        self.results_text.insert(tk.END, f"Prediction: {'MASK' if prediction == 1 else 'NO MASK'}\n")
        self.results_text.insert(tk.END, f"Confidence: {confidence[prediction]:.4f}\n\n")
        
        self.results_text.insert(tk.END, "Confidence Scores:\n")
        self.results_text.insert(tk.END, f"  No Mask: {confidence[0]:.4f}\n")
        self.results_text.insert(tk.END, f"  Mask: {confidence[1]:.4f}\n\n")
        
        if result['label'] is not None:
            actual = "MASK" if result['label'] == 1 else "NO MASK"
            correct = "✓" if prediction == result['label'] else "✗"
            self.results_text.insert(tk.END, f"Actual: {actual}\n")
            self.results_text.insert(tk.END, f"Correct: {correct}\n")
            
    def previous_image(self):
        if self.test_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            self.display_current_results()
            
    def next_image(self):
        if self.test_images and self.current_image_index < len(self.test_images) - 1:
            self.current_image_index += 1
            self.display_current_image()
            self.display_current_results()
            
    def run_batch_test(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "No model loaded. Please train or load a model first.")
            return
            
        # Get number of images to test
        try:
            num_images = int(self.image_count_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of images.")
            return
            
        def batch_test_thread():
            try:
                self.root.config(cursor="wait")
                
                # Import functions
                from algo import load_face_mask_dataset, create_balanced_dataset, apply_pca_dimensionality_reduction
                
                # Load dataset
                image_paths = load_face_mask_dataset(self.dataset_path)
                
                if len(image_paths) < num_images:
                    messagebox.showwarning("Warning", f"Only {len(image_paths)} images available. Using all images.")
                    num_images = len(image_paths)
                
                # Select random subset
                import random
                random.seed(42)
                selected_paths = random.sample(image_paths, num_images)
                
                # Create balanced dataset for testing
                features, labels = create_balanced_dataset(selected_paths, augmentation_factor=1)
                
                # Apply PCA
                features_reduced, pca = apply_pca_dimensionality_reduction(features, 0.95)
                
                # Make predictions
                predictions = self.classifier.predict(features_reduced)
                probabilities = self.classifier.predict_proba(features_reduced)
                
                # Calculate metrics
                accuracy = np.mean(predictions == labels)
                
                # Create results
                results = []
                for i, (path, pred, prob, label) in enumerate(zip(selected_paths, predictions, probabilities, labels)):
                    results.append({
                        'image': os.path.basename(path),
                        'prediction': pred,
                        'confidence': prob[pred],
                        'actual': label,
                        'correct': pred == label
                    })
                
                # Update GUI
                self.root.after(0, lambda: self.update_batch_results(results, accuracy))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Batch test failed: {error_msg}"))
            finally:
                self.root.after(0, lambda: self.root.config(cursor=""))
                
        threading.Thread(target=batch_test_thread, daemon=True).start()
        
    def update_batch_results(self, results, accuracy):
        # Update summary
        self.summary_text.delete(1.0, tk.END)
        
        self.summary_text.insert(tk.END, "=== BATCH TEST RESULTS ===\n\n")
        self.summary_text.insert(tk.END, f"Total Images Tested: {len(results)}\n")
        self.summary_text.insert(tk.END, f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        # Count predictions
        mask_predictions = sum(1 for r in results if r['prediction'] == 1)
        no_mask_predictions = sum(1 for r in results if r['prediction'] == 0)
        
        self.summary_text.insert(tk.END, f"Predictions:\n")
        self.summary_text.insert(tk.END, f"  Mask: {mask_predictions}\n")
        self.summary_text.insert(tk.END, f"  No Mask: {no_mask_predictions}\n\n")
        
        # Count actual labels
        mask_actual = sum(1 for r in results if r['actual'] == 1)
        no_mask_actual = sum(1 for r in results if r['actual'] == 0)
        
        self.summary_text.insert(tk.END, f"Actual Labels:\n")
        self.summary_text.insert(tk.END, f"  Mask: {mask_actual}\n")
        self.summary_text.insert(tk.END, f"  No Mask: {no_mask_actual}\n\n")
        
        # Correct predictions
        correct = sum(1 for r in results if r['correct'])
        self.summary_text.insert(tk.END, f"Correct Predictions: {correct}/{len(results)} ({correct/len(results):.2%})\n")
        
        # Update detailed results
        self.results_tree.delete(*self.results_tree.get_children())
        
        for result in results:
            prediction_text = "Mask" if result['prediction'] == 1 else "No Mask"
            actual_text = "Mask" if result['actual'] == 1 else "No Mask"
            correct_text = "✓" if result['correct'] else "✗"
            
            self.results_tree.insert('', 'end', values=(
                result['image'],
                prediction_text,
                f"{result['confidence']:.3f}",
                actual_text,
                correct_text
            ))
            
    def export_results(self):
        if not hasattr(self, 'test_results') or not self.test_results:
            messagebox.showwarning("Warning", "No results to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        
        if file_path:
            try:
                # Create DataFrame
                df_data = []
                for result in self.test_results:
                    df_data.append({
                        'Image': os.path.basename(result['path']),
                        'Prediction': 'Mask' if result['prediction'] == 1 else 'No Mask',
                        'Confidence': result['confidence'][result['prediction']],
                        'Actual': 'Mask' if result['label'] == 1 else 'No Mask' if result['label'] == 0 else 'Unknown',
                        'Correct': result['prediction'] == result['label'] if result['label'] is not None else 'Unknown'
                    })
                
                df = pd.DataFrame(df_data)
                
                # Export
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                    
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

def main():
    root = tk.Tk()
    app = FaceMaskClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
