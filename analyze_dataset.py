import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

def analyze_sample_images(dataset_path: str, num_samples: int = 10):
    """
    Analyze sample images from the dataset to understand the structure
    """
    # Get all image paths
    pattern = os.path.join(dataset_path, "maksssksksss*.png")
    image_paths = glob.glob(pattern)
    image_paths.sort()
    
    print(f"Total images found: {len(image_paths)}")
    
    # Analyze first few images
    for i in range(min(num_samples, len(image_paths))):
        image_path = image_paths[i]
        image = cv2.imread(image_path)
        
        if image is not None:
            print(f"\nImage {i+1}: {os.path.basename(image_path)}")
            print(f"  Shape: {image.shape}")
            print(f"  Size: {image.size} bytes")
            print(f"  Mean pixel values (BGR): {np.mean(image, axis=(0,1))}")
            
            # Check if image contains face-like features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            print(f"  Edge density: {edge_density:.4f}")
            
            # Save a sample for visual inspection
            if i < 5:
                cv2.imwrite(f"sample_{i+1}.png", image)
                print(f"  Saved as sample_{i+1}.png")
        else:
            print(f"Could not load image: {image_path}")

def check_for_different_patterns(dataset_path: str):
    """
    Check if there are different file patterns that might indicate different classes
    """
    pattern = os.path.join(dataset_path, "maksssksksss*.png")
    image_paths = glob.glob(pattern)
    
    # Analyze file sizes to see if there are clusters
    file_sizes = []
    for path in image_paths:
        size = os.path.getsize(path)
        file_sizes.append(size)
    
    file_sizes = np.array(file_sizes)
    
    print(f"File size statistics:")
    print(f"  Mean: {np.mean(file_sizes):.0f} bytes")
    print(f"  Std: {np.std(file_sizes):.0f} bytes")
    print(f"  Min: {np.min(file_sizes):.0f} bytes")
    print(f"  Max: {np.max(file_sizes):.0f} bytes")
    
    # Check for size clusters
    from sklearn.cluster import KMeans
    if len(file_sizes) > 10:
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(file_sizes.reshape(-1, 1))
        
        print(f"\nSize clusters:")
        for i in range(2):
            cluster_sizes = file_sizes[clusters == i]
            print(f"  Cluster {i+1}: {len(cluster_sizes)} images, "
                  f"mean size: {np.mean(cluster_sizes):.0f} bytes")

if __name__ == "__main__":
    dataset_path = r"C:\Users\ROG\Desktop\FaceMaskImage"
    
    print("=== Dataset Analysis ===")
    analyze_sample_images(dataset_path, num_samples=10)
    print("\n=== File Size Analysis ===")
    check_for_different_patterns(dataset_path)
