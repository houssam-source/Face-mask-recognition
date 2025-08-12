import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from typing import List, Tuple
import glob
import random
from sklearn.utils import shuffle

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range

    Args:
        image: Input image array

    Returns:
        Normalized image array
    """
    return image.astype(np.float32) / 255.0

def apply_data_augmentation(image: np.ndarray,
                           rotation_range: int = 15,
                           brightness_range: float = 0.2,
                           contrast_range: float = 0.15,
                           flip_prob: float = 0.5) -> np.ndarray:
    """
    Apply data augmentation to an image

    Args:
        image: Input image array
        rotation_range: Maximum rotation angle in degrees
        brightness_range: Brightness variation range
        contrast_range: Contrast variation range
        flip_prob: Probability of horizontal flip

    Returns:
        Augmented image array
    """
    augmented = image.copy()

    # Horizontal flip
    if random.random() < flip_prob:
        augmented = cv2.flip(augmented, 1)

    # Rotation
    if rotation_range > 0:
        angle = random.uniform(-rotation_range, rotation_range)
        height, width = augmented.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (width, height))

    # Brightness adjustment
    if brightness_range > 0:
        brightness_factor = random.uniform(1 - brightness_range, 1 + brightness_range)
        augmented = cv2.convertScaleAbs(augmented, alpha=brightness_factor, beta=0)

    # Contrast adjustment
    if contrast_range > 0:
        contrast_factor = random.uniform(1 - contrast_range, 1 + contrast_range)
        augmented = cv2.convertScaleAbs(augmented, alpha=contrast_factor, beta=0)

    return augmented

def create_augmented_dataset(image_paths: List[str],
                           labels: List[int],
                           augmentation_factor: int = 2,
                           target_size: Tuple[int, int] = (64, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented dataset with balanced classes

    Args:
        image_paths: List of image paths
        labels: List of corresponding labels
        augmentation_factor: How many augmented versions to create per image
        target_size: Target image size

    Returns:
        Augmented images and labels
    """
    augmented_images = []
    augmented_labels = []

    print(f"Creating augmented dataset with factor {augmentation_factor}...")

    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        try:
            # Load and preprocess original image
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.resize(image, target_size)
            image = normalize_image(image)

            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)

            # Create augmented versions
            for _ in range(augmentation_factor):
                aug_image = apply_data_augmentation(image)
                augmented_images.append(aug_image)
                augmented_labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    return np.array(augmented_images), np.array(augmented_labels)

def extract_hog_features_from_augmented(images: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from augmented images

    Args:
        images: Array of augmented images

    Returns:
        HOG feature matrix
    """
    features_list = []

    for i, image in enumerate(images):
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Extract HOG features
            features = hog(
                gray,
                orientations=20,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False
            )
            features_list.append(features)

            if (i + 1) % 500 == 0:
                print(f"Extracted HOG features from {i + 1}/{len(images)} images...")

        except Exception as e:
            print(f"Error extracting features from image {i}: {e}")
            continue

    return np.array(features_list)

def create_balanced_dataset(mask_paths: List[str],
                          no_mask_paths: List[str] = None,
                          augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create balanced dataset with mask and no-mask images

    Args:
        mask_paths: List of mask image paths
        no_mask_paths: List of no-mask image paths (if available)
        augmentation_factor: Augmentation factor for data augmentation

    Returns:
        Balanced features and labels
    """
    print("Creating balanced dataset...")

    # Analyze file sizes to separate classes
    file_sizes = []
    for path in mask_paths:
        size = os.path.getsize(path)
        file_sizes.append(size)

    file_sizes = np.array(file_sizes)

    # Use K-means to cluster images based on file size
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(file_sizes.reshape(-1, 1))

    # Separate images based on clusters
    cluster_0_indices = np.where(clusters == 0)[0]
    cluster_1_indices = np.where(clusters == 1)[0]

    # Determine which cluster is which class based on file size
    cluster_0_mean = np.mean(file_sizes[cluster_0_indices])
    cluster_1_mean = np.mean(file_sizes[cluster_1_indices])

    if cluster_0_mean < cluster_1_mean:
        # Cluster 0 = smaller files (likely no-mask), Cluster 1 = larger files (likely mask)
        no_mask_paths = [mask_paths[i] for i in cluster_0_indices]
        mask_paths_filtered = [mask_paths[i] for i in cluster_1_indices]
        print(f"Cluster 0 (No-mask): {len(no_mask_paths)} images, mean size: {cluster_0_mean:.0f} bytes")
        print(f"Cluster 1 (Mask): {len(mask_paths_filtered)} images, mean size: {cluster_1_mean:.0f} bytes")
    else:
        # Cluster 1 = smaller files (likely no-mask), Cluster 0 = larger files (likely mask)
        no_mask_paths = [mask_paths[i] for i in cluster_1_indices]
        mask_paths_filtered = [mask_paths[i] for i in cluster_0_indices]
        print(f"Cluster 0 (Mask): {len(mask_paths_filtered)} images, mean size: {cluster_0_mean:.0f} bytes")
        print(f"Cluster 1 (No-mask): {len(no_mask_paths)} images, mean size: {cluster_1_mean:.0f} bytes")

    # Create labels
    mask_labels = [1] * len(mask_paths_filtered)
    no_mask_labels = [0] * len(no_mask_paths)

    # Combine datasets
    all_paths = mask_paths_filtered + no_mask_paths
    all_labels = mask_labels + no_mask_labels

    # Shuffle data
    all_paths, all_labels = shuffle(all_paths, all_labels, random_state=42)

    print(f"Final dataset: {len(mask_paths_filtered)} mask images, {len(no_mask_paths)} no-mask images")

    # Create augmented dataset
    augmented_images, augmented_labels = create_augmented_dataset(
        all_paths, all_labels, augmentation_factor
    )

    # Extract HOG features
    print("Extracting HOG features from augmented images...")
    features = extract_hog_features_from_augmented(augmented_images)

    return features, augmented_labels

def train_stochastic_gradient_descent(X: np.ndarray,
                                    y: np.ndarray,
                                    test_size: float = 0.2,
                                    random_state: int = 42) -> Tuple[SGDClassifier, dict]:
    """
    Train stochastic gradient descent classifier with optimal parameters

    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Trained classifier and performance metrics
    """
    print("Training Stochastic Gradient Descent classifier...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize SGD classifier with optimal parameters
    sgd_classifier = SGDClassifier(
        loss='log_loss',               # Logistic regression loss
        learning_rate='adaptive',      # Adaptive learning rate
        eta0=0.01,                     # Initial learning rate
        max_iter=1000,                 # Maximum iterations
        tol=1e-3,                      # Tolerance for convergence
        alpha=0.01,                    # L2 regularization
        random_state=random_state,
        n_jobs=-1                      # Use all CPU cores
    )

    # Train the model
    sgd_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = sgd_classifier.predict(X_test)
    y_pred_proba = sgd_classifier.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(sgd_classifier, X, y, cv=5, scoring='accuracy')

    # Performance metrics
    metrics = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return sgd_classifier, metrics

def apply_pca_dimensionality_reduction(features: np.ndarray, n_components: float = 0.95) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA for dimensionality reduction

    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep (float for variance ratio, int for exact number)

    Returns:
        Reduced features and fitted PCA object
    """
    # Standardize features before PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)

    print(f"Original features: {features.shape[1]}")
    print(f"Reduced features: {features_reduced.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    return features_reduced, pca

def load_face_mask_dataset(dataset_path: str) -> List[str]:
    """
    Load face mask dataset from the specified path

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        List of image file paths
    """
    # Pattern to match face mask images
    pattern = os.path.join(dataset_path, "maksssksksss*.png")
    image_paths = glob.glob(pattern)

    print(f"Found {len(image_paths)} face mask images in {dataset_path}")

    # Sort paths for consistent ordering
    image_paths.sort()

    return image_paths

def feature_engineering_pipeline(image_paths: List[str],
                                 target_size: Tuple[int, int] = (64, 128),
                                 pca_variance: float = 0.95, extract_hog_features_batch=None) -> Tuple[np.ndarray, PCA]:
    """
    Complete feature engineering pipeline: HOG extraction + PCA reduction

    Args:
        image_paths: List of image file paths
        target_size: Target size for image resizing
        pca_variance: Variance ratio to preserve in PCA

    Returns:
        Reduced feature matrix and fitted PCA object
    """
    print("Step 1: Extracting HOG features with 20 orientation bins...")
    hog_features = extract_hog_features_batch(image_paths, target_size)

    print(f"Extracted HOG features shape: {hog_features.shape}")

    print("Step 2: Applying PCA for dimensionality reduction...")
    features_reduced, pca = apply_pca_dimensionality_reduction(hog_features, pca_variance)

    print("Feature engineering pipeline completed!")
    return features_reduced, pca

def complete_preprocessing_pipeline(dataset_path: str = r"C:\Users\ROG\Desktop\FaceMaskImage",
                                   augmentation_factor: int = 2,
                                   pca_variance: float = 0.95) -> Tuple[SGDClassifier, dict]:
    """
    Complete preprocessing and training pipeline

    Args:
        dataset_path: Path to dataset
        augmentation_factor: Data augmentation factor
        pca_variance: PCA variance to preserve

    Returns:
        Trained classifier and performance metrics
    """
    print("=== Complete Preprocessing and Training Pipeline ===")

    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    image_paths = load_face_mask_dataset(dataset_path)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    # Step 2: Create balanced dataset with augmentation
    print("Step 2: Creating balanced dataset with augmentation...")
    features, labels = create_balanced_dataset(image_paths, augmentation_factor=augmentation_factor)

    print(f"Augmented dataset shape: {features.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Step 3: Apply PCA for dimensionality reduction
    print("Step 3: Applying PCA dimensionality reduction...")
    features_reduced, _ = apply_pca_dimensionality_reduction(features, pca_variance)

    # Step 4: Train SGD classifier
    print("Step 4: Training Stochastic Gradient Descent...")
    classifier, metrics = train_stochastic_gradient_descent(features_reduced, labels)

    print("=== Pipeline Completed Successfully! ===")
    return classifier, metrics

def process_face_mask_dataset(dataset_path: str = r"C:\Users\ROG\Desktop\FaceMaskImage",
                             target_size: Tuple[int, int] = (64, 128),
                             pca_variance: float = 0.95) -> Tuple[np.ndarray, PCA]:
    """
    Complete pipeline to process the face mask dataset

    Args:
        dataset_path: Path to the face mask dataset
        target_size: Target size for image resizing
        pca_variance: Variance ratio to preserve in PCA

    Returns:
        Reduced feature matrix and fitted PCA object
    """
    print("Loading face mask dataset...")
    image_paths = load_face_mask_dataset(dataset_path)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    print(f"Starting feature engineering for {len(image_paths)} images...")
    features_reduced, pca = feature_engineering_pipeline(image_paths, target_size, pca_variance)

    return features_reduced, pca

# Example usage and testing
if __name__ == "__main__":
    try:
        # Run complete preprocessing and training pipeline
        classifier, metrics = complete_preprocessing_pipeline(
            augmentation_factor=2,  # Create 2 augmented versions per image
            pca_variance=0.95       # Preserve 95% variance
        )

        print("\n=== Complete Pipeline Results ===")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        # Save the trained model
        import joblib
        joblib.dump(classifier, 'face_mask_classifier.pkl')
        print("\nModel saved to 'face_mask_classifier.pkl'")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
