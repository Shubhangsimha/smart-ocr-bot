"""
Document Classifier Module for Smart OCR Bot
"""
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

class DocumentFeatureExtractor:
    def __init__(self):
        """Initialize document feature extractor"""
        pass
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from document image
        
        Parameters:
        image (np.ndarray): Document image
        
        Returns:
        np.ndarray: Feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        resized = cv2.resize(gray, (224, 224))
        
        # Extract basic features
        features = []
        
        # 1. Histogram features
        hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # 2. Texture features (Local Binary Pattern - simplified)
        lbp_features = self._extract_lbp_features(resized)
        features.extend(lbp_features)
        
        # 3. Edge features
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # 4. Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        avg_contour_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
        features.extend([num_contours, avg_contour_area])
        
        # 5. Structural features
        # Horizontal and vertical projections
        h_proj = np.sum(resized < 128, axis=0)
        v_proj = np.sum(resized < 128, axis=1)
        features.extend([
            np.mean(h_proj), np.std(h_proj),
            np.mean(v_proj), np.std(v_proj)
        ])
        
        return np.array(features)
    
    def _extract_lbp_features(self, image: np.ndarray, num_points: int = 8) -> List[float]:
        """
        Extract simplified LBP features
        
        Parameters:
        image (np.ndarray): Grayscale image
        num_points (int): Number of points for LBP
        
        Returns:
        list: LBP feature histogram
        """
        # Simplified LBP implementation
        lbp_image = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0
                
                # Check 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j-1],                   image[i, j+1],
                    image[i+1, j-1], image[i+1, j], image[i+1, j+1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp_image[i, j] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp_image.flatten(), bins=32, range=(0, 256))
        return hist.tolist()

class DocumentClassifier:
    def __init__(self):
        """Initialize document classifier"""
        self.feature_extractor = DocumentFeatureExtractor()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, images: List[np.ndarray], labels: List[str]):
        """
        Train the document classifier
        
        Parameters:
        images (list): List of document images
        labels (list): List of document type labels
        """
        print("Extracting features from training images...")
        features = []
        for image in images:
            feature_vector = self.feature_extractor.extract_features(image)
            features.append(feature_vector)
        
        features = np.array(features)
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print("Training classifier...")
        self.classifier.fit(features, encoded_labels)
        self.is_trained = True
        
        print("Training completed!")
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict document type
        
        Parameters:
        image (np.ndarray): Document image
        
        Returns:
        tuple: (predicted_label, confidence)
        """
        if not self.is_trained:
            # Return default prediction
            return "unknown", 0.5
        
        feature_vector = self.feature_extractor.extract_features(image)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.classifier.predict(feature_vector)[0]
        probabilities = self.classifier.predict_proba(feature_vector)[0]
        confidence = np.max(probabilities)
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_label, confidence
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Parameters:
        filepath (str): Path to save model
        """
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Parameters:
        filepath (str): Path to load model
        """
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    classifier = DocumentClassifier()
    
    # Create dummy training data
    # In practice, you would load actual document images
    sample_images = [
        np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8) for _ in range(20)
    ]
    sample_labels = ["invoice"] * 10 + ["contract"] * 5 + ["form"] * 5
    
    # Train classifier
    classifier.train(sample_images, sample_labels)
    
    # Save model
    classifier.save_model("document_classifier.pkl")
    
    # Test prediction
    test_image = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    predicted_label, confidence = classifier.predict(test_image)
    
    print(f"Predicted document type: {predicted_label} (confidence: {confidence:.2f})")