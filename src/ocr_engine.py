"""
OCR Engine Module for Smart OCR Bot
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self):
        """Initialize image preprocessor"""
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to improve OCR accuracy
        
        Parameters:
        image (np.ndarray): Input image
        
        Returns:
        np.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply dilation to connect text components
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        return dilated
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast
        
        Parameters:
        image (np.ndarray): Input image
        
        Returns:
        np.ndarray: Contrast-enhanced image
        """
        # Convert to PIL Image for easier processing
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(2.0)  # Increase contrast by 2x
        
        return np.array(enhanced)
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct image skew/rotation
        
        Parameters:
        image (np.ndarray): Input image
        
        Returns:
        np.ndarray: Deskewed image
        """
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

class TextDetector:
    def __init__(self):
        """Initialize text detector"""
        pass
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in image using EAST detector
        
        Parameters:
        image (np.ndarray): Input image
        
        Returns:
        list: List of bounding boxes (x, y, w, h)
        """
        # This is a simplified implementation
        # In practice, you would use EAST or CRAFT text detectors
        
        # For demonstration, we'll use a simple approach
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on size (adjust these thresholds as needed)
            if w > 10 and h > 10 and w < image.shape[1] and h < image.shape[0]:
                text_regions.append((x, y, w, h))
        
        return text_regions

class OCREngine:
    def __init__(self):
        """Initialize OCR engine"""
        self.preprocessor = ImagePreprocessor()
        self.text_detector = TextDetector()
        
        # Configure pytesseract
        # You may need to set the path to tesseract executable on Windows
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def extract_text(self, image: np.ndarray, lang: str = 'eng') -> str:
        """
        Extract text from image using Tesseract OCR
        
        Parameters:
        image (np.ndarray): Input image
        lang (str): Language code (default: 'eng')
        
        Returns:
        str: Extracted text
        """
        # Preprocess image
        processed = self.preprocessor.preprocess(image)
        
        # Apply additional enhancements
        enhanced = self.preprocessor.enhance_contrast(processed)
        deskewed = self.preprocessor.deskew(enhanced)
        
        # Extract text using pytesseract
        try:
            text = pytesseract.image_to_string(deskewed, lang=lang)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def extract_text_with_boxes(self, image: np.ndarray, lang: str = 'eng') -> List[Dict[str, Any]]:
        """
        Extract text with bounding boxes
        
        Parameters:
        image (np.ndarray): Input image
        lang (str): Language code (default: 'eng')
        
        Returns:
        list: List of text elements with bounding boxes
        """
        # Preprocess image
        processed = self.preprocessor.preprocess(image)
        
        # Detect text regions
        regions = self.text_detector.detect_text_regions(processed)
        
        # Extract text from each region
        text_elements = []
        for i, (x, y, w, h) in enumerate(regions):
            # Extract region
            region = processed[y:y+h, x:x+w]
            
            # Extract text
            try:
                text = pytesseract.image_to_string(region, lang=lang).strip()
                if text:  # Only add non-empty text
                    text_elements.append({
                        'text': text,
                        'bbox': (x, y, w, h),
                        'confidence': len(text)  # Simple confidence based on text length
                    })
            except Exception as e:
                print(f"OCR error for region {i}: {e}")
        
        return text_elements
    
    def extract_data(self, image: np.ndarray, lang: str = 'eng') -> Dict[str, Any]:
        """
        Extract detailed OCR data including text, boxes, and confidence
        
        Parameters:
        image (np.ndarray): Input image
        lang (str): Language code (default: 'eng')
        
        Returns:
        dict: Detailed OCR data
        """
        # Preprocess image
        processed = self.preprocessor.preprocess(image)
        
        # Extract detailed data using pytesseract
        try:
            data = pytesseract.image_to_data(processed, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Filter out empty text
            text_data = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:  # Only include non-empty text
                    text_data.append({
                        'text': text,
                        'bbox': (
                            data['left'][i], 
                            data['top'][i], 
                            data['width'][i], 
                            data['height'][i]
                        ),
                        'confidence': data['conf'][i]
                    })
            
            return {
                'text_elements': text_data,
                'full_text': ' '.join([item['text'] for item in text_data])
            }
        except Exception as e:
            print(f"OCR data extraction error: {e}")
            return {
                'text_elements': [],
                'full_text': ''
            }

class InformationExtractor:
    def __init__(self):
        """Initialize information extractor"""
        pass
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Parameters:
        text (str): Input text
        
        Returns:
        dict: Extracted entities
        """
        entities = {
            'dates': [],
            'amounts': [],
            'names': [],
            'addresses': [],
            'phone_numbers': [],
            'email_addresses': []
        }
        
        # Extract dates (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YY or MM-DD-YYYY
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YY or DD-MM-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(dates)
        
        # Extract monetary amounts
        amount_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
            r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b'  # 1000 USD
        ]
        
        for pattern in amount_patterns:
            amounts = re.findall(pattern, text)
            entities['amounts'].extend(amounts)
        
        # Extract phone numbers
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        entities['phone_numbers'] = re.findall(phone_pattern, text)
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['email_addresses'] = re.findall(email_pattern, text)
        
        # Extract potential names (simplified approach)
        # Look for capitalized words that might be names
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        potential_names = re.findall(name_pattern, text)
        # Filter out common words that are not names
        common_words = {'The', 'And', 'For', 'Are', 'But', 'Not', 'You', 'All', 'Can', 'Had', 'Her', 'Was', 'One', 'Our', 'Out', 'Day', 'Get', 'Has', 'Him', 'His', 'How', 'Its', 'May', 'New', 'Now', 'Old', 'See', 'Two', 'Who', 'Boy', 'Did', 'First', 'Girl', 'His', 'Its', 'Made', 'Make', 'Man', 'May', 'More', 'New', 'Now', 'Only', 'Our', 'Out', 'She', 'Some', 'Than', 'That', 'The', 'Their', 'Then', 'There', 'These', 'They', 'This', 'Time', 'Two', 'Was', 'Way', 'Well', 'What', 'When', 'Who', 'Will', 'With', 'Work', 'Year', 'Your'}
        entities['names'] = [name for name in potential_names if not any(word in name for word in common_words)]
        
        return entities

# Example usage
if __name__ == "__main__":
    # Create OCR engine
    ocr_engine = OCREngine()
    info_extractor = InformationExtractor()
    
    # Create a sample image with text (for demonstration)
    # In practice, you would load an actual document image
    sample_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.putText(sample_image, "Sample Invoice", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(sample_image, "Date: 01/15/2023", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(sample_image, "Amount: $1,250.00", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(sample_image, "John Smith", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Extract text
    extracted_text = ocr_engine.extract_text(sample_image)
    print("Extracted Text:")
    print(extracted_text)
    
    # Extract detailed data
    ocr_data = ocr_engine.extract_data(sample_image)
    print("\nDetailed OCR Data:")
    for item in ocr_data['text_elements']:
        print(f"Text: {item['text']}, Confidence: {item['confidence']}")
    
    # Extract entities
    entities = info_extractor.extract_entities(extracted_text)
    print("\nExtracted Entities:")
    for entity_type, values in entities.items():
        if values:
            print(f"{entity_type}: {values}")