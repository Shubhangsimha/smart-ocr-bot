"""Main script for 10-smart-ocr-bot"""
import argparse
import os
import sys
import json
from typing import List, Dict, Any
import cv2
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from src.document_classifier import DocumentClassifier
from src.ocr_engine import OCREngine, InformationExtractor
from src.data_validator import DataValidator, BusinessRuleValidator

def load_document_image(image_path: str) -> np.ndarray:
    """
    Load document image from file
    
    Parameters:
    image_path (str): Path to image file
    
    Returns:
    np.ndarray: Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image

def process_single_document(image_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Process a single document
    
    Parameters:
    image_path (str): Path to document image
    output_dir (str): Output directory for results
    
    Returns:
    dict: Processing results
    """
    print(f"Processing document: {image_path}")
    
    # Load image
    try:
        image = load_document_image(image_path)
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image_path': image_path
        }
    
    # Initialize components
    classifier = DocumentClassifier()
    ocr_engine = OCREngine()
    info_extractor = InformationExtractor()
    data_validator = DataValidator()
    business_validator = BusinessRuleValidator()
    
    # Classify document type
    print("Classifying document type...")
    doc_type, confidence = classifier.predict(image)
    print(f"Document type: {doc_type} (confidence: {confidence:.2f})")
    
    # Extract text
    print("Extracting text...")
    ocr_data = ocr_engine.extract_data(image)
    full_text = ocr_data['full_text']
    text_elements = ocr_data['text_elements']
    
    print(f"Extracted {len(text_elements)} text elements")
    
    # Extract entities
    print("Extracting entities...")
    entities = info_extractor.extract_entities(full_text)
    
    # Count extracted entities
    entity_count = sum(len(values) for values in entities.values())
    print(f"Extracted {entity_count} entities")
    
    # Validate data
    print("Validating data...")
    validation_result = data_validator.validate_document_structure(
        {'entities': entities}, doc_type
    )
    
    # Apply business rules validation
    print("Applying business rules validation...")
    if doc_type.lower() == 'invoice':
        business_validation = business_validator.validate_invoice({'entities': entities})
    elif doc_type.lower() == 'contract':
        business_validation = business_validator.validate_contract({'entities': entities})
    else:
        business_validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'business_validation_passed': True
        }
    
    # Combine validation results
    overall_validation = {
        'document_validation': validation_result,
        'business_validation': business_validation
    }
    
    # Prepare results
    results = {
        'success': True,
        'image_path': image_path,
        'document_type': doc_type,
        'document_confidence': confidence,
        'extracted_text': full_text,
        'text_elements': text_elements,
        'entities': entities,
        'validation': overall_validation
    }
    
    # Save results if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(image_path))[0]}_results.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {result_file}")
    
    return results

def process_batch_documents(input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Process a batch of documents
    
    Parameters:
    input_dir (str): Directory containing document images
    output_dir (str): Output directory for results
    
    Returns:
    list: List of processing results
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Find all image files
    image_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        print(f"No supported image files found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} document images to process")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcessing document {i}/{len(image_files)}")
        try:
            result = process_single_document(image_path, output_dir)
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'image_path': image_path
            })
    
    return results

def main():
    print("Starting 10-smart-ocr-bot: Smart OCR Bot for Document Automation...")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Smart OCR Bot for Document Automation")
    parser.add_argument("--input_path", type=str, help="Path to single document image")
    parser.add_argument("--input_dir", type=str, help="Directory containing document images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Process documents
    if args.input_path:
        # Process single document
        try:
            result = process_single_document(args.input_path, args.output_dir)
            if result['success']:
                print("\nProcessing completed successfully!")
                print(f"Document type: {result['document_type']}")
                print(f"Confidence: {result['document_confidence']:.2f}")
                print(f"Entities extracted: {sum(len(v) for v in result['entities'].values())}")
                
                # Print validation results
                doc_validation = result['validation']['document_validation']
                business_validation = result['validation']['business_validation']
                
                print(f"\nDocument Validation: {'Passed' if doc_validation['is_valid'] else 'Failed'}")
                print(f"Confidence Score: {doc_validation['confidence_score']:.2f}")
                
                if doc_validation['errors']:
                    print("Errors:")
                    for error in doc_validation['errors']:
                        print(f"  - {error}")
                
                if doc_validation['warnings']:
                    print("Warnings:")
                    for warning in doc_validation['warnings']:
                        print(f"  - {warning}")
                
                print(f"\nBusiness Validation: {'Passed' if business_validation['business_validation_passed'] else 'Failed'}")
                
                if business_validation['errors']:
                    print("Business Errors:")
                    for error in business_validation['errors']:
                        print(f"  - {error}")
                
                if business_validation['warnings']:
                    print("Business Warnings:")
                    for warning in business_validation['warnings']:
                        print(f"  - {warning}")
            else:
                print(f"Processing failed: {result['error']}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.input_dir:
        # Process batch of documents
        try:
            results = process_batch_documents(args.input_dir, args.output_dir)
            
            # Summary statistics
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            print(f"\nBatch processing completed!")
            print(f"Total documents: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            # Save batch summary
            summary_file = os.path.join(args.output_dir, "batch_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Batch results saved to: {summary_file}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        # Demonstration mode
        print("Running demonstration...")
        print("Please provide --input_path or --input_dir to process documents.")
        print("\nExample usage:")
        print("  python main.py --input_path data/sample_invoice.jpg")
        print("  python main.py --input_dir data/invoices/ --output_dir results/")

if __name__ == "__main__":
    main()