"""
Data Validator Module for Smart OCR Bot
"""
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd

class DataValidator:
    def __init__(self):
        """Initialize data validator"""
        pass
    
    def validate_date(self, date_str: str) -> Tuple[bool, str]:
        """
        Validate date format and check if it's a reasonable date
        
        Parameters:
        date_str (str): Date string to validate
        
        Returns:
        tuple: (is_valid, formatted_date)
        """
        # Common date formats
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y',
            '%d/%m/%Y', '%d-%m-%Y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%B %d, %Y', '%b %d, %Y',
            '%d %B %Y', '%d %b %Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # Check if date is reasonable (not in the future, not too old)
                if parsed_date.year < 1900 or parsed_date > datetime.now():
                    continue
                return True, parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return False, date_str
    
    def validate_amount(self, amount_str: str) -> Tuple[bool, float]:
        """
        Validate monetary amount
        
        Parameters:
        amount_str (str): Amount string to validate
        
        Returns:
        tuple: (is_valid, amount)
        """
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.]+', '', amount_str)
        
        try:
            amount = float(cleaned)
            # Check if amount is reasonable (positive, not excessively large)
            if amount < 0 or amount > 10000000:  # 10 million cap
                return False, amount
            return True, amount
        except ValueError:
            return False, 0.0
    
    def validate_email(self, email_str: str) -> Tuple[bool, str]:
        """
        Validate email address format
        
        Parameters:
        email_str (str): Email string to validate
        
        Returns:
        tuple: (is_valid, email)
        """
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = bool(re.match(email_pattern, email_str))
        return is_valid, email_str.lower() if is_valid else email_str
    
    def validate_phone(self, phone_str: str) -> Tuple[bool, str]:
        """
        Validate phone number format
        
        Parameters:
        phone_str (str): Phone number string to validate
        
        Returns:
        tuple: (is_valid, phone)
        """
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone_str)
        
        # Check if it's a valid length (10 or 11 digits)
        if len(digits_only) == 10:
            # Format as (XXX) XXX-XXXX
            formatted = f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
            return True, formatted
        elif len(digits_only) == 11 and digits_only[0] == '1':
            # Format as +1 (XXX) XXX-XXXX
            formatted = f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
            return True, formatted
        else:
            return False, phone_str
    
    def validate_document_structure(self, extracted_data: Dict[str, Any], 
                                  document_type: str) -> Dict[str, Any]:
        """
        Validate document structure based on document type
        
        Parameters:
        extracted_data (dict): Extracted data from document
        document_type (str): Type of document
        
        Returns:
        dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'confidence_score': 0.0
        }
        
        required_fields = self._get_required_fields(document_type)
        extracted_entities = extracted_data.get('entities', {})
        
        # Check required fields
        missing_fields = []
        for field in required_fields:
            if field not in extracted_entities or not extracted_entities[field]:
                missing_fields.append(field)
        
        if missing_fields:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate field formats
        for field_name, field_values in extracted_entities.items():
            for value in field_values:
                is_valid, validated_value = self._validate_field_format(field_name, value)
                if not is_valid:
                    validation_results['warnings'].append(
                        f"Invalid format for {field_name}: {value}"
                    )
        
        # Calculate confidence score
        validation_results['confidence_score'] = self._calculate_confidence_score(
            validation_results, len(required_fields), len(missing_fields)
        )
        
        return validation_results
    
    def _get_required_fields(self, document_type: str) -> List[str]:
        """
        Get required fields for a document type
        
        Parameters:
        document_type (str): Type of document
        
        Returns:
        list: Required fields
        """
        required_fields = {
            'invoice': ['dates', 'amounts', 'names'],
            'contract': ['dates', 'names', 'signatures'],
            'form': ['names', 'addresses'],
            'receipt': ['dates', 'amounts'],
            'unknown': []
        }
        
        return required_fields.get(document_type.lower(), [])
    
    def _validate_field_format(self, field_name: str, value: str) -> Tuple[bool, Any]:
        """
        Validate format of a specific field
        
        Parameters:
        field_name (str): Name of the field
        value (str): Value to validate
        
        Returns:
        tuple: (is_valid, validated_value)
        """
        if field_name == 'dates':
            return self.validate_date(value)
        elif field_name == 'amounts':
            is_valid, amount = self.validate_amount(value)
            return is_valid, amount
        elif field_name == 'email_addresses':
            return self.validate_email(value)
        elif field_name == 'phone_numbers':
            return self.validate_phone(value)
        else:
            # For other fields, basic validation
            return len(value.strip()) > 0, value
    
    def _calculate_confidence_score(self, validation_results: Dict[str, Any], 
                                  total_required: int, missing_count: int) -> float:
        """
        Calculate confidence score based on validation results
        
        Parameters:
        validation_results (dict): Validation results
        total_required (int): Total required fields
        missing_count (int): Number of missing fields
        
        Returns:
        float: Confidence score (0.0 to 1.0)
        """
        # Base score based on missing fields
        if total_required == 0:
            base_score = 1.0
        else:
            base_score = (total_required - missing_count) / total_required
        
        # Reduce score for errors
        error_penalty = len(validation_results['errors']) * 0.1
        
        # Reduce score for warnings
        warning_penalty = len(validation_results['warnings']) * 0.05
        
        # Calculate final score
        confidence_score = max(0.0, base_score - error_penalty - warning_penalty)
        
        return min(1.0, confidence_score)

class BusinessRuleValidator:
    def __init__(self):
        """Initialize business rule validator"""
        pass
    
    def validate_invoice(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate invoice-specific business rules
        
        Parameters:
        invoice_data (dict): Extracted invoice data
        
        Returns:
        dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'business_validation_passed': True
        }
        
        entities = invoice_data.get('entities', {})
        
        # Check if amounts are consistent
        amounts = entities.get('amounts', [])
        if len(amounts) >= 2:
            # Try to identify total amount (usually the largest)
            numeric_amounts = []
            for amount in amounts:
                is_valid, num_val = self._extract_numeric_amount(amount)
                if is_valid:
                    numeric_amounts.append(num_val)
            
            if len(numeric_amounts) >= 2:
                total_amount = max(numeric_amounts)
                other_amounts_sum = sum(numeric_amounts) - total_amount
                
                # Check if total matches sum of parts (within tolerance)
                tolerance = 0.01  # 1 cent tolerance
                if abs(total_amount - other_amounts_sum) > tolerance:
                    validation_results['warnings'].append(
                        f"Total amount (${total_amount:.2f}) doesn't match sum of parts (${other_amounts_sum:.2f})"
                    )
        
        # Check date consistency
        dates = entities.get('dates', [])
        if len(dates) >= 2:
            # Try to parse dates
            parsed_dates = []
            for date_str in dates:
                validator = DataValidator()
                is_valid, formatted_date = validator.validate_date(date_str)
                if is_valid:
                    try:
                        parsed_date = datetime.strptime(formatted_date, '%Y-%m-%d')
                        parsed_dates.append(parsed_date)
                    except ValueError:
                        continue
            
            if len(parsed_dates) >= 2:
                # Check if dates are in logical order
                sorted_dates = sorted(parsed_dates)
                if sorted_dates != parsed_dates:
                    validation_results['warnings'].append("Dates are not in chronological order")
        
        return validation_results
    
    def _extract_numeric_amount(self, amount_str: str) -> Tuple[bool, float]:
        """
        Extract numeric value from amount string
        
        Parameters:
        amount_str (str): Amount string
        
        Returns:
        tuple: (is_valid, numeric_value)
        """
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.]+', '', amount_str)
        
        try:
            amount = float(cleaned)
            return True, amount
        except ValueError:
            return False, 0.0
    
    def validate_contract(self, contract_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate contract-specific business rules
        
        Parameters:
        contract_data (dict): Extracted contract data
        
        Returns:
        dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'business_validation_passed': True
        }
        
        entities = contract_data.get('entities', {})
        
        # Check for signatures
        signatures = entities.get('signatures', [])
        if not signatures:
            validation_results['errors'].append("No signatures found in contract")
            validation_results['business_validation_passed'] = False
        
        # Check for party names
        names = entities.get('names', [])
        if len(names) < 2:
            validation_results['warnings'].append("Fewer than 2 party names found in contract")
        
        return validation_results

# Example usage
if __name__ == "__main__":
    # Create validators
    data_validator = DataValidator()
    business_validator = BusinessRuleValidator()
    
    # Test date validation
    test_dates = ["01/15/2023", "2023-01-15", "Jan 15, 2023", "invalid-date"]
    print("Date Validation Results:")
    for date in test_dates:
        is_valid, formatted = data_validator.validate_date(date)
        print(f"  {date}: {'Valid' if is_valid else 'Invalid'} -> {formatted}")
    
    # Test amount validation
    test_amounts = ["$1,250.00", "1500 USD", "invalid-amount"]
    print("\nAmount Validation Results:")
    for amount in test_amounts:
        is_valid, numeric = data_validator.validate_amount(amount)
        print(f"  {amount}: {'Valid' if is_valid else 'Invalid'} -> ${numeric:.2f}")
    
    # Test email validation
    test_emails = ["user@example.com", "invalid-email", "test@domain.co.uk"]
    print("\nEmail Validation Results:")
    for email in test_emails:
        is_valid, validated = data_validator.validate_email(email)
        print(f"  {email}: {'Valid' if is_valid else 'Invalid'} -> {validated}")
    
    # Test phone validation
    test_phones = ["123-456-7890", "(123) 456-7890", "+1 123-456-7890", "invalid-phone"]
    print("\nPhone Validation Results:")
    for phone in test_phones:
        is_valid, formatted = data_validator.validate_phone(phone)
        print(f"  {phone}: {'Valid' if is_valid else 'Invalid'} -> {formatted}")
    
    # Test document structure validation
    sample_invoice_data = {
        'entities': {
            'dates': ['01/15/2023'],
            'amounts': ['$1,250.00'],
            'names': ['John Smith']
        }
    }
    
    validation_result = data_validator.validate_document_structure(sample_invoice_data, 'invoice')
    print(f"\nDocument Structure Validation: {'Valid' if validation_result['is_valid'] else 'Invalid'}")
    print(f"Confidence Score: {validation_result['confidence_score']:.2f}")
    if validation_result['errors']:
        print("Errors:", validation_result['errors'])
    if validation_result['warnings']:
        print("Warnings:", validation_result['warnings'])