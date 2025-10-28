#!/usr/bin/env python3
"""
CRITICAL FIX: Extract search terms from natural language queries
"""

import re

def extract_search_terms(query: str) -> str:
    """
    Extract the actual search term from natural language queries.
    
    Examples:
    - "phone number of Noam" → "Noam"
    - "contact info for John Smith" → "John Smith"
    - "email of David" → "David"
    - "Moishi Chen" → "Moishi Chen" (unchanged)
    """
    
    # Remove common filler phrases (case insensitive)
    filler_patterns = [
        r'\b(phone|telephone|tel|mobile|cell)\s+(number|no\.?|#)?\s+(of|for)?\s*',
        r'\b(contact|info|information|details)\s+(of|for|about)?\s*',
        r'\be-?mail\s+(of|for|address)?\s*',
        r'\baddress\s+(of|for)?\s*',
        r'\b(find|get|show|give\s+me|tell\s+me)\s+(the)?\s*',
        r'\b(what|what\'s|whats)\s+(is|are)?\s+(the)?\s*',
        r'\bdo\s+you\s+have\s+(the)?\s*',
    ]
    
    cleaned = query
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Remove trailing prepositions and articles
    cleaned = re.sub(r'\s+(of|for|the|a|an)$', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned if cleaned else query


# Test
if __name__ == "__main__":
    test_cases = [
        "phone number of Noam",
        "contact info for Moishi Chen",
        "email of David",
        "what's the phone number for John Smith",
        "find Moishi",
        "Noam",
        "get me the contact details of Sarah",
    ]
    
    print("🧪 Testing search term extraction:\n")
    for test in test_cases:
        result = extract_search_terms(test)
        print(f"'{test}' → '{result}'")
