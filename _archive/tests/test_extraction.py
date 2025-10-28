#!/usr/bin/env python3
"""
ROBUST term extraction with comprehensive testing
"""

import re

def extract_search_terms_v2(query: str) -> tuple[str, bool]:
    """
    Extract name/term from natural language query.
    Returns: (extracted_term, was_modified)
    
    Examples:
    - "phone of Noah" → ("Noah", True)
    - "phone number of Noah" → ("Noah", True)  
    - "contact info for Moishi Chen" → ("Moishi Chen", True)
    - "Noah" → ("Noah", False)
    - "phone of אבי אתרוגים" → ("אבי אתרוגים", True)
    """
    
    original = query
    cleaned = query
    
    # Phase 1: Remove leading filler phrases (more aggressive)
    leading_patterns = [
        # Phone variants
        r'^(what\'?s?\s+)?(the\s+)?phone\s+(number\s+)?(of|for)\s+',
        r'^(what\'?s?\s+)?(the\s+)?telephone\s+(number\s+)?(of|for)\s+',
        r'^(what\'?s?\s+)?(the\s+)?mobile\s+(number\s+)?(of|for)\s+',
        r'^(what\'?s?\s+)?(the\s+)?cell\s+(number\s+)?(of|for)\s+',
        
        # Contact/info variants
        r'^(what\'?s?\s+)?(the\s+)?contact\s+(info|information|details)\s+(of|for)\s+',
        r'^(what\'?s?\s+)?(the\s+)?email\s+(address\s+)?(of|for)\s+',
        r'^(what\'?s?\s+)?(the\s+)?e-mail\s+(address\s+)?(of|for)\s+',
        
        # Action verbs
        r'^(find|get|show|give\s+me|tell\s+me)\s+(the\s+)?(phone|contact|email|info)\s+(of|for)\s+',
        r'^(find|get|show|lookup|search\s+for)\s+',
        
        # Questions
        r'^(who\s+is|what\s+is|where\s+is)\s+',
        r'^(do\s+you\s+have|can\s+you\s+find)\s+(the\s+)?',
    ]
    
    for pattern in leading_patterns:
        new_cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        if new_cleaned != cleaned:
            cleaned = new_cleaned
            break  # Only apply first match
    
    # Phase 2: Remove trailing filler words
    trailing_patterns = [
        r'\s+(please|pls|thanks|thank\s+you)\.?$',
        r'[?.!]+$',  # Remove trailing punctuation
    ]
    
    for pattern in trailing_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Phase 3: Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Phase 4: Remove isolated articles/prepositions
    cleaned = re.sub(r'^(the|a|an|of|for)\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+(the|a|an|of|for)$', '', cleaned, flags=re.IGNORECASE)
    
    was_modified = (cleaned != original)
    return (cleaned if cleaned else original, was_modified)


# ============================================================================
# COMPREHENSIVE TESTS
# ============================================================================

def run_tests():
    test_cases = [
        # English
        ("phone of Noah", "Noah"),
        ("phone number of Noah", "Noah"),
        ("what's the phone of Noah", "Noah"),
        ("contact info for Moishi Chen", "Moishi Chen"),
        ("find Moishi", "Moishi"),
        ("Noah", "Noah"),
        ("get me contact details of Sarah", "Sarah"),
        ("email of David", "David"),
        
        # Hebrew
        ("phone of אבי אתרוגים", "אבי אתרוגים"),
        ("אבי אתרוגים", "אבי אתרוגים"),
        ("what's the phone number for אבי אתרוגים", "אבי אתרוגים"),
        ("phone of נעם", "נעם"),
        
        # Mixed
        ("phone of Noam please", "Noam"),
        ("find phone number for Moishi Chen?", "Moishi Chen"),
    ]
    
    print("🧪 Testing Term Extraction V2\n")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        extracted, was_modified = extract_search_terms_v2(query)
        status = "✅" if extracted == expected else "❌"
        
        if extracted == expected:
            passed += 1
        else:
            failed += 1
        
        mod_indicator = "📝" if was_modified else "  "
        
        print(f"\n{status} {mod_indicator} '{query}'")
        print(f"   Expected: '{expected}'")
        print(f"   Got:      '{extracted}'")
        
        if extracted != expected:
            print(f"   ⚠️  MISMATCH!")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed - review extraction logic")
