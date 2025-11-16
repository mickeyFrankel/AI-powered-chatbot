#!/usr/bin/env python3
"""
Comprehensive Test Suite for VectorDB Chatbot
Tests all critical functionality: counting, searching, and result presentation

Run: python test_comprehensive.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vectoric_search import AdvancedVectorDBQASystem
except ImportError as e:
    print(f"❌ Error importing vectoric_search: {e}")
    sys.exit(1)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.tests = []
    
    def add(self, name: str, passed: bool, expected: str, actual: str, details: str = ""):
        self.tests.append({
            'name': name,
            'passed': passed,
            'expected': expected,
            'actual': actual,
            'details': details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed > 0:
            print("\n" + "="*80)
            print("FAILED TESTS")
            print("="*80)
            for test in self.tests:
                if not test['passed']:
                    print(f"\n❌ {test['name']}")
                    print(f"   Expected: {test['expected']}")
                    print(f"   Actual: {test['actual']}")
                    if test['details']:
                        print(f"   Details: {test['details']}")
        
        print("\n" + "="*80)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print("⚠️  SOME TESTS FAILED - SEE DETAILS ABOVE")
        print("="*80 + "\n")


results = TestResults()


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_counting_by_language(qa_system: AdvancedVectorDBQASystem):
    """Test counting Hebrew vs English contacts"""
    print_section("TEST SUITE 1: COUNTING BY LANGUAGE")
    
    all_metas = qa_system._get_all_metadatas()
    total = len(all_metas)
    
    # Method 1: Using cached first_char (OLD/BUGGY WAY)
    cached_hebrew = 0
    cached_english = 0
    for meta in all_metas:
        first_char = meta.get('first_char', '')
        if first_char and '\u0590' <= first_char <= '\u05FF':
            cached_hebrew += 1
        elif first_char and first_char.isascii() and first_char.isalpha():
            cached_english += 1
    
    # Method 2: Calculate from actual names (CORRECT WAY)
    actual_hebrew = 0
    actual_english = 0
    actual_other = 0
    
    for meta in all_metas:
        name = meta.get('name', '').strip()
        if not name:
            actual_other += 1
            continue
        
        first_char = None
        for ch in name:
            if ch.isalpha():
                first_char = ch
                break
        
        if not first_char:
            actual_other += 1
        elif '\u0590' <= first_char <= '\u05FF':
            actual_hebrew += 1
        elif first_char.isascii() and first_char.isalpha():
            actual_english += 1
        else:
            actual_other += 1
    
    print(f"Total contacts: {total}")
    print(f"\nMethod 1 (cached metadata):")
    print(f"  Hebrew: {cached_hebrew}")
    print(f"  English: {cached_english}")
    print(f"\nMethod 2 (actual names):")
    print(f"  Hebrew: {actual_hebrew}")
    print(f"  English: {actual_english}")
    print(f"  Other: {actual_other}")
    
    # Test 1.1: Hebrew count accuracy
    test_name = "Hebrew Count Accuracy"
    passed = cached_hebrew == actual_hebrew
    results.add(
        test_name,
        passed,
        f"{actual_hebrew} Hebrew contacts",
        f"{cached_hebrew} Hebrew contacts (cached)",
        "Cached metadata matches actual names" if passed else "Metadata cache is stale/wrong"
    )
    print(f"\n{'✅' if passed else '❌'} {test_name}: {cached_hebrew} (cached) vs {actual_hebrew} (actual)")
    
    # Test 1.2: English count accuracy
    test_name = "English Count Accuracy"
    passed = cached_english == actual_english
    results.add(
        test_name,
        passed,
        f"{actual_english} English contacts",
        f"{cached_english} English contacts (cached)",
        "Cached metadata matches actual names" if passed else "Metadata cache is stale/wrong"
    )
    print(f"{'✅' if passed else '❌'} {test_name}: {cached_english} (cached) vs {actual_english} (actual)")
    
    # Test 1.3: Total validation
    test_name = "Total Count Validation"
    passed = (actual_hebrew + actual_english + actual_other) == total
    results.add(
        test_name,
        passed,
        f"{total} total contacts",
        f"{actual_hebrew + actual_english + actual_other} counted",
        "Sum of categories matches total"
    )
    print(f"{'✅' if passed else '❌'} {test_name}")


def test_counting_by_prefix(qa_system: AdvancedVectorDBQASystem):
    """Test counting contacts by first letter"""
    print_section("TEST SUITE 2: COUNTING BY PREFIX")
    
    # Test specific letters that might have issues
    test_letters = ['A', 'B', 'D', 'M', 'ש', 'א', 'ד']
    
    for letter in test_letters:
        # Method 1: Using cached metadata
        all_metas = qa_system._get_all_metadatas()
        cached_count = sum(1 for md in all_metas 
                          if (md.get('first_char', '') or '').upper() == letter.upper())
        
        # Method 2: Checking actual names
        actual_contacts = []
        for meta in all_metas:
            name = meta.get('name', '').strip()
            if name and name.upper().startswith(letter.upper()):
                actual_contacts.append(name)
        actual_count = len(actual_contacts)
        
        # Test
        test_name = f"Count by Prefix '{letter}'"
        passed = cached_count == actual_count
        results.add(
            test_name,
            passed,
            f"{actual_count} contacts starting with '{letter}'",
            f"{cached_count} contacts (cached)",
            f"Samples: {', '.join(actual_contacts[:3])}" if actual_contacts else "No contacts found"
        )
        
        status = '✅' if passed else '❌'
        print(f"{status} Letter '{letter}': {cached_count} (cached) vs {actual_count} (actual)")
        if actual_contacts[:3]:
            print(f"   Examples: {', '.join(actual_contacts[:3])}")


def test_counting_by_length(qa_system: AdvancedVectorDBQASystem):
    """Test counting names by length"""
    print_section("TEST SUITE 3: COUNTING BY LENGTH")
    
    # Test specific lengths
    test_lengths = [3, 5, 10, 15]
    
    for length in test_lengths:
        # Get names of exact length
        names = qa_system.names_by_length(length, limit=100)
        count = len(names)
        
        # Validate by recounting
        all_names = qa_system.get_all_names()
        actual_count = sum(1 for name in all_names if len(name) == length)
        
        test_name = f"Count by Length {length}"
        passed = count == actual_count
        results.add(
            test_name,
            passed,
            f"{actual_count} names with {length} chars",
            f"{count} names returned",
            f"Samples: {', '.join(names[:3])}" if names else "No names found"
        )
        
        status = '✅' if passed else '❌'
        print(f"{status} Length {length}: {count} names")
        if names[:3]:
            print(f"   Examples: {', '.join(names[:3])}")


def test_search_keyword_context(qa_system: AdvancedVectorDBQASystem):
    """Test that keyword search shows WHERE the keyword was found"""
    print_section("TEST SUITE 4: KEYWORD SEARCH CONTEXT")
    
    # Test keywords that should exist in the database
    test_keywords = [
        ('curtain', 'Should find curtain salesmen/companies'),
        ('doctor', 'Should find medical professionals'),
        ('רופא', 'Should find doctors (Hebrew)'),
        ('עורך דין', 'Should find lawyers (Hebrew)'),
    ]
    
    for keyword, description in test_keywords:
        results_list = qa_system.search_full_text(keyword, limit=10)
        
        # Test 4.1: Results found
        test_name = f"Keyword '{keyword}' - Results Found"
        passed = len(results_list) > 0
        results.add(
            test_name,
            passed,
            f"Found some results for '{keyword}'",
            f"Found {len(results_list)} results",
            description
        )
        
        status = '✅' if passed else '❌'
        print(f"\n{status} Keyword '{keyword}': {len(results_list)} results")
        
        if results_list:
            # Test 4.2: Context is provided
            first_result = results_list[0]
            has_context = 'keyword_context' in first_result
            
            test_name = f"Keyword '{keyword}' - Context Provided"
            results.add(
                test_name,
                has_context,
                "Context field present in results",
                "Context field present" if has_context else "Context field missing",
                ""
            )
            
            status = '✅' if has_context else '❌'
            print(f"  {status} Context provided: {has_context}")
            
            # Test 4.3: Keyword appears in context
            if has_context:
                context = first_result['keyword_context']
                keyword_in_context = keyword.lower() in context.lower()
                
                test_name = f"Keyword '{keyword}' - Appears in Context"
                results.add(
                    test_name,
                    keyword_in_context,
                    f"'{keyword}' visible in context",
                    f"Context: {context[:100]}...",
                    ""
                )
                
                status = '✅' if keyword_in_context else '❌'
                print(f"  {status} Keyword in context: {keyword_in_context}")
                
                # Show first result
                print(f"\n  📋 Example Result:")
                print(f"     Name: {first_result['name']}")
                print(f"     Context: {context[:120]}...")
                
                # Show metadata
                phone = first_result['metadata'].get('phone', 'N/A')
                if phone and phone != 'N/A':
                    print(f"     Phone: {phone}")


def test_semantic_search(qa_system: AdvancedVectorDBQASystem):
    """Test semantic/similarity search"""
    print_section("TEST SUITE 5: SEMANTIC SEARCH")
    
    # Test queries
    test_queries = [
        ('plumber', 5, 'Should find plumbers/שרברב'),
        ('lawyer', 5, 'Should find lawyers/עורך דין'),
        ('דוד', 3, 'Should find uncle or name David'),
    ]
    
    for query, expected_min, description in test_queries:
        search_results = qa_system.search(query, n_results=10)
        count = len(search_results.get('results', []))
        
        test_name = f"Semantic Search '{query}'"
        passed = count >= expected_min
        results.add(
            test_name,
            passed,
            f"≥{expected_min} results",
            f"{count} results found",
            description
        )
        
        status = '✅' if passed else '❌'
        print(f"{status} Query '{query}': {count} results (expected ≥{expected_min})")
        
        if search_results.get('results'):
            first = search_results['results'][0]
            print(f"   Top result: {first['metadata'].get('name', 'Unknown')}")
            print(f"   Similarity: {first.get('similarity_score', 0):.3f}")


def test_list_by_prefix_tool(qa_system: AdvancedVectorDBQASystem):
    """Test the list_by_prefix tool returns counts AND names"""
    print_section("TEST SUITE 6: LIST BY PREFIX TOOL")
    
    test_letters = ['A', 'D', 'M']
    
    for letter in test_letters:
        # Call the tool (this is what the agent uses)
        tool_result = qa_system.list_by_prefix(letter)
        names = tool_result if isinstance(tool_result, list) else []
        
        # Validate
        actual_names = [name for name in qa_system.get_all_names() 
                       if name.upper().startswith(letter.upper())]
        
        test_name = f"list_by_prefix('{letter}') Returns Names"
        passed = len(names) == len(actual_names)
        results.add(
            test_name,
            passed,
            f"{len(actual_names)} names starting with '{letter}'",
            f"{len(names)} names returned",
            f"Samples: {', '.join(actual_names[:3])}" if actual_names else ""
        )
        
        status = '✅' if passed else '❌'
        print(f"{status} Letter '{letter}': {len(names)} names returned")
        if names[:3]:
            print(f"   Examples: {', '.join(names[:3])}")


def test_data_integrity(qa_system: AdvancedVectorDBQASystem):
    """Test overall data integrity"""
    print_section("TEST SUITE 7: DATA INTEGRITY")
    
    all_metas = qa_system._get_all_metadatas()
    total = len(all_metas)
    
    # Test 7.1: All contacts have names
    missing_names = sum(1 for md in all_metas if not md.get('name', '').strip())
    test_name = "All Contacts Have Names"
    passed = missing_names == 0
    results.add(
        test_name,
        passed,
        "0 contacts missing names",
        f"{missing_names} contacts missing names",
        f"Out of {total} total contacts"
    )
    print(f"{'✅' if passed else '❌'} Missing names: {missing_names}/{total}")
    
    # Test 7.2: first_char metadata consistency
    inconsistent = 0
    for meta in all_metas:
        name = meta.get('name', '').strip()
        cached_first = meta.get('first_char', '').strip()
        
        if name:
            actual_first = None
            for ch in name:
                if ch.isalpha():
                    actual_first = ch.upper() if ch.isascii() else ch
                    break
            
            if cached_first and actual_first and cached_first.upper() != actual_first.upper():
                inconsistent += 1
    
    test_name = "first_char Metadata Consistency"
    passed = inconsistent == 0
    results.add(
        test_name,
        passed,
        "0 inconsistent first_char values",
        f"{inconsistent} inconsistent first_char values",
        "Cached first_char matches actual first character"
    )
    print(f"{'✅' if passed else '❌'} Inconsistent first_char: {inconsistent}/{total}")
    
    if inconsistent > 0:
        results.warnings += 1
        print(f"   ⚠️  WARNING: Metadata cache needs refresh!")


def main():
    print("\n" + "="*80)
    print("  🔍 COMPREHENSIVE CHATBOT TEST SUITE")
    print("  Testing: Counting, Searching, Context, Data Integrity")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Initialize system
        print("Initializing VectorDB system...")
        qa_system = AdvancedVectorDBQASystem()
        
        total_contacts = qa_system.collection.count()
        print(f"✅ Loaded database with {total_contacts:,} contacts\n")
        
        if total_contacts == 0:
            print("❌ Error: Database is empty. Please load data first.")
            return
        
        # Run test suites
        test_counting_by_language(qa_system)
        test_counting_by_prefix(qa_system)
        test_counting_by_length(qa_system)
        test_search_keyword_context(qa_system)
        test_semantic_search(qa_system)
        test_list_by_prefix_tool(qa_system)
        test_data_integrity(qa_system)
        
        # Print summary
        results.print_summary()
        
        # Exit code
        sys.exit(0 if results.failed == 0 else 1)
        
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
