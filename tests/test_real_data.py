#!/usr/bin/env python3
"""
REAL DATA TEST SUITE - Based on contacts.csv Analysis
Tests against actual data patterns found in your contact database

Features tested:
- Real names from your data (David, Eli, Ben, תומר, etc.)
- Real keywords (טרמפ, עורך דין, מזרנים, etc.)
- Phone number formats (scientific notation handling)
- Hebrew/English mixed data
- Organization names
- Context display for search results

Run: python test_real_data.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    from vectoric_search import AdvancedVectorDBQASystem
except ImportError as e:
    print(f"❌ Error importing: {e}")
    sys.exit(1)


class TestTracker:
    """Track test results"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add(self, name: str, passed: bool, expected: str, actual: str, details: str = ""):
        result = {
            'name': name,
            'expected': expected,
            'actual': actual,
            'details': details
        }
        (self.passed if passed else self.failed).append(result)
        
        status = '✅' if passed else '❌'
        print(f"{status} {name}")
        if not passed:
            print(f"   Expected: {expected}")
            print(f"   Actual: {actual}")
        if details:
            print(f"   → {details}")
    
    def warn(self, message: str):
        self.warnings.append(message)
        print(f"⚠️  {message}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total: {total} | ✅ Passed: {len(self.passed)} | "
              f"❌ Failed: {len(self.failed)} | ⚠️  Warnings: {len(self.warnings)}")
        
        if self.failed:
            print("\n" + "="*80)
            print("FAILED TESTS")
            print("="*80)
            for test in self.failed:
                print(f"\n❌ {test['name']}")
                print(f"   Expected: {test['expected']}")
                print(f"   Actual: {test['actual']}")
                if test['details']:
                    print(f"   Details: {test['details']}")
        
        print("\n" + "="*80)
        if len(self.failed) == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {len(self.failed)} TEST(S) FAILED")
        print("="*80 + "\n")
        
        return len(self.failed) == 0


tracker = TestTracker()


def section(title: str):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_specific_contacts(qa: AdvancedVectorDBQASystem):
    """Test for specific contacts we know exist in contacts.csv"""
    section("TEST SUITE 1: SPECIFIC KNOWN CONTACTS")
    
    # These names appear in the CSV
    known_contacts = [
        ('David', 'David Rubinstein'),
        ('Ben', 'Ben Wittenberg'),
        ('Eli', 'Eli Huji C&C++'),
        ('Adir', 'Adir ערעור'),
        ('Noah', 'Noah Tradonsky'),
    ]
    
    for search_name, expected_name in known_contacts:
        results = qa.search(search_name, n_results=10)
        found = any(expected_name.lower() in r['metadata'].get('name', '').lower() 
                   for r in results.get('results', []))
        
        tracker.add(
            f"Find '{expected_name}'",
            found,
            f"Found in search results for '{search_name}'",
            "Found" if found else "Not found",
            f"Searched for '{search_name}'"
        )


def test_english_names_by_letter(qa: AdvancedVectorDBQASystem):
    """Test counting English names by first letter (real data)"""
    section("TEST SUITE 2: ENGLISH NAMES BY LETTER")
    
    # Letters that should have contacts based on CSV
    test_cases = [
        ('A', 1, ['Adir', 'ALINA', 'Avinoam', 'Ambulance']),  # Adir, ALINA, etc.
        ('B', 1, ['Ben', 'Burstin']),  # Ben Wittenberg, Burstin
        ('D', 1, ['David']),  # David Rubinstein - THE BUG!
        ('E', 1, ['Eli', 'Eden']),  # Eli, Eden Ajami
        ('N', 1, ['Noah', 'Noam', 'Nadav']),  # Noah, Noam, Nadav
        ('M', 1, ['Moishi', 'Michelle', 'Michal', 'Meir']),  # Multiple M names
    ]
    
    for letter, min_expected, examples in test_cases:
        # Get actual names
        names = qa.list_by_prefix(letter)
        count = len(names)
        
        # Check if any examples found
        found_examples = [ex for ex in examples if any(ex.lower() in n.lower() for n in names)]
        
        passed = count >= min_expected
        tracker.add(
            f"Letter '{letter}' count",
            passed,
            f"≥{min_expected} contact(s)",
            f"{count} contact(s)",
            f"Found: {', '.join(names[:3]) if names else 'None'}"
        )
        
        if names:
            print(f"   Examples: {', '.join(names[:5])}")


def test_hebrew_names_by_letter(qa: AdvancedVectorDBQASystem):
    """Test Hebrew names (תומר, תמר, etc. from CSV)"""
    section("TEST SUITE 3: HEBREW NAMES BY LETTER")
    
    # Hebrew letters that should have many contacts
    test_cases = [
        ('ת', 10, 'Many תומר, תמר contacts'),  # Most common in CSV
        ('א', 1, 'אבי, אלינה, etc.'),
        ('ד', 1, 'דניאל, etc.'),
        ('ש', 1, 'שריה, etc.'),
    ]
    
    for letter, min_expected, description in test_cases:
        names = qa.list_by_prefix(letter)
        count = len(names)
        
        passed = count >= min_expected
        tracker.add(
            f"Hebrew letter '{letter}' count",
            passed,
            f"≥{min_expected} contact(s)",
            f"{count} contact(s)",
            description
        )
        
        if names:
            print(f"   Examples: {', '.join(names[:5])}")


def test_language_distribution(qa: AdvancedVectorDBQASystem):
    """Test Hebrew vs English distribution"""
    section("TEST SUITE 4: LANGUAGE DISTRIBUTION")
    
    all_metas = qa._get_all_metadatas()
    total = len(all_metas)
    
    # Calculate actual counts (on-the-fly)
    hebrew_count = 0
    english_count = 0
    other_count = 0
    
    for meta in all_metas:
        name = meta.get('name', '').strip()
        if not name:
            other_count += 1
            continue
        
        first_char = None
        for ch in name:
            if ch.isalpha():
                first_char = ch
                break
        
        if not first_char:
            other_count += 1
        elif '\u0590' <= first_char <= '\u05FF':
            hebrew_count += 1
        elif first_char.isascii() and first_char.isalpha():
            english_count += 1
        else:
            other_count += 1
    
    print(f"Total contacts: {total}")
    print(f"Hebrew: {hebrew_count} | English: {english_count} | Other: {other_count}")
    
    # Test: Hebrew should be majority (based on CSV data)
    tracker.add(
        "Hebrew majority",
        hebrew_count > english_count,
        "Hebrew count > English count",
        f"Hebrew: {hebrew_count}, English: {english_count}",
        "Based on CSV data observation"
    )
    
    # Test: English should be significant (20+ contacts)
    tracker.add(
        "English contacts present",
        english_count >= 10,
        "≥10 English contacts",
        f"{english_count} English contacts",
        "David, Ben, Eli, Noah, etc."
    )
    
    # Test: Total adds up
    calculated_total = hebrew_count + english_count + other_count
    tracker.add(
        "Total count validation",
        calculated_total == total,
        f"{total} total",
        f"{calculated_total} calculated",
        "Sum of categories equals total"
    )


def test_keyword_search_tremp(qa: AdvancedVectorDBQASystem):
    """Test keyword 'טרמפ' (ride/tremp) - appears in CSV"""
    section("TEST SUITE 5: KEYWORD SEARCH - טרמפ (RIDE/TREMP)")
    
    keyword = 'טרמפ'
    results = qa.search_full_text(keyword, limit=20)
    
    print(f"Searching for: '{keyword}'")
    print(f"Results found: {len(results)}")
    
    # Test 5.1: Results found
    tracker.add(
        f"Keyword '{keyword}' - Results found",
        len(results) > 0,
        "≥1 result",
        f"{len(results)} result(s)",
        "rina rogers טרמפ, תמר טרמפ appear in CSV"
    )
    
    if results:
        # Test 5.2: Context provided
        first_result = results[0]
        has_context = 'keyword_context' in first_result and first_result['keyword_context']
        
        tracker.add(
            f"Keyword '{keyword}' - Context provided",
            has_context,
            "Context field present with data",
            "Context present" if has_context else "Context missing",
            "Shows WHERE keyword was found"
        )
        
        # Test 5.3: Keyword visible in context
        if has_context:
            context = first_result['keyword_context']
            keyword_in_context = keyword in context
            
            tracker.add(
                f"Keyword '{keyword}' - Visible in context",
                keyword_in_context,
                f"'{keyword}' appears in context",
                "Present" if keyword_in_context else "Missing",
                ""
            )
            
            # Show result
            print(f"\n  📋 Example Result:")
            print(f"     Name: {first_result['name']}")
            print(f"     Context: {context[:150]}...")
        
        # Show first 3 results
        print(f"\n  Found contacts:")
        for r in results[:3]:
            print(f"    - {r['name']}")


def test_keyword_search_lawyer(qa: AdvancedVectorDBQASystem):
    """Test keyword 'עורך דין' (lawyer) - Burstin appears in CSV"""
    section("TEST SUITE 6: KEYWORD SEARCH - עורך דין (LAWYER)")
    
    keyword = 'עורך דין'
    results = qa.search_full_text(keyword, limit=20)
    
    print(f"Searching for: '{keyword}'")
    print(f"Results found: {len(results)}")
    
    tracker.add(
        f"Keyword '{keyword}' - Results found",
        len(results) > 0,
        "≥1 result",
        f"{len(results)} result(s)",
        "Burstin (עורך הדין של אדון פרדריק) in CSV"
    )
    
    if results:
        has_context = 'keyword_context' in results[0] and results[0]['keyword_context']
        tracker.add(
            f"Keyword '{keyword}' - Context provided",
            has_context,
            "Context field with data",
            "Present" if has_context else "Missing",
            ""
        )
        
        print(f"\n  Found contacts:")
        for r in results[:3]:
            name = r['name']
            context = r.get('keyword_context', '')[:100]
            print(f"    - {name}")
            if context and keyword in context:
                print(f"      Context: ...{context}...")


def test_organization_search(qa: AdvancedVectorDBQASystem):
    """Test searching organization names from CSV"""
    section("TEST SUITE 7: ORGANIZATION NAME SEARCH")
    
    # Organizations that appear in CSV
    orgs = [
        ('ALINA', 'ALINA אלינה מזרנים'),  # Mattress company
        ('Job Breach', 'Job Breach'),
        ('Fresh Gifts', 'Fresh Gifts'),
    ]
    
    for search_term, expected in orgs:
        results = qa.search(search_term, n_results=5)
        found = any(expected.lower() in r['metadata'].get('name', '').lower() 
                   for r in results.get('results', []))
        
        tracker.add(
            f"Organization '{expected}'",
            found,
            f"Found '{expected}'",
            "Found" if found else "Not found",
            f"Searched for '{search_term}'"
        )


def test_phone_number_formats(qa: AdvancedVectorDBQASystem):
    """Test that phone numbers are readable (not scientific notation)"""
    section("TEST SUITE 8: PHONE NUMBER FORMAT")
    
    # Get some contacts with phones
    all_metas = qa._get_all_metadatas()
    
    phone_fields = []
    for meta in all_metas[:50]:  # Check first 50
        for key in meta:
            if 'phone' in key.lower() and meta[key]:
                phone_fields.append((meta.get('name', 'Unknown'), meta[key]))
    
    if not phone_fields:
        tracker.warn("No phone numbers found in first 50 contacts")
        return
    
    print(f"Checking {len(phone_fields)} phone numbers...")
    
    # Test: Phone numbers shouldn't contain 'E+' (scientific notation)
    scientific_notation_count = sum(1 for _, phone in phone_fields if 'e+' in str(phone).lower())
    
    tracker.add(
        "Phone numbers readable",
        scientific_notation_count == 0,
        "0 phones in scientific notation",
        f"{scientific_notation_count} phones in scientific notation",
        "Should be converted during ingestion"
    )
    
    # Show sample phone numbers
    print(f"\n  Sample phone numbers:")
    for name, phone in phone_fields[:5]:
        print(f"    {name}: {phone}")
    
    if scientific_notation_count > 0:
        tracker.warn(f"Found {scientific_notation_count} phone numbers in scientific notation - needs fixing!")


def test_metadata_consistency(qa: AdvancedVectorDBQASystem):
    """Test first_char metadata vs actual names"""
    section("TEST SUITE 9: METADATA CONSISTENCY")
    
    all_metas = qa._get_all_metadatas()
    
    missing_first_char = 0
    wrong_first_char = 0
    
    for meta in all_metas:
        name = meta.get('name', '').strip()
        cached_first = meta.get('first_char', '').strip()
        
        if name:
            # Get actual first alphabetic char
            actual_first = None
            for ch in name:
                if ch.isalpha():
                    actual_first = ch.upper() if ch.isascii() else ch
                    break
            
            if not cached_first:
                missing_first_char += 1
            elif actual_first and cached_first.upper() != actual_first.upper():
                wrong_first_char += 1
    
    total = len(all_metas)
    print(f"Total contacts: {total}")
    print(f"Missing first_char: {missing_first_char}")
    print(f"Wrong first_char: {wrong_first_char}")
    
    tracker.add(
        "first_char metadata completeness",
        missing_first_char == 0,
        "0 missing first_char",
        f"{missing_first_char} missing",
        "All contacts should have first_char"
    )
    
    tracker.add(
        "first_char metadata accuracy",
        wrong_first_char == 0,
        "0 wrong first_char",
        f"{wrong_first_char} wrong",
        "Cached first_char should match actual name"
    )
    
    if missing_first_char > 0 or wrong_first_char > 0:
        tracker.warn("Metadata cache issues detected! This causes counting bugs.")


def test_emergency_contacts(qa: AdvancedVectorDBQASystem):
    """Test emergency contacts (Police, Ambulance, Fire) from CSV"""
    section("TEST SUITE 10: EMERGENCY CONTACTS")
    
    emergency_contacts = [
        ('Police', '100'),
        ('Ambulance', '101'),
        ('Fire', '102'),
    ]
    
    for name, expected_number in emergency_contacts:
        results = qa.search(name, n_results=5)
        found = any(name.lower() in r['metadata'].get('name', '').lower() 
                   for r in results.get('results', []))
        
        tracker.add(
            f"Emergency contact '{name}'",
            found,
            f"Found '{name}'",
            "Found" if found else "Not found",
            f"Should have phone {expected_number}"
        )


def main():
    print("\n" + "="*80)
    print("  🔬 REAL DATA TEST SUITE")
    print("  Based on actual contacts.csv content")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        print("Initializing VectorDB system...")
        qa = AdvancedVectorDBQASystem()
        
        total = qa.collection.count()
        print(f"✅ Database loaded: {total:,} contacts\n")
        
        if total == 0:
            print("❌ Database is empty! Load contacts.csv first:")
            print("   python vectoric_search.py")
            print("   > load contacts.csv")
            return 1
        
        # Run all test suites
        test_specific_contacts(qa)
        test_english_names_by_letter(qa)
        test_hebrew_names_by_letter(qa)
        test_language_distribution(qa)
        test_keyword_search_tremp(qa)
        test_keyword_search_lawyer(qa)
        test_organization_search(qa)
        test_phone_number_formats(qa)
        test_metadata_consistency(qa)
        test_emergency_contacts(qa)
        
        # Summary
        all_passed = tracker.summary()
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
