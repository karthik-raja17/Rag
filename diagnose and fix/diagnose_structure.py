#!/usr/bin/env python3
"""
DIAGNOSTIC: Examine actual cache file structure
Shows what patterns exist in your specific PDF
"""
import os
import sys
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import CACHE_DIR

def analyze_definitions_section(text):
    """Check how definitions are formatted"""
    print("\n" + "="*80)
    print("🔬 DEFINITIONS ANALYSIS")
    print("="*80)
    
    # Find the definitions section
    lines = text.split('\n')
    in_definitions = False
    definitions_start = -1
    definitions_end = -1
    
    for i, line in enumerate(lines):
        if re.match(r'##\s*1\.?\s*DEFINITIONS', line.strip(), re.IGNORECASE):
            in_definitions = True
            definitions_start = i
            print(f"\n✅ Found DEFINITIONS section at line {i}")
            print(f"   Header: {line.strip()}")
        elif in_definitions and re.match(r'##\s*\d+', line.strip()):
            definitions_end = i
            break
    
    if definitions_start == -1:
        print("\n❌ No DEFINITIONS section found!")
        return
    
    if definitions_end == -1:
        definitions_end = len(lines)
    
    # Extract definitions section
    def_section = lines[definitions_start:definitions_end]
    
    print(f"\n📏 Definitions section spans {len(def_section)} lines")
    
    # Look for definition patterns
    print(f"\n🔍 Searching for definition patterns...")
    
    patterns = [
        (r'["\"]([^"\"]+)["\"]?\s+means', 'Standard: "Term" means'),
        (r'^\s*\([a-z]\)\s+["\"]([^"\"]+)["\"]', 'Lettered: (a) "Term"'),
        (r'^\s*\d+\.\d+\s+["\"]([^"\"]+)["\"]', 'Numbered: 1.1 "Term"'),
        (r'^\s*["\']([A-Z][^"\']+)["\']:', 'Quoted: "Term":'),
    ]
    
    matches_by_pattern = {}
    for pattern, name in patterns:
        matches = []
        for i, line in enumerate(def_section[:100]):  # First 100 lines
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                matches.append((i + definitions_start, line.strip(), match.group(1)))
        matches_by_pattern[name] = matches
    
    for pattern_name, matches in matches_by_pattern.items():
        if matches:
            print(f"\n   ✅ {pattern_name}: Found {len(matches)} matches")
            for i, (line_num, line, term) in enumerate(matches[:3]):
                print(f"      {i+1}. Line {line_num}: {term}")
                print(f"         Full: {line[:80]}...")
        else:
            print(f"\n   ❌ {pattern_name}: No matches")
    
    # Show sample of definitions section
    print(f"\n📄 Sample (first 50 lines of definitions):")
    print("-"*80)
    for i, line in enumerate(def_section[:50]):
        print(f"{i:3d}: {line}")
    print("-"*80)

def analyze_tables(text):
    """Check how tables are formatted"""
    print("\n" + "="*80)
    print("🔬 TABLE ANALYSIS")
    print("="*80)
    
    lines = text.split('\n')
    
    # Find potential tables
    table_indicators = []
    
    # Pattern 1: Markdown tables with |
    for i, line in enumerate(lines):
        if '|' in line and line.count('|') >= 3:
            table_indicators.append(('Markdown', i, line.strip()))
    
    if not table_indicators:
        print("\n❌ No markdown tables (with |) found!")
        print("\nLet's check for other table formats...")
        
        # Look for "Key Information" or similar
        for i, line in enumerate(lines):
            if 'key information' in line.lower() or 'table' in line.lower():
                print(f"\n✅ Found table-related text at line {i}:")
                print(f"   {line.strip()}")
                # Show surrounding context
                print(f"\n   Context (10 lines before and after):")
                start = max(0, i-10)
                end = min(len(lines), i+10)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"{marker} {j:4d}: {lines[j]}")
    else:
        print(f"\n✅ Found {len(table_indicators)} lines with | (potential tables)")
        
        # Group consecutive lines
        groups = []
        current_group = [table_indicators[0]]
        
        for indicator in table_indicators[1:]:
            if indicator[1] - current_group[-1][1] <= 2:  # Within 2 lines
                current_group.append(indicator)
            else:
                if len(current_group) >= 3:  # At least 3 lines = potential table
                    groups.append(current_group)
                current_group = [indicator]
        
        if len(current_group) >= 3:
            groups.append(current_group)
        
        print(f"\n📊 Found {len(groups)} table group(s):")
        
        for idx, group in enumerate(groups[:3], 1):  # Show first 3 tables
            print(f"\n   Table {idx}: Lines {group[0][1]}-{group[-1][1]} ({len(group)} rows)")
            print(f"   Sample rows:")
            for i, (_, line_num, line) in enumerate(group[:5]):
                print(f"      {line_num:4d}: {line}")
            
            # Check for separator line (|---|---|)
            has_separator = any(re.match(r'^\|[\s\-:|]+\|$', line) 
                               for _, _, line in group)
            print(f"   Has separator line: {'✅' if has_separator else '❌'}")

def analyze_clause_structure(text):
    """Check clause numbering structure"""
    print("\n" + "="*80)
    print("🔬 CLAUSE STRUCTURE ANALYSIS")
    print("="*80)
    
    lines = text.split('\n')
    
    # Find all numbered headings
    numbered_sections = []
    for i, line in enumerate(lines):
        match = re.match(r'^##\s+(\d+)\.?\s+(.+)', line.strip())
        if match:
            clause_num = match.group(1)
            title = match.group(2)
            numbered_sections.append((i, clause_num, title))
    
    print(f"\n✅ Found {len(numbered_sections)} numbered sections")
    
    if numbered_sections:
        print(f"\n📋 All sections:")
        for i, (line_num, num, title) in enumerate(numbered_sections):
            print(f"   {num:2s}. {title[:60]}")
    
    # Check for subsections
    print(f"\n🔍 Checking for subsections (1.1, 1.2, etc.)...")
    subsections = []
    for i, line in enumerate(lines[:500]):  # Check first 500 lines
        match = re.match(r'^\s*(\d+\.\d+)\s+(.+)', line.strip())
        if match:
            subsections.append((i, match.group(1), match.group(2)))
    
    if subsections:
        print(f"   ✅ Found {len(subsections)} subsections")
        print(f"   First 10:")
        for i, (line_num, num, title) in enumerate(subsections[:10]):
            print(f"      {num}: {title[:50]}")
    else:
        print(f"   ❌ No subsections found (no 1.1, 1.2 format)")

def main():
    print("🔬 DOCUMENT STRUCTURE DIAGNOSTIC")
    print("="*80)
    print("This will show what's actually in your cached markdown")
    print("="*80)
    
    # Find cache file
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    
    if not cache_files:
        print("\n❌ No cache files found!")
        print(f"   Expected location: {CACHE_DIR}")
        print("\n   Run: python pipeline/ingest_01.py")
        return
    
    cache_file = cache_files[0]
    cache_path = os.path.join(CACHE_DIR, cache_file)
    
    print(f"\n📄 Analyzing: {cache_file}")
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read cache: {e}")
        return
    
    if not cache_data or not isinstance(cache_data, list):
        print("❌ Invalid cache format")
        return
    
    full_doc = cache_data[0]
    full_text = full_doc.get('text', '')
    metadata = full_doc.get('metadata', {})
    
    print(f"\n📊 Document Stats:")
    print(f"   Filename: {metadata.get('filename', 'Unknown')}")
    print(f"   Total characters: {len(full_text):,}")
    print(f"   Total lines: {len(full_text.split(chr(10))):,}")
    
    # Run analyses
    analyze_clause_structure(full_text)
    analyze_definitions_section(full_text)
    analyze_tables(full_text)
    
    # Summary
    print("\n" + "="*80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*80)
    print("""
Based on this analysis, we can determine:
1. How definitions are formatted → Adjust regex in index_02_ENHANCED.py
2. How tables are formatted → Adjust table detection logic
3. Why enhancements didn't trigger → Fix the patterns

Next steps:
1. Review the output above
2. Share the "DEFINITIONS ANALYSIS" and "TABLE ANALYSIS" sections
3. I'll provide a custom-tuned indexer for your PDF format
""")

if __name__ == "__main__":
    main()
