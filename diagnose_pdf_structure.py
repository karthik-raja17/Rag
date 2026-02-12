#!/usr/bin/env python3
"""
EMERGENCY DIAGNOSTIC: Inspect your actual cache file structure
This will show us EXACTLY what patterns exist in your PDF
"""
import os
import sys
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import CACHE_DIR

def analyze_markdown_structure(text, filename):
    """
    Deep analysis of markdown structure to find splitting patterns
    """
    print(f"\n{'='*80}")
    print(f"🔬 DEEP ANALYSIS: {filename}")
    print(f"{'='*80}")
    
    lines = text.split('\n')
    total_lines = len(lines)
    
    print(f"\n📊 Basic Stats:")
    print(f"   Total lines: {total_lines:,}")
    print(f"   Total chars: {len(text):,}")
    print(f"   Non-empty lines: {sum(1 for l in lines if l.strip()):,}")
    
    # Find ALL heading patterns
    print(f"\n📋 HEADING ANALYSIS:")
    
    # H1 headings
    h1_lines = [(i, line) for i, line in enumerate(lines) if line.strip().startswith('# ') and not line.strip().startswith('##')]
    print(f"\n   # (H1) headings: {len(h1_lines)}")
    if h1_lines:
        print(f"   First 10 H1 headings:")
        for i, (line_num, line) in enumerate(h1_lines[:10]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # H2 headings
    h2_lines = [(i, line) for i, line in enumerate(lines) if line.strip().startswith('## ') and not line.strip().startswith('###')]
    print(f"\n   ## (H2) headings: {len(h2_lines)}")
    if h2_lines:
        print(f"   First 10 H2 headings:")
        for i, (line_num, line) in enumerate(h2_lines[:10]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # H3 headings
    h3_lines = [(i, line) for i, line in enumerate(lines) if line.strip().startswith('### ')]
    print(f"\n   ### (H3) headings: {len(h3_lines)}")
    if h3_lines:
        print(f"   First 10 H3 headings:")
        for i, (line_num, line) in enumerate(h3_lines[:10]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # NUMBERED PATTERNS
    print(f"\n🔢 NUMBERED SECTION PATTERNS:")
    
    # Pattern 1: ## 1. TITLE
    pattern1 = r'^##\s+\d+\.\s+[A-Z]'
    matches1 = [(i, line) for i, line in enumerate(lines) if re.match(pattern1, line.strip())]
    print(f"\n   Pattern: '## 1. TITLE' (with period): {len(matches1)}")
    if matches1:
        for i, (line_num, line) in enumerate(matches1[:5]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # Pattern 2: ## 1 TITLE (no period)
    pattern2 = r'^##\s+\d+\s+[A-Z]'
    matches2 = [(i, line) for i, line in enumerate(lines) if re.match(pattern2, line.strip()) and not re.match(pattern1, line.strip())]
    print(f"\n   Pattern: '## 1 TITLE' (no period): {len(matches2)}")
    if matches2:
        for i, (line_num, line) in enumerate(matches2[:5]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # Pattern 3: # 1. TITLE (H1 level)
    pattern3 = r'^#\s+\d+\.\s+[A-Z]'
    matches3 = [(i, line) for i, line in enumerate(lines) if re.match(pattern3, line.strip())]
    print(f"\n   Pattern: '# 1. TITLE' (H1 numbered): {len(matches3)}")
    if matches3:
        for i, (line_num, line) in enumerate(matches3[:5]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # Pattern 4: Roman numerals
    pattern4 = r'^##\s+[IVX]+\.\s+[A-Z]'
    matches4 = [(i, line) for i, line in enumerate(lines) if re.match(pattern4, line.strip())]
    print(f"\n   Pattern: '## I. TITLE' (Roman numerals): {len(matches4)}")
    if matches4:
        for i, (line_num, line) in enumerate(matches4[:5]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # Pattern 5: Uppercase sections without numbers
    pattern5 = r'^##\s+[A-Z][A-Z\s]+$'
    matches5 = [(i, line) for i, line in enumerate(lines) if re.match(pattern5, line.strip())]
    print(f"\n   Pattern: '## UPPERCASE TITLE' (no numbers): {len(matches5)}")
    if matches5:
        for i, (line_num, line) in enumerate(matches5[:10]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # Find section-like patterns that don't start with #
    print(f"\n📄 NON-MARKDOWN SECTION PATTERNS:")
    
    # All caps lines (potential section headers)
    all_caps_lines = [(i, line) for i, line in enumerate(lines) 
                      if line.strip() and line.strip().isupper() and len(line.strip()) > 5 
                      and not line.strip().startswith('#')]
    print(f"\n   ALL CAPS lines (potential sections): {len(all_caps_lines)}")
    if all_caps_lines:
        for i, (line_num, line) in enumerate(all_caps_lines[:10]):
            print(f"      Line {line_num}: {line.strip()[:100]}")
    
    # Lines with underscores (table separators or formatting)
    underscore_lines = [line for line in lines if '___' in line or '\\_\\_\\_' in line]
    print(f"\n   Lines with underscores: {len(underscore_lines)}")
    if underscore_lines:
        print(f"   First 5 examples:")
        for line in underscore_lines[:5]:
            print(f"      {line.strip()[:100]}")
    
    # SAMPLE CONTENT
    print(f"\n📝 DOCUMENT SAMPLE (first 50 lines):")
    print("   " + "─" * 76)
    for i, line in enumerate(lines[:50]):
        print(f"   {i:3d}: {line[:75]}")
    print("   " + "─" * 76)
    
    # RECOMMENDATION
    print(f"\n💡 RECOMMENDATIONS:")
    
    best_pattern = None
    best_count = 0
    
    if len(matches1) > best_count:
        best_pattern = "'## N. TITLE' (with period)"
        best_count = len(matches1)
    
    if len(matches2) > best_count:
        best_pattern = "'## N TITLE' (no period)"
        best_count = len(matches2)
    
    if len(matches3) > best_count:
        best_pattern = "'# N. TITLE' (H1 level)"
        best_count = len(matches3)
    
    if len(matches5) > best_count and len(matches5) < 100:  # Not too many
        best_pattern = "'## UPPERCASE' (no numbers)"
        best_count = len(matches5)
    
    if len(all_caps_lines) > best_count and len(all_caps_lines) < 100:
        best_pattern = "'ALL CAPS' (no markdown)"
        best_count = len(all_caps_lines)
    
    if best_pattern:
        print(f"   ✅ Best pattern: {best_pattern} ({best_count} matches)")
        print(f"   → Use this pattern for clause splitting")
    else:
        print(f"   ⚠️  No clear section pattern found")
        print(f"   → Document may not have structured sections")
        print(f"   → Consider semantic chunking instead")
    
    print(f"\n{'='*80}\n")

def main():
    print("🔬 EMERGENCY DIAGNOSTIC: PDF Structure Analysis")
    print("="*80)
    
    if not os.path.exists(CACHE_DIR):
        print(f"❌ Cache directory not found: {CACHE_DIR}")
        print("Run ingest_01.py first!")
        return
    
    # Get cache files
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    
    if not cache_files:
        print(f"⚠️  No cache files in {CACHE_DIR}")
        print("\nTo generate cache files:")
        print("  1. Place PDF in data/ directory")
        print("  2. Delete ingestion_tracker.db")
        print("  3. Run: python pipeline/ingest_01.py")
        return
    
    print(f"✅ Found {len(cache_files)} cache file(s)\n")
    
    # Analyze each file
    for cf in cache_files:
        cache_path = os.path.join(CACHE_DIR, cf)
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            full_text = cache_data[0]['text']
            filename = cache_data[0].get('metadata', {}).get('filename', 'unknown.pdf')
            
            analyze_markdown_structure(full_text, filename)
            
        except Exception as e:
            print(f"❌ Failed to analyze {cf}: {e}")

if __name__ == "__main__":
    main()
