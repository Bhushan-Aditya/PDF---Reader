#!/usr/bin/env python3
"""
Accuracy Assessment Tool
Evaluates the performance of our PDF processor against expected outputs
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

class AccuracyAssessor:
    def __init__(self):
        self.sample_outputs_dir = Path("sample_dataset/outputs")
        self.current_outputs_dir = Path("output")
        
    def load_json_file(self, file_path: Path) -> Dict:
        """Load JSON file safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def calculate_title_similarity(self, expected: str, actual: str) -> float:
        """Calculate title similarity score."""
        # Handle empty strings - if both are empty, they match
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0
        
        # Normalize strings
        expected_norm = expected.strip().lower()
        actual_norm = actual.strip().lower()
        
        if expected_norm == actual_norm:
            return 1.0
        
        # Check if actual contains expected (partial match)
        if expected_norm in actual_norm:
            return 0.8
        
        # Check if expected contains actual (partial match)
        if actual_norm in expected_norm:
            return 0.8
        
        # Calculate word overlap
        expected_words = set(expected_norm.split())
        actual_words = set(actual_norm.split())
        
        if not expected_words or not actual_words:
            return 0.0
        
        intersection = expected_words.intersection(actual_words)
        union = expected_words.union(actual_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_outline_similarity(self, expected: List[Dict], actual: List[Dict]) -> float:
        """Calculate outline similarity score."""
        if not expected and not actual:
            return 1.0
        
        if not expected or not actual:
            return 0.0
        
        # Normalize outlines
        expected_norm = []
        for item in expected:
            text = item.get('text', '').strip().lower()
            level = item.get('level', '')
            expected_norm.append((level, text))
        
        actual_norm = []
        for item in actual:
            text = item.get('text', '').strip().lower()
            level = item.get('level', '')
            actual_norm.append((level, text))
        
        # Calculate exact matches
        exact_matches = 0
        for exp_item in expected_norm:
            if exp_item in actual_norm:
                exact_matches += 1
        
        # Calculate partial matches (same level, similar text)
        partial_matches = 0
        for exp_item in expected_norm:
            exp_level, exp_text = exp_item
            for act_item in actual_norm:
                act_level, act_text = act_item
                if exp_level == act_level:
                    # Check if texts are similar
                    if exp_text in act_text or act_text in exp_text:
                        partial_matches += 1
                        break
        
        total_expected = len(expected_norm)
        if total_expected == 0:
            return 1.0 if len(actual_norm) == 0 else 0.0
        
        exact_score = exact_matches / total_expected
        partial_score = partial_matches / total_expected
        
        return max(exact_score, partial_score * 0.8)
    
    def assess_file(self, filename: str) -> Dict:
        """Assess accuracy for a single file."""
        expected_file = self.sample_outputs_dir / f"{filename}.json"
        actual_file = self.current_outputs_dir / f"{filename}.json"
        
        if not expected_file.exists():
            print(f"Warning: Expected file {expected_file} not found")
            return {"error": "Expected file not found"}
        
        if not actual_file.exists():
            print(f"Warning: Actual file {actual_file} not found")
            return {"error": "Actual file not found"}
        
        expected_data = self.load_json_file(expected_file)
        actual_data = self.load_json_file(actual_file)
        
        # Calculate title accuracy
        expected_title = expected_data.get('title', '')
        actual_title = actual_data.get('title', '')
        title_score = self.calculate_title_similarity(expected_title, actual_title)
        
        # Calculate outline accuracy
        expected_outline = expected_data.get('outline', [])
        actual_outline = actual_data.get('outline', [])
        outline_score = self.calculate_outline_similarity(expected_outline, actual_outline)
        
        # Calculate overall accuracy
        overall_score = (title_score + outline_score) / 2
        
        return {
            'filename': filename,
            'title_score': title_score,
            'outline_score': outline_score,
            'overall_score': overall_score,
            'expected_title': expected_title,
            'actual_title': actual_title,
            'expected_outline_count': len(expected_outline),
            'actual_outline_count': len(actual_outline)
        }
    
    def assess_all_files(self) -> Dict:
        """Assess accuracy for all files."""
        results = {}
        total_title_score = 0
        total_outline_score = 0
        total_overall_score = 0
        file_count = 0
        
        # Get all expected files
        expected_files = list(self.sample_outputs_dir.glob("*.json"))
        
        for expected_file in expected_files:
            filename = expected_file.stem
            result = self.assess_file(filename)
            
            if 'error' not in result:
                results[filename] = result
                total_title_score += result['title_score']
                total_outline_score += result['outline_score']
                total_overall_score += result['overall_score']
                file_count += 1
        
        if file_count > 0:
            avg_title_score = total_title_score / file_count
            avg_outline_score = total_outline_score / file_count
            avg_overall_score = total_overall_score / file_count
        else:
            avg_title_score = avg_outline_score = avg_overall_score = 0
        
        return {
            'individual_results': results,
            'summary': {
                'total_files': file_count,
                'average_title_score': avg_title_score,
                'average_outline_score': avg_outline_score,
                'average_overall_score': avg_overall_score
            }
        }
    
    def print_detailed_results(self, results: Dict):
        """Print detailed assessment results."""
        print("=" * 60)
        print("ACCURACY ASSESSMENT RESULTS")
        print("=" * 60)
        
        summary = results['summary']
        print(f"\n📊 SUMMARY:")
        print(f"   Total Files: {summary['total_files']}")
        print(f"   Average Title Score: {summary['average_title_score']:.3f} ({summary['average_title_score']*100:.1f}%)")
        print(f"   Average Outline Score: {summary['average_outline_score']:.3f} ({summary['average_outline_score']*100:.1f}%)")
        print(f"   Average Overall Score: {summary['average_overall_score']:.3f} ({summary['average_overall_score']*100:.1f}%)")
        
        print(f"\n📋 DETAILED RESULTS:")
        for filename, result in results['individual_results'].items():
            print(f"\n   {filename}:")
            print(f"     Title: {result['title_score']:.3f} ({result['title_score']*100:.1f}%)")
            print(f"     Outline: {result['outline_score']:.3f} ({result['outline_score']*100:.1f}%)")
            print(f"     Overall: {result['overall_score']:.3f} ({result['overall_score']*100:.1f}%)")
            
            if result['title_score'] < 0.9:
                print(f"     ⚠️  Title Issue: Expected '{result['expected_title'][:50]}...' vs Actual '{result['actual_title'][:50]}...'")
            
            if result['outline_score'] < 0.8:
                print(f"     ⚠️  Outline Issue: Expected {result['expected_outline_count']} items, got {result['actual_outline_count']} items")

def main():
    assessor = AccuracyAssessor()
    results = assessor.assess_all_files()
    assessor.print_detailed_results(results)

if __name__ == "__main__":
    main() 