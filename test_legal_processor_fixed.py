"""
Test script to verify LegalProcessor correctly reads test set labels
"""
import sys
sys.path.insert(0, '.')

from src.nli_data_processing import LegalProcessor
import os

# Initialize processor
processor = LegalProcessor()

# Get test examples
data_dir = "data/data_raw/justice_legal_dataset"
test_examples = processor.get_test_examples(data_dir)

print(f"✓ Loaded {len(test_examples)} test examples")

# Check first 10 examples
print("\nFirst 10 test examples:")
print("=" * 80)
for i, ex in enumerate(test_examples[:10]):
    print(f"{i+1}. GUID: {ex.guid}")
    print(f"   Label: {ex.label}")
    print(f"   Text (first 100 chars): {ex.text_a[:100]}...")
    print()

# Count label distribution
label_counts = {"0": 0, "1": 0}
for ex in test_examples:
    if ex.label in label_counts:
        label_counts[ex.label] += 1
    else:
        print(f"WARNING: Unexpected label '{ex.label}' in test set!")

print("\nLabel Distribution in Test Set:")
print("=" * 80)
print(f"Class 0 (False): {label_counts['0']} examples ({100*label_counts['0']/len(test_examples):.1f}%)")
print(f"Class 1 (True):  {label_counts['1']} examples ({100*label_counts['1']/len(test_examples):.1f}%)")
print(f"Total:           {len(test_examples)} examples")

# Check if any examples have the old dummy label issue
if label_counts['0'] == len(test_examples):
    print("\n❌ ERROR: All test examples have label '0' - dummy label issue still present!")
elif label_counts['0'] == 0 or label_counts['1'] == 0:
    print(f"\n⚠ WARNING: Only one class present - check dataset!")
else:
    print("\n✓ SUCCESS: Test set has mixed labels - fix is working!")
