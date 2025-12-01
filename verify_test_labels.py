"""
Simple verification that test.tsv has real labels and they're being read correctly
"""
import csv

# Read the raw test.tsv file
test_file = "data/data_raw/justice_legal_dataset/test.tsv"

print("Reading test.tsv file directly...")
print("=" * 80)

label_counts = {"True": 0, "False": 0}
total_lines = 0

with open(test_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t", quotechar='"')
    for i, line in enumerate(reader):
        if i == 0:
            print(f"Header: {line}")
            continue

        total_lines += 1

        # Show first 5 examples
        if i <= 5:
            print(f"\nRow {i}:")
            print(f"  Index: {line[0]}")
            print(f"  Text (first 80 chars): {line[1][:80]}...")
            print(f"  Label: {line[2]}")

        # Count labels
        if line[2] in label_counts:
            label_counts[line[2]] += 1

print("\n" + "=" * 80)
print("Label Distribution in Raw test.tsv:")
print(f"  True:  {label_counts['True']} ({100*label_counts['True']/total_lines:.1f}%)")
print(f"  False: {label_counts['False']} ({100*label_counts['False']/total_lines:.1f}%)")
print(f"  Total: {total_lines} examples")

if label_counts['True'] > 0 and label_counts['False'] > 0:
    print("\n✓ Test set has both True and False labels")
    print("✓ Labels are present in the file")
else:
    print("\n❌ Only one class present!")
