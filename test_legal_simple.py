"""
Simple test script to verify Legal dataset loading without importing torch
"""
import csv
import os

def read_tsv(file_path):
    """Read TSV with quote handling"""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def test_legal_data():
    print("="*80)
    print("Testing Legal Dataset Files")
    print("="*80)

    data_dir = r"c:\Users\rhrou\Documents\DS_Projects_Local\PKD-for-BERT-Model-Compression\data\data_raw\LEGAL"

    # Test each file
    for filename in ["train.tsv", "dev.tsv", "test.tsv"]:
        filepath = os.path.join(data_dir, filename)
        print(f"\n{'-'*80}")
        print(f"Testing: {filename}")
        print(f"{'-'*80}")

        lines = read_tsv(filepath)
        print(f"Total lines (including header): {len(lines)}")

        # Check header
        header = lines[0]
        print(f"Header: {header}")

        # Process data lines
        examples = []
        label_counts = {"True": 0, "False": 0, "0": 0, "1": 0, "other": 0}

        for i, line in enumerate(lines[1:]):  # Skip header
            if len(line) < 3:
                print(f"  WARNING: Line {i+1} has only {len(line)} columns: {line}")
                continue

            index = line[0]
            text = line[1]
            label = line[2]

            # Count labels
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts["other"] += 1

            # Convert True/False to 1/0
            if label == "True":
                label_converted = "1"
            elif label == "False":
                label_converted = "0"
            else:
                label_converted = label

            examples.append({
                "index": index,
                "text": text,
                "label_raw": label,
                "label_converted": label_converted
            })

        print(f"\nExamples loaded: {len(examples)}")
        print(f"\nLabel distribution (raw):")
        for label, count in label_counts.items():
            if count > 0:
                print(f"  {label}: {count} ({count/len(examples)*100:.1f}%)")

        # Show first 3 examples
        print(f"\nFirst 3 examples:")
        for i, ex in enumerate(examples[:3]):
            print(f"\n  Example {i}:")
            print(f"    Index: {ex['index']}")
            print(f"    Text length: {len(ex['text'])} chars")
            print(f"    Text preview: {ex['text'][:80]}...")
            print(f"    Label (raw): {ex['label_raw']}")
            print(f"    Label (converted): {ex['label_converted']}")

        # Validation
        print(f"\nValidation:")

        # Check for valid labels after conversion
        invalid_labels = [ex for ex in examples if ex['label_converted'] not in ["0", "1"]]
        if invalid_labels:
            print(f"  [X] Found {len(invalid_labels)} invalid labels after conversion")
            for ex in invalid_labels[:5]:
                print(f"    - Index {ex['index']}: {ex['label_raw']} -> {ex['label_converted']}")
        else:
            print(f"  [OK] All labels valid after conversion")

        # Check for empty text
        empty_text = [ex for ex in examples if not ex['text'] or len(ex['text'].strip()) == 0]
        if empty_text:
            print(f"  [X] Found {len(empty_text)} examples with empty text")
        else:
            print(f"  [OK] All examples have non-empty text")

    print("\n" + "="*80)
    print("[OK] DATA VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    try:
        test_legal_data()
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
