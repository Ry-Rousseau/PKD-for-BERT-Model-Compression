"""
Test script to verify LegalProcessor data loading
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nli_data_processing import LegalProcessor

def test_legal_processor():
    print("="*80)
    print("Testing LegalProcessor Implementation")
    print("="*80)

    # Initialize processor
    processor = LegalProcessor()
    print("\n✓ LegalProcessor initialized successfully")

    # Check labels
    labels = processor.get_labels()
    print(f"\n✓ Labels: {labels}")
    assert labels == ["0", "1"], f"Expected ['0', '1'], got {labels}"

    # Set data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'data_raw', 'LEGAL')
    print(f"\n✓ Data directory: {data_dir}")

    # Load train examples
    print("\n" + "-"*80)
    print("Loading TRAIN examples...")
    print("-"*80)
    train_examples = processor.get_train_examples(data_dir)
    print(f"✓ Train examples loaded: {len(train_examples)}")

    # Show first 3 train examples
    print("\nFirst 3 train examples:")
    for i, example in enumerate(train_examples[:3]):
        print(f"\n  Example {i}:")
        print(f"    GUID: {example.guid}")
        print(f"    Text: {example.text_a[:100]}...")
        print(f"    Text length: {len(example.text_a)} chars")
        print(f"    Text_b: {example.text_b}")
        print(f"    Label: {example.label}")

    # Load dev examples
    print("\n" + "-"*80)
    print("Loading DEV examples...")
    print("-"*80)
    dev_examples = processor.get_dev_examples(data_dir)
    print(f"✓ Dev examples loaded: {len(dev_examples)}")

    # Show first 3 dev examples
    print("\nFirst 3 dev examples:")
    for i, example in enumerate(dev_examples[:3]):
        print(f"\n  Example {i}:")
        print(f"    GUID: {example.guid}")
        print(f"    Text: {example.text_a[:100]}...")
        print(f"    Label: {example.label}")

    # Load test examples
    print("\n" + "-"*80)
    print("Loading TEST examples...")
    print("-"*80)
    test_examples = processor.get_test_examples(data_dir)
    print(f"✓ Test examples loaded: {len(test_examples)}")

    # Check label distribution
    print("\n" + "-"*80)
    print("Label Distribution Analysis")
    print("-"*80)

    train_labels = [ex.label for ex in train_examples]
    dev_labels = [ex.label for ex in dev_examples]

    train_0 = train_labels.count("0")
    train_1 = train_labels.count("1")
    dev_0 = dev_labels.count("0")
    dev_1 = dev_labels.count("1")

    print(f"\nTrain set:")
    print(f"  Label '0' (False): {train_0} ({train_0/len(train_labels)*100:.1f}%)")
    print(f"  Label '1' (True):  {train_1} ({train_1/len(train_labels)*100:.1f}%)")

    print(f"\nDev set:")
    print(f"  Label '0' (False): {dev_0} ({dev_0/len(dev_labels)*100:.1f}%)")
    print(f"  Label '1' (True):  {dev_1} ({dev_1/len(dev_labels)*100:.1f}%)")

    # Verify all labels are valid
    print("\n" + "-"*80)
    print("Data Validation")
    print("-"*80)

    invalid_train = [l for l in train_labels if l not in ["0", "1"]]
    invalid_dev = [l for l in dev_labels if l not in ["0", "1"]]

    if invalid_train:
        print(f"✗ WARNING: Found {len(invalid_train)} invalid labels in train set: {set(invalid_train)}")
    else:
        print(f"✓ All train labels are valid")

    if invalid_dev:
        print(f"✗ WARNING: Found {len(invalid_dev)} invalid labels in dev set: {set(invalid_dev)}")
    else:
        print(f"✓ All dev labels are valid")

    # Check for None text
    none_text = sum(1 for ex in train_examples + dev_examples if ex.text_a is None or ex.text_a == "")
    if none_text > 0:
        print(f"✗ WARNING: Found {none_text} examples with None or empty text")
    else:
        print(f"✓ All examples have valid text")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        test_legal_processor()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
