"""
Pre-flight diagnostics for PKD Stage 1: Teacher Fine-tuning (No PyTorch import)
Tests all prerequisites before starting legal-bert-base teacher training
"""
import os
import sys
import json

print("="*80)
print("PKD STAGE 1 READINESS CHECK: Teacher Fine-tuning")
print("="*80)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def test_pass(name):
    results["passed"].append(name)
    print(f"  [OK] {name}")

def test_fail(name, details):
    results["failed"].append((name, details))
    print(f"  [X] {name}")
    print(f"      {details}")

def test_warn(name, details):
    results["warnings"].append((name, details))
    print(f"  [!] {name}")
    print(f"      {details}")

# =============================================================================
# TEST 1: Model Files
# =============================================================================
print("\n" + "-"*80)
print("TEST 1: Legal-BERT-Base Model Files")
print("-"*80)

model_dir = os.path.join(os.path.dirname(__file__), 'models', 'legal-bert-base')
required_files = ['bert_config.json', 'pytorch_model.bin', 'vocab.txt']

if not os.path.exists(model_dir):
    test_fail("Model directory exists", f"Not found: {model_dir}")
else:
    test_pass("Model directory exists")

    for fname in required_files:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            test_fail(f"Model file: {fname}", f"Not found: {fpath}")
        else:
            fsize = os.path.getsize(fpath) / (1024**2)  # MB
            test_pass(f"Model file: {fname} ({fsize:.1f} MB)")

# Verify config contents
config_path = os.path.join(model_dir, 'bert_config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    expected_config = {
        'num_hidden_layers': 12,
        'hidden_size': 768,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'vocab_size': 30522
    }

    for key, expected_val in expected_config.items():
        actual_val = config.get(key)
        if actual_val == expected_val:
            test_pass(f"Config {key}: {actual_val}")
        else:
            test_fail(f"Config {key}", f"Expected {expected_val}, got {actual_val}")

# =============================================================================
# TEST 2: Student Model Files (for future stages)
# =============================================================================
print("\n" + "-"*80)
print("TEST 2: Legal-BERT-Small Model Files (for Stage 3)")
print("-"*80)

student_dir = os.path.join(os.path.dirname(__file__), 'models', 'legal-bert-small')
if not os.path.exists(student_dir):
    test_warn("Student model directory", f"Not found: {student_dir} (needed for Stage 3)")
else:
    test_pass("Student model directory exists")
    for fname in required_files:
        fpath = os.path.join(student_dir, fname)
        if os.path.exists(fpath):
            test_pass(f"Student file: {fname}")

# =============================================================================
# TEST 3: Dataset Files
# =============================================================================
print("\n" + "-"*80)
print("TEST 3: Legal Dataset Files")
print("-"*80)

data_dir = os.path.join(os.path.dirname(__file__), 'data', 'data_raw', 'LEGAL')
data_files = ['train.tsv', 'dev.tsv', 'test.tsv']

if not os.path.exists(data_dir):
    test_fail("Data directory exists", f"Not found: {data_dir}")
else:
    test_pass("Data directory exists")

    for fname in data_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            test_fail(f"Data file: {fname}", f"Not found: {fpath}")
        else:
            # Count lines
            with open(fpath, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            test_pass(f"Data file: {fname} ({num_lines} lines)")

# =============================================================================
# TEST 4: Code Implementation
# =============================================================================
print("\n" + "-"*80)
print("TEST 4: Code Implementation")
print("-"*80)

# Check LegalProcessor
try:
    from nli_data_processing import processors, output_modes

    if 'legal' in processors:
        test_pass("LegalProcessor registered in processors dict")
    else:
        test_fail("LegalProcessor registration", "Not found in processors dict")

    if 'legal' in output_modes:
        test_pass("Legal task registered in output_modes")
    else:
        test_fail("Legal task registration", "Not found in output_modes dict")

    # Test instantiation
    processor = processors['legal']()
    labels = processor.get_labels()
    if labels == ["0", "1"]:
        test_pass(f"LegalProcessor.get_labels(): {labels}")
    else:
        test_fail("LegalProcessor.get_labels()", f"Expected ['0', '1'], got {labels}")

    # Test data loading
    train_examples = processor.get_train_examples(data_dir)
    dev_examples = processor.get_dev_examples(data_dir)
    test_examples = processor.get_test_examples(data_dir)

    test_pass(f"Train examples loaded: {len(train_examples)}")
    test_pass(f"Dev examples loaded: {len(dev_examples)}")
    test_pass(f"Test examples loaded: {len(test_examples)}")

    # Verify example structure
    if len(train_examples) > 0:
        ex = train_examples[0]
        if hasattr(ex, 'guid') and hasattr(ex, 'text_a') and hasattr(ex, 'label'):
            test_pass("Example structure correct (guid, text_a, label)")
        else:
            test_fail("Example structure", "Missing expected attributes")

        if ex.label in ["0", "1"]:
            test_pass(f"Label conversion working: {ex.label}")
        else:
            test_fail("Label conversion", f"Expected '0' or '1', got '{ex.label}'")

except ImportError as e:
    test_fail("Import nli_data_processing", str(e))
except Exception as e:
    test_fail("LegalProcessor instantiation", str(e))

# Check argument parser
try:
    from argument_parser import get_legal_teacher_argv

    test_pass("get_legal_teacher_argv() function exists")

    # Test function call (will fail on torch import, but we can catch it)
    try:
        argv = get_legal_teacher_argv('legal-bert-base')
        test_pass(f"get_legal_teacher_argv() returns {len(argv)} arguments")

        # Verify key arguments
        arg_checks = {
            '--task_name': 'legal',
            '--student_hidden_layers': '12',
            '--alpha': '0.0',
            '--kd_model': 'kd',
            '--do_train': 'True',
            '--do_eval': 'True'
        }

        for key, expected_val in arg_checks.items():
            try:
                idx = argv.index(key)
                actual_val = argv[idx + 1]
                if actual_val == expected_val:
                    test_pass(f"Argument {key}: {actual_val}")
                else:
                    test_warn(f"Argument {key}", f"Expected {expected_val}, got {actual_val}")
            except (ValueError, IndexError):
                test_fail(f"Argument {key}", "Not found in argv")
    except ImportError as ie:
        test_warn("get_legal_teacher_argv() call", f"Requires torch: {ie}")

except ImportError as e:
    test_fail("Import argument_parser", str(e))
except Exception as e:
    test_fail("get_legal_teacher_argv() import", str(e))

# =============================================================================
# TEST 5: Output Directories
# =============================================================================
print("\n" + "-"*80)
print("TEST 5: Output Directories")
print("-"*80)

output_dirs = [
    'data/outputs',
    'data/outputs/legal'
]

for dir_path in output_dirs:
    full_path = os.path.join(os.path.dirname(__file__), dir_path)
    if os.path.exists(full_path):
        test_pass(f"Output directory exists: {dir_path}")
    else:
        try:
            os.makedirs(full_path, exist_ok=True)
            test_pass(f"Created output directory: {dir_path}")
        except Exception as e:
            test_fail(f"Create directory: {dir_path}", str(e))

# =============================================================================
# TEST 6: Dependencies (without torch)
# =============================================================================
print("\n" + "-"*80)
print("TEST 6: Python Dependencies (Non-Torch)")
print("-"*80)

required_packages = [
    'numpy',
    'tqdm',
    'pandas',
]

for package in required_packages:
    try:
        __import__(package)
        test_pass(f"Package installed: {package}")
    except ImportError:
        test_fail(f"Package missing: {package}", f"Run: pip install {package}")

# Check for BERT modules (without importing)
bert_module_path = os.path.join(os.path.dirname(__file__), 'BERT', 'pytorch_pretrained_bert')
if os.path.exists(bert_module_path):
    test_pass("BERT modules directory exists")
else:
    test_fail("BERT modules", f"Directory not found: {bert_module_path}")

# =============================================================================
# TEST 7: Training Script Readiness
# =============================================================================
print("\n" + "-"*80)
print("TEST 7: Training Script Readiness")
print("-"*80)

training_script = os.path.join(os.path.dirname(__file__), 'NLI_KD_training.py')
if os.path.exists(training_script):
    test_pass("NLI_KD_training.py exists")

    # Check for legal teacher argv line
    with open(training_script, 'r') as f:
        content = f.read()
        if 'get_legal_teacher_argv' in content:
            test_pass("Training script references get_legal_teacher_argv()")
        else:
            test_fail("Training script check", "get_legal_teacher_argv not found in NLI_KD_training.py")
else:
    test_fail("Training script", f"Not found: {training_script}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n[OK] Tests Passed: {len(results['passed'])}")
for test in results['passed'][:5]:  # Show first 5
    print(f"  - {test}")
if len(results['passed']) > 5:
    print(f"  ... and {len(results['passed']) - 5} more")

if results['warnings']:
    print(f"\n[!] Warnings: {len(results['warnings'])}")
    for test, details in results['warnings']:
        print(f"  - {test}")
        print(f"    Details: {details}")

if results['failed']:
    print(f"\n[X] Tests Failed: {len(results['failed'])}")
    for test, details in results['failed']:
        print(f"  - {test}")
        print(f"    Details: {details}")
    print("\n" + "="*80)
    print("RESULT: NOT READY - Fix failed tests before proceeding")
    print("="*80)
    sys.exit(1)
else:
    print("\n" + "="*80)
    if results['warnings']:
        print("RESULT: MOSTLY READY (with warnings)")
    else:
        print("RESULT: READY FOR STAGE 1 - Teacher Fine-tuning")
    print("="*80)
    print("\nNext steps:")
    print("1. In NLI_KD_training.py, uncomment line 49:")
    print("   argv = get_legal_teacher_argv('legal-bert-base')")
    print("2. Run: python NLI_KD_training.py")
    print("3. Monitor training progress and dev accuracy")
    print("\nNote: PyTorch environment test skipped (requires working torch install)")
    print("="*80)
    sys.exit(0)
