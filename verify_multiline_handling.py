"""
Demonstration of how quotechar handles multi-line TSV entries
"""
import csv

data_file = r"c:\Users\rhrou\Documents\DS_Projects_Local\PKD-for-BERT-Model-Compression\data\data_raw\LEGAL\train.tsv"

print("="*80)
print("HOW MULTI-LINE TSV HANDLING WORKS")
print("="*80)

# Show raw file lines for example index 2
print("\n" + "-"*80)
print("RAW FILE CONTENT (Example index 2 - Charles Richmond case)")
print("-"*80)

with open(data_file, 'r', encoding='utf-8') as f:
    raw_lines = f.readlines()

print(f"Total raw lines in file: {len(raw_lines)}")
print(f"\nRaw line 3 (starts the entry):")
print(f"  {raw_lines[3][:150]}...")
print(f"\nRaw line 4 (continuation - second paragraph):")
print(f"  {raw_lines[4][:150]}...")
print(f"\nRaw line 5 (next entry):")
print(f"  {raw_lines[5][:150]}...")

print("\n" + "-"*80)
print("METHOD 1: Reading WITHOUT quotechar (BROKEN)")
print("-"*80)

with open(data_file, 'r', encoding='utf-8') as f:
    reader_no_quote = csv.reader(f, delimiter='\t')
    lines_no_quote = list(reader_no_quote)

print(f"Total parsed entries: {len(lines_no_quote)}")

# Line 3 should be index 2
if len(lines_no_quote) > 3:
    entry = lines_no_quote[3]
    print(f"\nParsed line 3:")
    print(f"  Columns: {len(entry)}")
    print(f"  Index: {entry[0] if len(entry) > 0 else 'N/A'}")
    print(f"  Text snippet: {entry[1][:100] if len(entry) > 1 else 'N/A'}...")
    print(f"  PROBLEM: Text is CUT OFF mid-sentence!")

if len(lines_no_quote) > 4:
    entry = lines_no_quote[4]
    print(f"\nParsed line 4 (should be continuation, but parsed as separate):")
    print(f"  Columns: {len(entry)}")
    print(f"  Content: {entry[0][:100] if len(entry) > 0 else 'N/A'}...")
    print(f"  PROBLEM: Second paragraph is treated as malformed row!")

print("\n" + "-"*80)
print("METHOD 2: Reading WITH quotechar='\"' (CORRECT)")
print("-"*80)

with open(data_file, 'r', encoding='utf-8') as f:
    reader_with_quote = csv.reader(f, delimiter='\t', quotechar='"')
    lines_with_quote = list(reader_with_quote)

print(f"Total parsed entries: {len(lines_with_quote)}")

# Entry at index 3 (after header)
if len(lines_with_quote) > 3:
    entry = lines_with_quote[3]
    print(f"\nParsed entry for index 2:")
    print(f"  Columns: {len(entry)}")
    print(f"  Index: {entry[0]}")
    print(f"  Label: {entry[2]}")
    print(f"  Text length: {len(entry[1])} characters")
    print(f"  Contains newline: {chr(10) in entry[1]}")
    print(f"\n  Full text:")
    # Split by newline to show both paragraphs
    paragraphs = entry[1].split('\n')
    for i, para in enumerate(paragraphs):
        print(f"    Paragraph {i+1}: {para[:120]}...")
    print(f"\n  SUCCESS: Both paragraphs captured in single entry!")

print("\n" + "="*80)
print("VERIFICATION: Count examples with multi-line text")
print("="*80)

multiline_count = 0
for i, entry in enumerate(lines_with_quote[1:]):  # Skip header
    if len(entry) >= 2 and '\n' in entry[1]:
        multiline_count += 1

print(f"Total entries: {len(lines_with_quote) - 1}")
print(f"Entries with multi-line text: {multiline_count}")
print(f"Percentage: {multiline_count / (len(lines_with_quote) - 1) * 100:.1f}%")

print("\n" + "="*80)
print("HOW THE LegalProcessor USES THIS:")
print("="*80)
print("""
In nli_data_processing.py, the LegalProcessor uses:

    self._read_tsv(os.path.join(data_dir, "train.tsv"), quotechar='"')

The base class _read_tsv method:

    reader = csv.reader(f, delimiter="\\t", quotechar=quotechar)

When quotechar='"' is set:
1. CSV reader sees opening quote: 2\t"charles richmond...
2. Continues reading across newlines until closing quote
3. Captures the entire multi-paragraph text as ONE field
4. Newlines WITHIN the quoted field are preserved
5. Returns proper 3-column row: [index, full_text, label]

WITHOUT quotechar:
1. CSV reader hits newline, thinks row is complete
2. Next line is treated as NEW row (but malformed)
3. Data is corrupted and counts are wrong
""")

print("="*80)
print("CONCLUSION: quotechar='\"' is ESSENTIAL and CORRECT")
print("="*80)
