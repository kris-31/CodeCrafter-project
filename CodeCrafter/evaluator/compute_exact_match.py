import argparse
import evaluate

# Load the ExactMatch metric
exact_match = evaluate.load("exact_match")

# Argument parser to handle command-line inputs
parser = argparse.ArgumentParser(description="Compute exact match between code snippets.")

# Define command-line arguments for candicode and refcode file paths
parser.add_argument(
    "--candicode",
    required=True,
    help="Path to the candicodes text file.",
)

parser.add_argument(
    "--refcode",
    required=True,
    help="Path to the refcodes text file.",
)

# Parse the command-line arguments
args = parser.parse_args()

# Function to read text files
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()  # Read the file line-by-line
            return [line.strip() for line in data]  # Strip extra spaces and newlines
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# Read the candicode and refcode text files from the provided arguments
candicodes_list = read_text_file(args.candicode)
refcodes_list = read_text_file(args.refcode)

# Check if data was read successfully
if not candicodes_list or not refcodes_list:
    print("Failed to read one or both text files. Please check the data.")
    exit(1)

# Compute exact match between candicodes and refcodes
results = exact_match.compute(
    predictions=candicodes_list,
    references=refcodes_list,
    ignore_case=True,
    ignore_punctuation=True,
)

# Output the exact match score
print("Exact Match Score:", round(results["exact_match"], 2))
