import argparse
import os
import datasets
from transformers import T5ForConditionalGeneration, AutoTokenizer
###content/CodeCrafter-project/CodeCrafter/saved_models/checkpoint-120
def generate_output(input_text, output_dir, model, tokenizer, device):
    # Generate output
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using CodeCrafter model.")
    parser.add_argument("--dataset_name", default="mbpp", help="Name of the Hugging Face dataset")
    parser.add_argument("--output_dir", default="generated_output", help="Output directory to save generated text")
    parser.add_argument("--model_checkpoint", default="/home/kris/Project/Proposed-working/CodeCrafter/saved_models/checkpoint-120", help="Path to the model checkpoint")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = datasets.load_dataset(args.dataset_name)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint).to(args.device)

    # Open file to save reference codes
    refcode_file = open("refcodecolab.txt", "w")

    # Open file to save all generated solutions
    allsolutions_file = open("allsolutions.txt", "w")

    # Generate outputs for all questions in the dataset
    for i, example in enumerate(dataset):
        input_text = example["text"]
        code = example["code"]

        # Save reference code to file
        refcode_file.write(f"{code}\n")

        # Generate output for input text
        generated_text = generate_output(input_text, args.output_dir, model, tokenizer, args.device)

        # Save generated solution to file
        allsolutions_file.write(f"{generated_text}\n")

        print(f"Generated output for question {i+1}")

    # Close files
    refcode_file.close()
    allsolutions_file.close()

if __name__ == "__main__":
    main()
