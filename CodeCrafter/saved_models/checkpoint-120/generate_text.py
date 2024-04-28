import argparse
import os
from transformers import T5ForConditionalGeneration, AutoTokenizer

def generate_output(input_prompt, output_dir, model_checkpoint, device):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)

    # Generate output
    inputs = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

    # Save output to a text file
    output_file = os.path.join(output_dir, "generated_output.txt")
    with open(output_file, "w") as f:
        f.write(generated_text)
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a T5 model.")
    parser.add_argument("input_prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--output_dir", default="generated_output", help="Output directory to save generated text")
    parser.add_argument("--model_checkpoint", default="/home/kris/Project/Proposed-working/CodeCrafter/saved_models/checkpoint-120/", help="Path to the model checkpoint")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output
    generate_output(args.input_prompt, args.output_dir, args.model_checkpoint, args.device)
