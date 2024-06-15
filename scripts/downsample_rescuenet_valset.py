import json
import random

keep_probs = {
    'connectivity': 0.2,
    'seg': 0.1,
    'counting': 0.05,
    'existence': 0.02,
}

def process_jsonl(input_file, output_file):
    """Reads a jsonl file and writes each JSON object to a new jsonl file in one loop."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            # keep_prob = keep_probs[json_obj['type']]
            keep_prob = 0.01
            if (random.random() < keep_prob):
                outfile.write(json.dumps(json_obj) + '\n')

if __name__ == "__main__":
    input_file = 'rescuenet_agent_val_small.json'  # Replace with your input file path
    output_file = 'rescuenet_agent_val_tiny.json'  # Replace with your output file path
    process_jsonl(input_file, output_file)
