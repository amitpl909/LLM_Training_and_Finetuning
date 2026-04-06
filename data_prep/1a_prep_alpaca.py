import json
from datasets import load_dataset

def main():
    print("Downloading Alpaca-cleaned dataset...")
    # yahma/alpaca-cleaned is a standard, high-quality variant of the Stanford dataset
    dataset = load_dataset("yahma/alpaca-cleaned")["train"]
    
    # Shuffle the dataset using a fixed seed for reproducibility
    dataset = dataset.shuffle(seed=42)
    
    # Split 100 examples for the held-out evaluation set
    split = dataset.train_test_split(test_size=100)
    
    train_data = split["train"].to_list()
    eval_data = split["test"].to_list()
    
    # Save to disk in the standard (instruction, input, output) schema
    with open("data_prep/alpaca_train.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open("data_prep/alpaca_eval.json", "w") as f:
        json.dump(eval_data, f, indent=4)
        
    print(f"Saved {len(train_data)} training examples to data_prep/alpaca_train.json")
    print(f"Saved {len(eval_data)} evaluation examples to data_prep/alpaca_eval.json")

if __name__ == "__main__":
    main()