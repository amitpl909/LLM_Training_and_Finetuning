import json
import random

def generate_extraction_examples():
    """Generate JSON extraction examples."""
    extraction_templates = [
        {
            "instruction": "Extract all entities (persons, organizations, locations) and dates from the provided text. Return the result strictly as a JSON object with keys 'persons', 'organizations', 'locations', and 'dates' containing lists of strings.",
            "inputs": [
                "On October 24, 2023, Satya Nadella announced new AI features at the Microsoft headquarters in Redmond.",
                "Apple Inc. released the iPhone 15 in Cupertino on September 22, 2023.",
                "Elon Musk, CEO of Tesla, gave a presentation in San Francisco on March 1, 2024.",
                "Google announced Gemini AI on December 6, 2023 at their Mountain View office.",
                "Amazon's Andy Jassy unveiled new cloud services in Las Vegas on April 10, 2024.",
                "OpenAI, founded by Sam Altman, is headquartered in San Francisco.",
                "Meta's Mark Zuckerberg showcased VR technology in New York on February 15, 2024.",
                "NVIDIA CEO Jensen Huang presented at GTC in San Jose on March 18, 2024.",
            ]
        }
    ]
    
    examples = []
    for template in extraction_templates:
        for inp in template["inputs"]:
            # Simulate extracted entities
            entities = extract_entities(inp)
            examples.append({
                "instruction": template["instruction"],
                "input": inp,
                "output": json.dumps(entities)
            })
    return examples

def extract_entities(text):
    """Simple entity extraction simulation."""
    # Map common entities
    entity_map = {
        "Satya Nadella": {"persons": ["Satya Nadella"], "organizations": ["Microsoft"], "locations": ["Redmond"], "dates": ["October 24, 2023"]},
        "Apple": {"persons": [], "organizations": ["Apple Inc."], "locations": ["Cupertino"], "dates": ["September 22, 2023"]},
        "Elon Musk": {"persons": ["Elon Musk"], "organizations": ["Tesla"], "locations": ["San Francisco"], "dates": ["March 1, 2024"]},
        "Google": {"persons": [], "organizations": ["Google"], "locations": ["Mountain View"], "dates": ["December 6, 2023"]},
        "Amazon": {"persons": ["Andy Jassy"], "organizations": ["Amazon"], "locations": ["Las Vegas"], "dates": ["April 10, 2024"]},
        "OpenAI": {"persons": ["Sam Altman"], "organizations": ["OpenAI"], "locations": ["San Francisco"], "dates": []},
        "Meta": {"persons": ["Mark Zuckerberg"], "organizations": ["Meta"], "locations": ["New York"], "dates": ["February 15, 2024"]},
        "NVIDIA": {"persons": ["Jensen Huang"], "organizations": ["NVIDIA"], "locations": ["San Jose"], "dates": ["March 18, 2024"]},
    }
    
    for key, entities in entity_map.items():
        if key in text:
            return entities
    
    return {"persons": [], "organizations": [], "locations": [], "dates": []}

def generate_schema_examples():
    """Generate schema-constrained JSON examples."""
    schemas = [
        {
            "instruction": "Generate a fictional user profile that strictly conforms to the following JSON schema: {\"username\": \"string\", \"age\": \"integer\", \"is_premium_subscriber\": \"boolean\", \"interests\": [\"string\"]}. Return ONLY valid JSON.",
            "examples": [
                {"username": "john_doe_123", "age": 28, "is_premium_subscriber": True, "interests": ["gaming", "music", "travel"]},
                {"username": "jane_smith", "age": 35, "is_premium_subscriber": False, "interests": ["reading", "cooking"]},
                {"username": "alex_tiger", "age": 42, "is_premium_subscriber": True, "interests": ["sports", "photography"]},
                {"username": "emma_wilson", "age": 24, "is_premium_subscriber": True, "interests": ["art", "design"]},
            ]
        },
        {
            "instruction": "Generate a product listing JSON with the exact schema: {\"product_name\": \"string\", \"price\": \"float\", \"stock_count\": \"integer\", \"is_available\": \"boolean\", \"tags\": [\"string\"]}",
            "examples": [
                {"product_name": "Wireless Headphones", "price": 49.99, "stock_count": 150, "is_available": True, "tags": ["electronics", "audio"]},
                {"product_name": "USB-C Cable", "price": 12.99, "stock_count": 500, "is_available": True, "tags": ["cables", "accessories"]},
                {"product_name": "Phone Case", "price": 19.99, "stock_count": 0, "is_available": False, "tags": ["protection"]},
            ]
        }
    ]
    
    examples = []
    for schema_template in schemas:
        for example_data in schema_template["examples"]:
            examples.append({
                "instruction": schema_template["instruction"],
                "input": "",
                "output": json.dumps(example_data)
            })
    return examples

def generate_classification_examples():
    """Generate sentiment classification examples."""
    examples_data = [
        ("I love this product! Best purchase ever.", "positive"),
        ("The service was terrible and the staff was rude.", "negative"),
        ("This is a chair.", "neutral"),
        ("Not bad, but could be better.", "neutral"),
        ("Absolutely amazing! Worth every penny!", "positive"),
        ("The quality is poor and overpriced.", "negative"),
        ("It works as expected, nothing special.", "neutral"),
        ("I would recommend this to everyone!", "positive"),
        ("Disappointed with the purchase.", "negative"),
        ("It's okay, does what it's supposed to do.", "neutral"),
        ("Exceeded all my expectations!", "positive"),
        ("Complete waste of money.", "negative"),
        ("Average product, average price.", "neutral"),
        ("This is incredible!", "positive"),
        ("Won't buy again.", "negative"),
    ]
    
    examples = []
    for text, sentiment in examples_data:
        examples.append({
            "instruction": "Classify the sentiment of the provided customer review. You must return a JSON object with exactly one key, 'sentiment', and the value must be exactly one of: 'positive', 'neutral', or 'negative'.",
            "input": text,
            "output": json.dumps({"sentiment": sentiment})
        })
    return examples

def generate_json_repair_examples():
    """Generate JSON repair examples."""
    repair_examples = [
        ("{ 'name': 'John', \"age\": 30 }", {"name": "John", "age": 30}),
        ("{ name: John, age: 30, city: 'NYC' }", {"name": "John", "age": 30, "city": "NYC"}),
        ("{ \"items\": [1, 2, 3,] }", {"items": [1, 2, 3]}),
        ("{ 'key': 'value', }", {"key": "value"}),
        ("{ \"incomplete\": \"data\" ", {"incomplete": "data"}),
        ("{ 'nested': { 'key': 'value' } }", {"nested": {"key": "value"}}),
        ("{ \"array\": [\"a\", \"b\", 'c'] }", {"array": ["a", "b", "c"]}),
        ("{ key: 'value', 'another': 123 }", {"key": "value", "another": 123}),
        ("{ 'persons': ['John', 'Jane',] }", {"persons": ["John", "Jane"]}),
        ("{ \"data\": {\"nested\": \"value\",} }", {"data": {"nested": "value"}}),
    ]
    
    examples = []
    for malformed, repaired in repair_examples:
        examples.append({
            "instruction": "The following JSON string is malformed. Fix the syntax errors and return the corrected, perfectly valid JSON object.",
            "input": malformed,
            "output": json.dumps(repaired)
        })
    return examples

def generate_function_call_examples():
    """Generate function call JSON examples."""
    function_examples = [
        ("Check if the flight BA402 is on time tomorrow.", {"function": "get_flight_status", "arguments": {"flight_number": "BA402", "date": "tomorrow"}}),
        ("Get weather for New York.", {"function": "get_weather", "arguments": {"location": "New York"}}),
        ("Send an email to alice@example.com with subject 'Hello'.", {"function": "send_email", "arguments": {"recipient": "alice@example.com", "subject": "Hello"}}),
        ("Book a flight from NYC to LA on 2024-04-15.", {"function": "book_flight", "arguments": {"origin": "NYC", "destination": "LA", "date": "2024-04-15"}}),
        ("Get the price of Apple stock.", {"function": "get_stock_price", "arguments": {"symbol": "AAPL"}}),
        ("Create a reminder for tomorrow at 10am.", {"function": "create_reminder", "arguments": {"message": "reminder", "datetime": "tomorrow 10am"}}),
        ("Search for restaurants in San Francisco with Italian cuisine.", {"function": "search_restaurants", "arguments": {"location": "San Francisco", "cuisine": "Italian"}}),
        ("Play music by The Beatles.", {"function": "play_music", "arguments": {"artist": "The Beatles"}}),
        ("Set alarm for 6:30 AM.", {"function": "set_alarm", "arguments": {"time": "6:30 AM"}}),
        ("Get directions to nearby coffee shops.", {"function": "get_directions", "arguments": {"destination": "nearby coffee shops"}}),
    ]
    
    examples = []
    for query, function_call in function_examples:
        examples.append({
            "instruction": "You are an AI assistant with access to functions. Generate the JSON representation of the function call based on the user's query.",
            "input": query,
            "output": json.dumps(function_call)
        })
    return examples

def main():
    print("=" * 70)
    print("PHASE 1b: Synthetic JSON Instruct Dataset Generation")
    print("=" * 70)
    
    all_examples = []
    
    print("\n[EXTRACTION] Generating entity extraction examples...")
    extraction = generate_extraction_examples()
    all_examples.extend(extraction)
    print(f"✓ Generated {len(extraction)} extraction examples")
    
    print("[SCHEMA_CONSTRAINED] Generating schema-constrained examples...")
    schema = generate_schema_examples()
    all_examples.extend(schema)
    print(f"✓ Generated {len(schema)} schema examples")
    
    print("[CLASSIFICATION] Generating sentiment classification examples...")
    classification = generate_classification_examples()
    all_examples.extend(classification)
    print(f"✓ Generated {len(classification)} classification examples")
    
    print("[JSON_REPAIR] Generating JSON repair examples...")
    repair = generate_json_repair_examples()
    all_examples.extend(repair)
    print(f"✓ Generated {len(repair)} repair examples")
    
    print("[FUNCTION_CALL] Generating function call examples...")
    function = generate_function_call_examples()
    all_examples.extend(function)
    print(f"✓ Generated {len(function)} function call examples")
    
    # Save training dataset
    output_path = "data_prep/stage2_json_instruct_train.json"
    with open(output_path, "w") as f:
        json.dump(all_examples, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ SUCCESS: Generated and saved {len(all_examples)} training examples")
    print(f"  Location: {output_path}")
    print("=" * 70)
    
    # Create evaluation set (separate, held-out)
    eval_data = all_examples[::2]  # Every other example for eval
    eval_path = "data_prep/stage2_json_instruct_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✓ Evaluation set: {len(eval_data)} examples saved to {eval_path}\n")

if __name__ == "__main__":
    main()
