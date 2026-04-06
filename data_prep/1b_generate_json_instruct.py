import json
import yaml
import time
from openai import OpenAI

# Load central configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize OpenAI client using settings from config.yaml
client = OpenAI(
    api_key=config["teacher_api_key"],
    base_url=config["teacher_api_url"]
)

def generate_and_validate(instruction, input_text, max_retries=3):
    """Generate JSON from teacher model and validate it."""
    prompt = f"{instruction}\n\nInput: {input_text}" if input_text else instruction
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config["teacher_model"],
                messages=[
                    {"role": "system", "content": "You are a helpful data generation assistant. You MUST respond ONLY with raw, valid JSON. Do not include markdown formatting like ```json or any conversational text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=512
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # Clean markdown if model hallucinates it
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()
                
            # Validation: Must parse as valid JSON
            parsed_json = json.loads(raw_output)
            
            return {
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(parsed_json)
            }
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Attempt {attempt + 1} failed: {str(e)[:50]}")
            time.sleep(1)
            
    print(f"  ❌ Failed to generate valid JSON")
    return None

def generate_task_variations(task_type, base_instruction, base_inputs, num_variations=20):
    """Generate multiple variations of a task to create diverse examples."""
    examples = []
    
    variations = {
        "extraction": [
            ("Extract entities from: Apple Inc. announced new products at their Cupertino headquarters on March 15, 2024.", 
             "Extract person, company, location, and date"),
            ("Extract all named entities (PERSON, ORG, LOC, DATE) from: Tesla CEO Elon Musk presented earnings on February 1, 2024 in Austin, Texas.",
             "Extract entities"),
            ("From the text: 'Jane Smith works at Google in Mountain View since 2021', extract all persons, companies, locations, and years.",
             "Extract persons, companies, locations, years"),
            ("Extract: FDA approved Pfizer vaccine on December 10, 2020 in Washington DC.",
             "Extract organizations, locations, dates"),
            ("Text: Microsoft CEO Satya Nadella opened the conference in Seattle on April 20, 2024. Extract all proper nouns.",
             "Extract proper nouns"),
            ("Extract from: Amazon headquarters relocated from Seattle to Arlington, Virginia in 2023.",
             "Extract company, locations, dates"),
            ("Parse: LinkedIn founder Reid Hoffman announced IPO in 2011 from San Francisco offices.",
             "Extract persons, companies, locations, dates"),
            ("Extract entities: OpenAI released ChatGPT in November 2022 from San Francisco.",
             "Extract organization, products, dates, locations"),
        ],
        "schema_constrained": [
            ("", "Generate a random user profile JSON with fields: username (string), age (int), premium (bool), interests (list)"),
            ("", "Create a product listing JSON: name, price, stock_count, is_available, tags"),
            ("", "Generate a blog post JSON: title, author, date, word_count, tags"),
            ("", "Create a movie info JSON: title, genre, release_year, rating, duration_minutes"),
            ("", "Generate a person record JSON: first_name, last_name, email, phone, is_active"),
            ("", "Create a weather report JSON: temperature_celsius, humidity_percent, wind_speed, precipitation"),
            ("", "Generate a book entry JSON: title, author, isbn, publish_year, pages"),
            ("", "Create a restaurant info JSON: name, cuisine_type, rating, address, is_open"),
        ],
        "exact_label_classification": [
            ("I love this product! Best purchase ever.", "Classify sentiment as positive, neutral, or negative"),
            ("The service was terrible and the staff was rude.", "Classify sentiment"),
            ("This is a chair.", "Classify sentiment"),
            ("Not bad, but could be better.", "Classify sentiment"),
            ("Absolutely amazing! Worth every penny!", "Classify sentiment"),
            ("The quality is poor and overpriced.", "Classify sentiment"),
            ("It works as expected, nothing special.", "Classify sentiment"),
            ("I would recommend this to everyone!", "Classify sentiment"),
        ],
        "json_repair": [
            ("{ 'name': 'John', \"age\": 30 }", "Fix invalid JSON syntax"),
            ("{ name: John, age: 30, city: 'NYC' }", "Repair malformed JSON"),
            ("{ \"items\": [1, 2, 3,] }", "Fix trailing comma"),
            ("{ 'key': 'value', }", "Repair JSON with syntax errors"),
            ("{ \"incomplete\": \"data\" ", "Complete incomplete JSON"),
            ("{ 'nested': { 'key': 'value' } }", "Fix quote mismatch"),
            ("{ \"array\": [\"a\", \"b\", 'c'] }", "Normalize quotes in JSON"),
            ("{ key: 'value', 'another': 123 }", "Fix unquoted keys"),
        ],
        "function_call": [
            ("Check if the flight BA402 is on time tomorrow.", "Generate call: get_flight_status(flight_number, date)"),
            ("Get weather for New York.", "Generate call: get_weather(location)"),
            ("Send an email to alice@example.com with subject 'Hello'.", "Generate call: send_email(recipient, subject, body)"),
            ("Book a flight from NYC to LA on 2024-04-15.", "Generate call: book_flight(origin, destination, date)"),
            ("Get the price of Apple stock.", "Generate call: get_stock_price(symbol)"),
            ("Create a reminder for tomorrow at 10am.", "Generate call: create_reminder(message, datetime)"),
            ("Search for restaurants in San Francisco with Italian cuisine.", "Generate call: search_restaurants(location, cuisine)"),
            ("Play music by The Beatles.", "Generate call: play_music(artist, genre)"),
        ],
    }
    
    # Get variations for this task type, default to base if not found
    task_variations = variations.get(task_type, [(base_inputs[0] if base_inputs else "", base_instruction)] * num_variations)
    
    # Generate examples for each variation (up to num_variations)
    for i, (variant_input, variant_instruction) in enumerate(task_variations[:num_variations]):
        if i >= num_variations:
            break
            
        print(f"  Generating variation {i+1}/{num_variations}...")
        example = generate_and_validate(variant_instruction, variant_input)
        if example:
            examples.append(example)
        else:
            print(f"    Skipped variation {i+1}")
        time.sleep(0.5)  # Rate limiting
            
    return examples

def main():
    print("=" * 70)
    print("PHASE 1b: Teacher-Generated JSON Instruct Dataset Construction")
    print("=" * 70)
    
    # Load base prompt templates
    with open("prompts/teacher_prompts.json", "r") as f:
        task_templates = json.load(f)
    
    training_data = []
    total_examples = 0
    
    for template in task_templates:
        task_type = template["task_type"]
        print(f"\n[{task_type.upper()}]")
        print(f"Generating 20 variations using Llama 3.1 70B teacher model...")
        
        # Generate 20 variations per task type
        examples = generate_task_variations(
            task_type,
            template["instruction"],
            [template["input"]] if template["input"] else [],
            num_variations=20
        )
        
        training_data.extend(examples)
        total_examples += len(examples)
        print(f"✓ Generated {len(examples)}/20 examples for {task_type}")
    
    # Save the training dataset
    output_path = "data_prep/stage2_json_instruct_train.json"
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ SUCCESS: Generated and saved {total_examples} training examples")
    print(f"  Location: {output_path}")
    print("=" * 70)
    
    # Create evaluation set (separate, held-out)
    eval_data = training_data[::2]  # Every other example for eval
    eval_path = "data_prep/stage2_json_instruct_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✓ Evaluation set: {len(eval_data)} examples saved to {eval_path}")

if __name__ == "__main__":
    main()