"""
Expand JSON evaluation set from 26 to 100 prompts.
Generates 20 prompts for each of the 5 required task types.
Uses teacher model (Llama 3.3-70B) to generate gold-standard responses.
"""

import json
import yaml
import random
from openai import OpenAI

# Load config
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize teacher client
client = OpenAI(
    base_url=config["teacher_api_url"],
    api_key=config["teacher_api_key"]
)

# Task type templates
TASK_TEMPLATES = {
    "extraction": [
        "Extract {entities} from the following text",
        "Parse the text and extract {entities} into a JSON object",
        "Identify {entities} and return them as valid JSON",
        "From the given text, extract {entities}",
        "Pull out {entities} from the passage and format as JSON",
    ],
    "schema": [
        "Generate a JSON object following this schema: {schema}. Topic: {topic}",
        "Create valid JSON matching schema {schema} for: {topic}",
        "Produce JSON conforming to schema {schema} about {topic}",
        "Return JSON for {topic} using only fields from schema: {schema}",
        "Build JSON with schema {schema} describing: {topic}",
    ],
    "classification": [
        "Classify the sentiment as one of {labels} and return as JSON",
        "Determine the category as {labels} and output as JSON",
        "Classify the emotion as {labels} and format as JSON",
        "Return the intent as one of {labels} in JSON format",
        "Identify the topic as {labels} and output valid JSON",
    ],
    "repair": [
        "Fix the malformed JSON: {broken_json}",
        "Repair and return valid JSON from: {broken_json}",
        "Correct the JSON errors in: {broken_json}",
        "Convert the broken JSON to valid JSON: {broken_json}",
        "Fix the JSON formatting in: {broken_json}",
    ],
    "tool_call": [
        "Generate a JSON function call for {function_name} with args {args} to {action}",
        "Create a JSON object representing a call to {function_name} with {args} to {action}",
        "Return JSON for calling {function_name} with parameters {args} to {action}",
        "Produce JSON for invoking {function_name} with {args} to {action}",
        "Generate a function call JSON for {function_name} with {args} to {action}",
    ]
}

# Sample data for generation
SAMPLE_TEXTS = [
    "Apple Inc. announced new products at their Cupertino headquarters on March 15, 2024.",
    "Dr. Sarah Johnson from Stanford University published a paper on AI safety in January 2024.",
    "The conference will be held in New York on July 4th with guests from Google and Microsoft.",
    "Netflix released a new series featuring actor Tom Cruise, filmed in Sydney.",
    "The Eiffel Tower in Paris receives 7 million visitors yearly since 2023.",
]

SCHEMAS = [
    '{"person": "string", "company": "string", "location": "string"}',
    '{"title": "string", "author": "string", "date": "string"}',
    '{"name": "string", "role": "string", "department": "string"}',
    '{"product": "string", "price": "number", "availability": "boolean"}',
    '{"event": "string", "location": "string", "date": "string"}',
]

LABELS = [
    "positive, neutral, negative",
    "urgent, normal, low-priority",
    "happy, sad, angry, neutral",
    "bug, feature-request, documentation",
    "valid, suspicious, invalid",
]

BROKEN_JSONS = [
    '{"name": "John" "age": 30}',  # missing comma
    '{name: "Jane", age: 25}',  # quotes missing
    '{"items": [1, 2, 3,]}',  # trailing comma
    '{"key": "value"',  # missing closing brace
    '{"nested": {"inner": "value"}',  # incomplete nesting
]

FUNCTIONS = [
    ("send_email", "to, subject, body", "send a notification"),
    ("create_user", "username, email, role", "register a new user"),
    ("update_database", "table, id, fields", "modify existing records"),
    ("fetch_data", "database, query, limit", "retrieve information"),
    ("process_payment", "amount, currency, method", "complete a transaction"),
]

def get_teacher_response(task_instruction: str) -> str:
    """Query teacher model for JSON response."""
    try:
        response = client.chat.completions.create(
            model=config["teacher_model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON generation expert. Respond ONLY with valid JSON, no explanations."
                },
                {"role": "user", "content": task_instruction}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying teacher: {e}")
        return None

def validate_json(response: str) -> bool:
    """Check if response is valid JSON."""
    try:
        json.loads(response)
        return True
    except:
        return False

def generate_extraction_prompts(count: int) -> list:
    """Generate JSON extraction task prompts."""
    prompts = []
    entities = [
        "name, location, date",
        "person, organization, location",
        "author, title, publication_date",
        "company, product, price",
        "event, venue, attendee_count",
    ]
    
    for i in range(count):
        template = random.choice(TASK_TEMPLATES["extraction"])
        text = random.choice(SAMPLE_TEXTS)
        entity = random.choice(entities)
        instruction = template.format(entities=entity)
        
        full_instruction = f"{instruction}:\n\n{text}"
        response = get_teacher_response(full_instruction)
        
        if response and validate_json(response):
            prompts.append({
                "instruction": instruction,
                "input": text,
                "output": response,
                "task_type": "extraction"
            })
            print(f"✓ Extraction {len(prompts)}/{count}")
        else:
            print(f"✗ Extraction failed, retrying...")
            i -= 1  # Retry
    
    return prompts

def generate_classification_prompts(count: int) -> list:
    """Generate classification task prompts."""
    prompts = []
    texts = [
        "I absolutely love this product! Best purchase ever!",
        "This is okay, nothing special.",
        "Terrible experience, would not recommend.",
        "The system crashed and lost all my data!",
        "Great service, very helpful staff.",
    ]
    
    for i in range(count):
        template = random.choice(TASK_TEMPLATES["classification"])
        text = random.choice(texts)
        labels = random.choice(LABELS)
        instruction = template.format(labels=labels)
        
        full_instruction = f"{instruction}: \"{text}\""
        response = get_teacher_response(full_instruction)
        
        if response and validate_json(response):
            prompts.append({
                "instruction": instruction,
                "input": text,
                "output": response,
                "task_type": "classification"
            })
            print(f"✓ Classification {len(prompts)}/{count}")
        else:
            print(f"✗ Classification failed, retrying...")
            i -= 1
    
    return prompts

def generate_schema_prompts(count: int) -> list:
    """Generate schema-constrained generation prompts."""
    prompts = []
    topics = [
        "a new smartphone release",
        "a company employee profile",
        "a recent conference event",
        "a movie review",
        "a weather report",
    ]
    
    for i in range(count):
        template = random.choice(TASK_TEMPLATES["schema"])
        schema = random.choice(SCHEMAS)
        topic = random.choice(topics)
        instruction = template.format(schema=schema, topic=topic)
        
        response = get_teacher_response(instruction)
        
        if response and validate_json(response):
            prompts.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "task_type": "schema"
            })
            print(f"✓ Schema {len(prompts)}/{count}")
        else:
            print(f"✗ Schema failed, retrying...")
            i -= 1
    
    return prompts

def generate_repair_prompts(count: int) -> list:
    """Generate JSON repair task prompts."""
    prompts = []
    
    for i in range(count):
        template = random.choice(TASK_TEMPLATES["repair"])
        broken = random.choice(BROKEN_JSONS)
        instruction = template.format(broken_json=broken)
        
        response = get_teacher_response(instruction)
        
        if response and validate_json(response):
            prompts.append({
                "instruction": "Fix the malformed JSON",
                "input": broken,
                "output": response,
                "task_type": "repair"
            })
            print(f"✓ Repair {len(prompts)}/{count}")
        else:
            print(f"✗ Repair failed, retrying...")
            i -= 1
    
    return prompts

def generate_tool_call_prompts(count: int) -> list:
    """Generate tool-call/function-call JSON prompts."""
    prompts = []
    
    for i in range(count):
        template = random.choice(TASK_TEMPLATES["tool_call"])
        func_name, args, action = random.choice(FUNCTIONS)
        instruction = template.format(function_name=func_name, args=args, action=action)
        
        response = get_teacher_response(instruction)
        
        if response and validate_json(response):
            prompts.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "task_type": "tool_call"
            })
            print(f"✓ Tool call {len(prompts)}/{count}")
        else:
            print(f"✗ Tool call failed, retrying...")
            i -= 1
    
    return prompts

def main():
    print("=" * 60)
    print("Expanding JSON Eval Set from 26 to 100 Prompts")
    print("=" * 60)
    
    all_prompts = []
    
    print("\nGenerating extraction prompts (20)...")
    all_prompts.extend(generate_extraction_prompts(20))
    
    print("\nGenerating classification prompts (20)...")
    all_prompts.extend(generate_classification_prompts(20))
    
    print("\nGenerating schema prompts (20)...")
    all_prompts.extend(generate_schema_prompts(20))
    
    print("\nGenerating repair prompts (20)...")
    all_prompts.extend(generate_repair_prompts(20))
    
    print("\nGenerating tool-call prompts (20)...")
    all_prompts.extend(generate_tool_call_prompts(20))
    
    # Save expanded eval set
    with open("stage2_json_instruct_eval_expanded.json", "w") as f:
        json.dump(all_prompts, f, indent=2)
    
    print(f"\n✅ Generated {len(all_prompts)} evaluation prompts")
    print("Task type distribution:")
    from collections import Counter
    types = Counter([p["task_type"] for p in all_prompts])
    for ttype, count in types.items():
        print(f"  {ttype}: {count}")

if __name__ == "__main__":
    main()
