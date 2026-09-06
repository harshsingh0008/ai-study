# ============================================
# TEST YOUR FINE-TUNED MODEL
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load base model and tokenizer
print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Load your fine-tuned adapter
print("Loading your fine-tuned model...")
model_path = "./study-coach"
model = PeftModel.from_pretrained(base_model, model_path)
model = model.to(device)
model.eval()

print("✅ Model loaded successfully!\n")

# Test cases
test_dialogues = [
    "#Person1#: I need to see my test results. #Person2#: Come in this afternoon.",
    "#Person1#: Can you help me with this problem? #Person2#: Sure, let me take a look.",
    "#Person1#: Thank you for your help. #Person2#: You're welcome, anytime.",
]

def generate_summary(text):
    prompt = f"Summarize: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run tests
print("="*50)
print("🧪 TESTING YOUR MODEL")
print("="*50)

for i, dialogue in enumerate(test_dialogues, 1):
    print(f"\n📌 Test {i}:")
    print(f"Dialogue: {dialogue}")
    
    summary = generate_summary(dialogue)
    print(f"Summary: {summary}")
    print("-"*40)

print("\n✅ Testing complete!")