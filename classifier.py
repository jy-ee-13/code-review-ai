# classifier.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # suppress loading bars

MODEL_DIR = os.path.join(os.path.dirname(__file__), "training", "severity_model")
BASE_MODEL = "microsoft/phi-2"

ID2LABEL = {0: "critical", 1: "warning", 2: "suggestion"}

print("[Classifier] Loading fine-tuned severity model...")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_tokenizer.pad_token = _tokenizer.eos_token

# Step 1: Load the base model fresh
_base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3,
    dtype=torch.float32,
    trust_remote_code=True,
)
_base_model.config.pad_token_id = _tokenizer.eos_token_id

# Step 2: Load the LoRA adapter ON TOP of the base model
_model = PeftModel.from_pretrained(_base_model, MODEL_DIR)

# Step 3: Merge LoRA weights into the base model for faster inference
_model = _model.merge_and_unload()
_model.eval()

print("[Classifier] Model ready.")


def classify(issue_description: str) -> str:
    """
    Takes an issue description string.
    Returns one of: 'critical', 'warning', 'suggestion'
    """
    inputs = _tokenizer(
        issue_description,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    return ID2LABEL[predicted_id]


if __name__ == "__main__":
    test_cases = [
        ("Undefined variable 'db' used before assignment", "critical"),
        ("No test coverage for function process()", "warning"),
        ("Missing docstring on public method", "suggestion"),
        ("Division by zero will occur at runtime", "critical"),
        ("Variable name is not descriptive", "suggestion"),
    ]

    print("\n=== Classifier Smoke Test ===")
    correct = 0
    for text, expected in test_cases:
        predicted = classify(text)
        status = "✅" if predicted == expected else "❌"
        print(f"{status} Expected: {expected:10} | Got: {predicted:10} | '{text}'")
        if predicted == expected:
            correct += 1
    print(f"\nAccuracy: {correct}/{len(test_cases)}")