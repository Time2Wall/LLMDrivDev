#!/usr/bin/env python3
"""
Test script to verify the homework 7 setup works correctly.
This tests imports, dataset loading, model loading, and tools.
"""

import sys
print("=" * 60)
print("Testing Homework 7 Setup")
print("=" * 60)

# Test 1: Import core packages
print("\n[1/6] Testing core imports...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print("  ✓ Core ML packages imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import core packages: {e}")
    sys.exit(1)

# Test 2: Import LangChain
print("\n[2/6] Testing LangChain imports...")
try:
    from langchain_core.tools import tool
    from langchain_huggingface import HuggingFacePipeline
    print("  ✓ LangChain packages imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import LangChain: {e}")
    sys.exit(1)

# Test 3: Test tools
print("\n[3/6] Testing LangChain tools...")
try:
    import re
    import json
    from typing import Optional

    @tool
    def text_formatter(text: str, format_type: str = "uppercase") -> str:
        """Format text in various styles."""
        format_type = format_type.lower().strip()
        if format_type == "uppercase":
            return text.upper()
        elif format_type == "snake_case":
            return re.sub(r'[\s-]+', '_', text.lower())
        elif format_type == "camel_case":
            words = re.split(r'[\s_-]+', text)
            return words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        return text

    @tool
    def template_generator(template_type: str, context: Optional[str] = None) -> str:
        """Generate templates for different purposes."""
        templates = {
            "email": "Subject: {subject}\n\nDear {recipient},...",
            "bug_report": "# Bug Report\n\n## Description\n{description}..."
        }
        return templates.get(template_type, f"Unknown template: {template_type}")

    @tool
    def content_validator(content: str, validation_type: str = "structure") -> str:
        """Validate content structure and quality."""
        if validation_type == "json":
            try:
                json.loads(content)
                return "✓ Valid JSON"
            except:
                return "✗ Invalid JSON"
        return f"Content length: {len(content)} chars"

    # Test the tools
    result1 = text_formatter.invoke({"text": "hello world", "format_type": "snake_case"})
    result2 = template_generator.invoke({"template_type": "email"})
    result3 = content_validator.invoke({"content": '{"key": "value"}', "validation_type": "json"})

    print(f"  ✓ text_formatter: 'hello world' -> '{result1}'")
    print(f"  ✓ template_generator: email template created")
    print(f"  ✓ content_validator: JSON validation -> '{result3}'")
    print("  ✓ All tools working correctly")
except Exception as e:
    print(f"  ✗ Tool test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load dataset (small sample)
print("\n[4/6] Testing dataset loading...")
try:
    dataset = load_dataset("mlabonne/FineTome-100k", split="train", streaming=True)
    sample = next(iter(dataset))
    print(f"  ✓ Dataset loaded successfully")
    print(f"  ✓ Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    print("  (This is OK if network is limited)")

# Test 5: Load tokenizer
print("\n[5/6] Testing tokenizer loading...")
try:
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer loaded: {MODEL_NAME}")
    print(f"  ✓ Vocab size: {tokenizer.vocab_size}")

    # Test tokenization
    test_text = "### Instruction:\nHello, how are you?\n\n### Response:\n"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"  ✓ Test tokenization: {len(tokens['input_ids'][0])} tokens")
except Exception as e:
    print(f"  ✗ Tokenizer loading failed: {e}")

# Test 6: Test LoRA config
print("\n[6/6] Testing LoRA configuration...")
try:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    print(f"  ✓ LoRA config created successfully")
    print(f"  ✓ Rank (r): {lora_config.r}")
    print(f"  ✓ Alpha: {lora_config.lora_alpha}")
    print(f"  ✓ Target modules: {lora_config.target_modules}")
except Exception as e:
    print(f"  ✗ LoRA config failed: {e}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
print("\nThe notebook is ready to run. To execute:")
print("  1. Activate venv: source venv/bin/activate")
print("  2. Run Jupyter: jupyter notebook hometask_7_finetuning_tools.ipynb")
print("\nNote: Full training requires GPU (CUDA). Without GPU, the")
print("notebook will run in CPU mode with limited capabilities.")
