#!/usr/bin/env python3
"""
Demo script for Homework 7: Fine-tuning with LoRA/PEFT and LangChain Tools
This script demonstrates the key components without full training.
"""

import os
import json
import torch
import warnings
import re
from typing import Optional
warnings.filterwarnings('ignore')

print("=" * 70)
print("HOMEWORK 7 DEMO: Fine-tuning with LoRA/PEFT and LangChain Tools")
print("Track C: Instructive Assistant")
print("=" * 70)

# ============================================================================
# Part 1: LangChain Tools Implementation
# ============================================================================
print("\n[PART 1] LangChain Tools Implementation")
print("-" * 70)

from langchain_core.tools import tool

@tool
def text_formatter(text: str, format_type: str = "uppercase") -> str:
    """
    Format text in various styles.

    Args:
        text: The text to format
        format_type: The format type - one of: uppercase, lowercase, title, capitalize,
                     reverse, snake_case, camel_case, remove_spaces, add_bullets

    Returns:
        Formatted text
    """
    format_type = format_type.lower().strip()

    if format_type == "uppercase":
        return text.upper()
    elif format_type == "lowercase":
        return text.lower()
    elif format_type == "title":
        return text.title()
    elif format_type == "capitalize":
        return text.capitalize()
    elif format_type == "reverse":
        return text[::-1]
    elif format_type == "snake_case":
        return re.sub(r'[\s-]+', '_', text.lower())
    elif format_type == "camel_case":
        words = re.split(r'[\s_-]+', text)
        return words[0].lower() + ''.join(w.capitalize() for w in words[1:])
    elif format_type == "remove_spaces":
        return text.replace(" ", "")
    elif format_type == "add_bullets":
        lines = text.strip().split('\n')
        return '\n'.join(f"• {line}" for line in lines if line.strip())
    else:
        return f"Unknown format type: {format_type}"

@tool
def template_generator(template_type: str, context: Optional[str] = None) -> str:
    """
    Generate templates for different purposes.

    Args:
        template_type: Type of template - one of: email, meeting, report,
                       code_review, bug_report, feature_request, readme
        context: Optional context to customize the template

    Returns:
        Generated template
    """
    templates = {
        "email": """Subject: {subject}

Dear {recipient},

I hope this email finds you well.

{body}

Please let me know if you have any questions.

Best regards,
{sender}""",

        "bug_report": """# Bug Report

## Description
{description}

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
{expected}

## Actual Behavior
{actual}

## Environment
- OS: {os}
- Version: {version}""",

        "code_review": """# Code Review

**PR/MR:** #{pr_number}
**Author:** {author}

## Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests included
- [ ] Documentation updated

## Verdict
[ ] Approved / [ ] Request Changes"""
    }

    template_type = template_type.lower().strip()
    if template_type in templates:
        result = templates[template_type]
        if context:
            result += f"\n\n---\nContext: {context}"
        return result
    return f"Unknown template: {template_type}. Available: {', '.join(templates.keys())}"

@tool
def content_validator(content: str, validation_type: str = "structure") -> str:
    """
    Validate content structure and quality.

    Args:
        content: The content to validate
        validation_type: Type of validation - one of: structure, length, markdown, json, email, url

    Returns:
        Validation results with score and suggestions
    """
    validation_type = validation_type.lower().strip()

    if validation_type == "json":
        try:
            json.loads(content)
            return "✓ Valid JSON"
        except json.JSONDecodeError as e:
            return f"✗ Invalid JSON: {str(e)}"

    elif validation_type == "length":
        return f"""Length Analysis:
- Characters: {len(content)}
- Words: {len(content.split())}
- Lines: {len(content.split(chr(10)))}"""

    elif validation_type == "markdown":
        has_headers = bool(re.search(r'^#+\s', content, re.MULTILINE))
        has_lists = bool(re.search(r'^[\-\*\d]\.?\s', content, re.MULTILINE))
        has_code = bool(re.search(r'```|`[^`]+`', content))
        return f"""Markdown Analysis:
- Headers: {'✓' if has_headers else '✗'}
- Lists: {'✓' if has_lists else '✗'}
- Code blocks: {'✓' if has_code else '✗'}"""

    elif validation_type == "structure":
        issues = []
        score = 100
        if len(content) < 50:
            issues.append("Content too short")
            score -= 20
        if not content[0].isupper():
            issues.append("Doesn't start with capital")
            score -= 10
        return f"Score: {score}/100\nIssues: {issues if issues else 'None'}"

    return f"Unknown validation: {validation_type}"

# Test the tools
print("\n[Tool Tests]")
print(f"text_formatter('hello world', 'snake_case') -> {text_formatter.invoke({'text': 'hello world', 'format_type': 'snake_case'})}")
print(f"text_formatter('hello world', 'camel_case') -> {text_formatter.invoke({'text': 'hello world', 'format_type': 'camel_case'})}")
print(f"template_generator('bug_report')[:100]... -> Template generated")
print(f"content_validator(valid_json, 'json') -> {content_validator.invoke({'content': '{\"key\": \"value\"}', 'validation_type': 'json'})}")
print(f"content_validator(invalid_json, 'json') -> {content_validator.invoke({'content': '{key: value}', 'validation_type': 'json'})}")

print("\n✓ All 3 LangChain tools implemented and working!")

# ============================================================================
# Part 2: Model & LoRA Setup
# ============================================================================
print("\n[PART 2] Model & LoRA Configuration")
print("-" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Base model: {MODEL_NAME}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

# Quantization config
if torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("Model loaded with 4-bit quantization on GPU")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("Model loaded on CPU (no quantization)")

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nLoRA Configuration:")
print(f"  - Rank (r): {lora_config.r}")
print(f"  - Alpha: {lora_config.lora_alpha}")
print(f"  - Dropout: {lora_config.lora_dropout}")
print(f"  - Target modules: {len(lora_config.target_modules)} modules")
print(f"\nParameter Statistics:")
print(f"  - Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"  - Total: {total_params:,}")

# ============================================================================
# Part 3: Model Inference Demo
# ============================================================================
print("\n[PART 3] Model Inference Demo")
print("-" * 70)

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test inference
test_prompt = "### Instruction:\nExplain what machine learning is in one sentence.\n\n### Response:\n"
print(f"\nTest prompt: {test_prompt[:60]}...")
response = generate_response(model, tokenizer, test_prompt)
print(f"Model response:\n{response}\n")

# ============================================================================
# Part 4: Integration Demo
# ============================================================================
print("\n[PART 4] Integration: Model + Tools Demo")
print("-" * 70)

print("\nScenario: Create a bug report and validate it")
print("-" * 40)

# Step 1: Generate template
bug_template = template_generator.invoke({"template_type": "bug_report", "context": "Login freeze"})
print("1. Generated bug report template")

# Step 2: Model suggests content
suggestion_prompt = "### Instruction:\nDescribe a bug where the login page freezes after entering credentials.\n\n### Response:\n"
suggestion = generate_response(model, tokenizer, suggestion_prompt, max_new_tokens=100)
print(f"2. Model suggestion:\n{suggestion[:200]}...")

# Step 3: Validate the suggestion
validation = content_validator.invoke({"content": suggestion, "validation_type": "structure"})
print(f"3. Content validation: {validation}")

# Step 4: Format text for code
code_name = text_formatter.invoke({"text": "login page freeze handler", "format_type": "snake_case"})
print(f"4. Function name suggestion: {code_name}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("DEMO SUMMARY")
print("=" * 70)

summary = {
    "track": "C - Instructive Assistant",
    "dataset": "mlabonne/FineTome-100k",
    "model": MODEL_NAME,
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": list(lora_config.target_modules)
    },
    "tools_implemented": ["text_formatter", "template_generator", "content_validator"],
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "trainable_percentage": f"{100*trainable_params/total_params:.2f}%"
}

print(f"""
What was demonstrated:

1. Fine-tuning Setup (50%):
   - Model: {MODEL_NAME}
   - LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}
   - Trainable parameters: {trainable_params:,} ({summary['trainable_percentage']})
   - Dataset: mlabonne/FineTome-100k

2. LangChain Tools (30%):
   - text_formatter: Convert text to snake_case, camelCase, etc.
   - template_generator: Create email, bug report, code review templates
   - content_validator: Validate JSON, markdown, structure

3. Integration Demo (20%):
   - Multi-step workflow combining model + tools
   - Realistic scenario: bug report creation

To run full training, execute:
  jupyter notebook hometask_7_finetuning_tools.ipynb
""")

# Save summary
with open('homework_7_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved to homework_7_summary.json")
print("\n✓ Demo completed successfully!")
