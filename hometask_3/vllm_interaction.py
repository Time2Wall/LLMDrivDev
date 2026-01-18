"""
vLLM Interaction Script

This script demonstrates two ways to interact with a vLLM server:
1. Using HTTP requests (httpx/requests)
2. Using the OpenAI Python client

Model used: Qwen/Qwen2-0.5B-Instruct (smallest model with chat template support)
"""

import json
import httpx
import requests
from openai import OpenAI


# Configuration
VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


def interact_with_httpx(question: str) -> str:
    """
    Interact with vLLM using httpx library (async-capable HTTP client).

    Args:
        question: The question to ask the model

    Returns:
        The model's response text
    """
    url = f"{VLLM_BASE_URL}/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise and accurate answers."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.95
    }

    headers = {
        "Content-Type": "application/json"
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def interact_with_requests(question: str) -> str:
    """
    Interact with vLLM using requests library (standard HTTP client).

    Args:
        question: The question to ask the model

    Returns:
        The model's response text
    """
    url = f"{VLLM_BASE_URL}/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise and accurate answers."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.95
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def interact_with_openai_client(question: str) -> str:
    """
    Interact with vLLM using OpenAI Python client.

    vLLM provides an OpenAI-compatible API, so we can use the official
    OpenAI client by pointing it to our local server.

    Args:
        question: The question to ask the model

    Returns:
        The model's response text
    """
    # Create OpenAI client pointing to local vLLM server
    client = OpenAI(
        base_url=f"{VLLM_BASE_URL}/v1",
        api_key="not-needed"  # vLLM doesn't require API key by default
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise and accurate answers."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=0.95
    )

    return response.choices[0].message.content


def check_server_health() -> bool:
    """Check if the vLLM server is running and healthy."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models() -> list:
    """List all models available on the vLLM server."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
        return response.json()["data"]
    except requests.exceptions.RequestException as e:
        print(f"Error listing models: {e}")
        return []


def main():
    """Main function demonstrating all interaction methods."""

    print("=" * 60)
    print("vLLM Interaction Demo")
    print(f"Model: {MODEL_NAME}")
    print(f"Server: {VLLM_BASE_URL}")
    print("=" * 60)

    # Check server health
    print("\n[1] Checking server health...")
    if not check_server_health():
        print("ERROR: vLLM server is not running!")
        print("Please start the server first:")
        print("  docker-compose --profile gpu up -d vllm-gpu")
        print("  OR")
        print("  docker-compose --profile cpu up -d vllm-cpu")
        print("  OR")
        print("  vllm serve Qwen/Qwen2-0.5B-Instruct --host 0.0.0.0 --port 8000")
        return
    print("Server is healthy!")

    # List available models
    print("\n[2] Available models:")
    models = list_available_models()
    for model in models:
        print(f"  - {model['id']}")

    # Test question
    question = "What is the capital of Germany?"
    print(f"\n[3] Test Question: {question}")

    # Method 1: Using httpx
    print("\n--- Method 1: Using httpx ---")
    try:
        response_httpx = interact_with_httpx(question)
        print(f"Response: {response_httpx}")
    except Exception as e:
        print(f"Error: {e}")

    # Method 2: Using requests
    print("\n--- Method 2: Using requests ---")
    try:
        response_requests = interact_with_requests(question)
        print(f"Response: {response_requests}")
    except Exception as e:
        print(f"Error: {e}")

    # Method 3: Using OpenAI client
    print("\n--- Method 3: Using OpenAI client ---")
    try:
        response_openai = interact_with_openai_client(question)
        print(f"Response: {response_openai}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
