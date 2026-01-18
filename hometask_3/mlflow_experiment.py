"""
MLflow Experiment with Custom GenAI Metric (LLM-as-a-Judge)

This script demonstrates how to:
1. Create a custom metric using make_genai_metric
2. Use a local vLLM model as the judge
3. Run experiments and evaluate responses

Model used: Qwen/Qwen2-0.5B-Instruct via local vLLM server
"""

import os
import pandas as pd
import mlflow
from mlflow.metrics.genai import make_genai_metric, EvaluationExample
from openai import OpenAI


# Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Or use local: "mlruns"


def setup_mlflow():
    """Configure MLflow tracking."""
    # Use local tracking if MLflow server is not running
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("vllm-llm-as-judge-experiment")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    except Exception:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("vllm-llm-as-judge-experiment")
        print("Using local MLflow tracking: mlruns/")


def create_answer_relevance_metric():
    """
    Create a custom GenAI metric for evaluating answer relevance.

    This metric uses the local vLLM model as a judge to evaluate
    how relevant and accurate the generated answers are to the questions.

    Returns:
        A custom genai metric for answer relevance evaluation
    """

    # Define grading examples for few-shot learning
    relevance_examples = [
        EvaluationExample(
            input="What is the capital of France?",
            output="The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.",
            score=5,
            justification="The answer directly addresses the question, provides the correct answer (Paris), and adds relevant context about the city's significance."
        ),
        EvaluationExample(
            input="What is the capital of France?",
            output="France is a country in Europe.",
            score=2,
            justification="The answer mentions France but fails to answer the actual question about its capital. The information provided is tangentially related but not what was asked."
        ),
        EvaluationExample(
            input="What is the capital of France?",
            output="I like pizza.",
            score=1,
            justification="The answer is completely irrelevant to the question and provides no useful information about the capital of France."
        ),
        EvaluationExample(
            input="Explain how photosynthesis works.",
            output="Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This occurs primarily in the chloroplasts of plant cells.",
            score=5,
            justification="The answer provides a clear, accurate explanation of photosynthesis including the inputs, outputs, and location where it occurs."
        ),
    ]

    # Create the custom metric
    answer_relevance = make_genai_metric(
        name="answer_relevance",
        definition=(
            "Answer relevance measures how well the generated response addresses "
            "the input question or prompt. A relevant answer should directly respond "
            "to what was asked, provide accurate information, and be appropriately detailed."
        ),
        grading_prompt=(
            "You are an expert evaluator assessing the relevance and quality of AI-generated answers.\n\n"
            "Evaluate the following response based on these criteria:\n"
            "1. Does the answer directly address the question asked?\n"
            "2. Is the information provided accurate?\n"
            "3. Is the answer appropriately detailed (not too brief, not excessive)?\n"
            "4. Is the response coherent and well-structured?\n\n"
            "Score from 1 to 5:\n"
            "- 5: Excellent - Directly addresses the question with accurate, well-structured information\n"
            "- 4: Good - Addresses the question well with minor issues\n"
            "- 3: Acceptable - Partially addresses the question but missing key information\n"
            "- 2: Poor - Tangentially related but doesn't properly answer the question\n"
            "- 1: Unacceptable - Completely irrelevant or incorrect\n\n"
            "Question: {input}\n"
            "Answer: {output}\n\n"
            "Provide your evaluation with a score and brief justification."
        ),
        examples=relevance_examples,
        model=f"endpoints:/{MODEL_NAME}",
        parameters={"temperature": 0.0, "max_tokens": 256},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    return answer_relevance


def create_factual_accuracy_metric():
    """
    Create a custom GenAI metric for factual accuracy evaluation.

    This metric evaluates whether the generated content is factually correct.

    Returns:
        A custom genai metric for factual accuracy
    """

    accuracy_examples = [
        EvaluationExample(
            input="What year did World War II end?",
            output="World War II ended in 1945.",
            score=5,
            justification="The answer is factually correct. WWII ended in 1945 with Germany's surrender in May and Japan's surrender in September."
        ),
        EvaluationExample(
            input="What year did World War II end?",
            output="World War II ended in 1943.",
            score=1,
            justification="The answer is factually incorrect. WWII ended in 1945, not 1943."
        ),
    ]

    factual_accuracy = make_genai_metric(
        name="factual_accuracy",
        definition=(
            "Factual accuracy measures whether the information provided in the response "
            "is correct and verifiable. This metric penalizes incorrect facts, "
            "made-up information, and misleading statements."
        ),
        grading_prompt=(
            "You are a fact-checker evaluating the factual accuracy of AI-generated responses.\n\n"
            "Evaluate whether the response contains accurate, verifiable information.\n\n"
            "Score from 1 to 5:\n"
            "- 5: All facts are correct and verifiable\n"
            "- 4: Mostly accurate with minor, inconsequential errors\n"
            "- 3: Some accurate information but also notable errors\n"
            "- 2: Multiple significant factual errors\n"
            "- 1: Mostly or entirely incorrect information\n\n"
            "Question: {input}\n"
            "Answer: {output}\n\n"
            "Evaluate the factual accuracy and provide your score with justification."
        ),
        examples=accuracy_examples,
        model=f"endpoints:/{MODEL_NAME}",
        parameters={"temperature": 0.0, "max_tokens": 256},
        greater_is_better=True,
        aggregations=["mean", "variance"],
    )

    return factual_accuracy


def generate_response(question: str) -> str:
    """Generate a response using the local vLLM model."""
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide accurate and concise answers."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=256
    )

    return response.choices[0].message.content


def create_evaluation_dataset() -> pd.DataFrame:
    """
    Create a sample dataset for evaluation.

    Returns:
        DataFrame with questions and model-generated answers
    """
    questions = [
        "What is the capital of Germany?",
        "Who wrote the play Romeo and Juliet?",
        "What is the chemical formula for water?",
        "In what year did humans first land on the Moon?",
        "What is the largest planet in our solar system?",
        "What programming language is known for its use in data science?",
        "What is the speed of light in vacuum?",
        "Who painted the Mona Lisa?",
    ]

    print("Generating responses for evaluation dataset...")
    data = []
    for q in questions:
        print(f"  - {q}")
        try:
            response = generate_response(q)
            data.append({"input": q, "output": response})
        except Exception as e:
            print(f"    Error: {e}")
            data.append({"input": q, "output": f"Error generating response: {e}"})

    return pd.DataFrame(data)


def run_evaluation_with_custom_model():
    """
    Run evaluation using custom metrics with the local vLLM model.

    This function demonstrates how to manually evaluate responses
    when mlflow.evaluate() isn't directly compatible with local endpoints.
    """
    print("\n" + "=" * 60)
    print("Running Manual Evaluation with Local vLLM Judge")
    print("=" * 60)

    # Create evaluation dataset
    eval_df = create_evaluation_dataset()

    print("\n--- Evaluation Dataset ---")
    for idx, row in eval_df.iterrows():
        print(f"\nQ{idx + 1}: {row['input']}")
        print(f"A{idx + 1}: {row['output'][:200]}...")

    # Manual LLM-as-Judge evaluation
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"
    )

    judge_prompt_template = """You are an expert evaluator assessing AI-generated answers.

Evaluate the following response on a scale of 1-5 based on:
1. Relevance: Does it address the question?
2. Accuracy: Is the information correct?
3. Clarity: Is it well-written?

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object in this format:
{{"score": <1-5>, "justification": "<brief explanation>"}}"""

    print("\n--- LLM-as-Judge Evaluation Results ---")
    results = []

    with mlflow.start_run(run_name="vllm-judge-evaluation"):
        # Log parameters
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("judge_model", MODEL_NAME)
        mlflow.log_param("num_samples", len(eval_df))

        for idx, row in eval_df.iterrows():
            prompt = judge_prompt_template.format(
                question=row["input"],
                answer=row["output"]
            )

            try:
                judge_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150
                )

                eval_text = judge_response.choices[0].message.content
                print(f"\nQ{idx + 1}: {row['input']}")
                print(f"Judge evaluation: {eval_text}")

                # Try to parse score
                try:
                    import json
                    # Find JSON in response
                    start = eval_text.find("{")
                    end = eval_text.rfind("}") + 1
                    if start != -1 and end != 0:
                        eval_json = json.loads(eval_text[start:end])
                        score = eval_json.get("score", 0)
                    else:
                        score = 0
                except Exception:
                    score = 0

                results.append({
                    "question": row["input"],
                    "answer": row["output"],
                    "evaluation": eval_text,
                    "score": score
                })

            except Exception as e:
                print(f"Error evaluating Q{idx + 1}: {e}")
                results.append({
                    "question": row["input"],
                    "answer": row["output"],
                    "evaluation": f"Error: {e}",
                    "score": 0
                })

        # Calculate and log aggregate metrics
        scores = [r["score"] for r in results if r["score"] > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            mlflow.log_metric("average_relevance_score", avg_score)
            mlflow.log_metric("num_evaluated", len(scores))
            print(f"\n--- Aggregate Results ---")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Samples Evaluated: {len(scores)}/{len(results)}")

        # Log the results dataframe
        results_df = pd.DataFrame(results)
        results_df.to_csv("evaluation_results.csv", index=False)
        mlflow.log_artifact("evaluation_results.csv")

        print(f"\nResults saved to evaluation_results.csv")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return results


def main():
    """Main function to run the MLflow experiment."""
    print("=" * 60)
    print("MLflow LLM-as-a-Judge Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"vLLM Server: {VLLM_BASE_URL}")
    print("=" * 60)

    # Check if vLLM server is running
    import requests
    try:
        health_url = VLLM_BASE_URL.replace("/v1", "/health")
        response = requests.get(health_url, timeout=5)
        if response.status_code != 200:
            raise Exception("Server not healthy")
        print("\nvLLM server is running!")
    except Exception as e:
        print(f"\nERROR: vLLM server is not available at {VLLM_BASE_URL}")
        print("Please start the vLLM server first:")
        print("  docker-compose --profile gpu up -d vllm-gpu")
        print("  OR")
        print("  vllm serve Qwen/Qwen2-0.5B-Instruct --host 0.0.0.0 --port 8000")
        return

    # Setup MLflow
    setup_mlflow()

    # Run evaluation
    results = run_evaluation_with_custom_model()

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("View results in MLflow UI: mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
