"""
Pytest tests for Ragas evaluation metrics.
Tests wrap Ragas evaluations and enforce minimum quality thresholds.
"""

import json
import os
import pytest
import pandas as pd

# Minimum threshold constants
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.7

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "ragas_results.json")
METRICS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation_metrics.csv")


@pytest.fixture(scope="session")
def ragas_results():
    """Load Ragas evaluation results from JSON file."""
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("Ragas results file not found. Run the notebook first.")
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def per_sample_metrics():
    """Load per-sample metrics from CSV file."""
    if not os.path.exists(METRICS_CSV_PATH):
        pytest.skip("Metrics CSV file not found. Run the notebook first.")
    return pd.read_csv(METRICS_CSV_PATH)


class TestRagasAggregateMetrics:
    """Test aggregate Ragas metrics against minimum thresholds."""

    def test_faithfulness_above_threshold(self, ragas_results):
        """Faithfulness score must be >= 0.7 to ensure no hallucinations."""
        score = ragas_results["aggregate_metrics"]["faithfulness"]
        assert score >= FAITHFULNESS_THRESHOLD, (
            f"Faithfulness score {score:.3f} is below threshold {FAITHFULNESS_THRESHOLD}"
        )

    def test_answer_relevancy_above_threshold(self, ragas_results):
        """Answer relevancy must be >= 0.7 to ensure answers are on-topic."""
        score = ragas_results["aggregate_metrics"]["answer_relevancy"]
        assert score >= ANSWER_RELEVANCY_THRESHOLD, (
            f"Answer relevancy score {score:.3f} is below threshold {ANSWER_RELEVANCY_THRESHOLD}"
        )

    def test_context_recall_above_threshold(self, ragas_results):
        """Context recall must be >= 0.7 to ensure relevant retrieval."""
        score = ragas_results["aggregate_metrics"]["context_recall"]
        assert score >= CONTEXT_RECALL_THRESHOLD, (
            f"Context recall score {score:.3f} is below threshold {CONTEXT_RECALL_THRESHOLD}"
        )


class TestRagasPerSampleMetrics:
    """Test per-sample Ragas metrics for quality consistency."""

    def test_no_zero_faithfulness(self, per_sample_metrics):
        """No sample should have zero faithfulness (complete hallucination)."""
        if "faithfulness" not in per_sample_metrics.columns:
            pytest.skip("Faithfulness column not found in metrics CSV")
        zero_faith = per_sample_metrics[per_sample_metrics["faithfulness"] == 0.0]
        assert len(zero_faith) == 0, (
            f"{len(zero_faith)} samples have zero faithfulness (complete hallucination): "
            f"{zero_faith['question'].tolist()}"
        )

    def test_minimum_samples_evaluated(self, per_sample_metrics):
        """At least 10 samples must have been evaluated."""
        assert len(per_sample_metrics) >= 10, (
            f"Only {len(per_sample_metrics)} samples evaluated, minimum is 10"
        )

    def test_all_metrics_present(self, per_sample_metrics):
        """All required metric columns must be present."""
        required_columns = ["faithfulness", "answer_relevancy", "context_recall"]
        for col in required_columns:
            assert col in per_sample_metrics.columns, (
                f"Required metric column '{col}' not found in results"
            )

    def test_metrics_in_valid_range(self, per_sample_metrics):
        """All metric values must be between 0 and 1."""
        metric_columns = ["faithfulness", "answer_relevancy", "context_recall"]
        for col in metric_columns:
            if col in per_sample_metrics.columns:
                values = per_sample_metrics[col].dropna()
                assert (values >= 0).all() and (values <= 1).all(), (
                    f"Metric '{col}' has values outside [0, 1] range"
                )


class TestRagasResultsIntegrity:
    """Test the integrity and completeness of Ragas results."""

    def test_results_have_required_keys(self, ragas_results):
        """Results JSON must contain all required sections."""
        required_keys = ["aggregate_metrics", "config", "thresholds"]
        for key in required_keys:
            assert key in ragas_results, f"Required key '{key}' missing from results"

    def test_thresholds_documented(self, ragas_results):
        """Quality thresholds must be documented in results."""
        thresholds = ragas_results.get("thresholds", {})
        assert "faithfulness" in thresholds, "Faithfulness threshold not documented"
        assert "answer_relevancy" in thresholds, "Answer relevancy threshold not documented"
        assert "context_recall" in thresholds, "Context recall threshold not documented"

    def test_quality_gate_pass(self, ragas_results):
        """Combined quality gate: all metrics must pass their thresholds."""
        metrics = ragas_results["aggregate_metrics"]
        thresholds = ragas_results["thresholds"]
        failures = []
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                failures.append(
                    f"{metric}: {metrics[metric]:.3f} < {threshold}"
                )
        assert len(failures) == 0, (
            f"Quality gate FAILED. Metrics below threshold:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
