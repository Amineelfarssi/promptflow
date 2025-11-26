# Implementation Guide: Evaluation System

This guide provides concrete implementation steps for adding the evaluation system to PromptFlow.

## Phase 1: Basic Evaluation Framework (Week 1-2)

### Step 1: Create Evaluation Models

```python
# src/promptflow/evaluation/models.py
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from pydantic import BaseModel, Field

class MetricType(str, Enum):
    """Types of evaluation metrics."""
    NUMERIC = "numeric"      # 0.0 to 1.0 score
    BOOLEAN = "boolean"      # pass/fail
    CATEGORICAL = "categorical"  # A, B, C, etc.

class TestCase(BaseModel):
    """A single test case for prompt evaluation."""
    id: str
    input_vars: dict[str, Any]
    expected_output: str | None = None
    reference_outputs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

class TestSet(BaseModel):
    """Collection of test cases."""
    id: str
    name: str
    description: str | None = None
    project: str = "default"
    test_cases: list[TestCase]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    version: int = 1

class EvaluationResult(BaseModel):
    """Result for a single test case."""
    test_case_id: str
    input_vars: dict[str, Any]
    rendered_prompt: str
    model_output: str | None = None
    expected_output: str | None = None
    scores: dict[str, float | bool | str] = Field(default_factory=dict)
    latency_ms: float = 0.0
    token_count: int | None = None
    error: str | None = None

class RunStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EvaluationRun(BaseModel):
    """A complete evaluation run."""
    id: str
    prompt_id: str
    prompt_version: int
    test_set_id: str
    test_set_name: str

    # Configuration
    model_id: str | None = None  # For actual LLM testing
    metrics: list[str] = Field(default_factory=list)

    # Results
    results: list[EvaluationResult] = Field(default_factory=list)
    aggregate_scores: dict[str, float] = Field(default_factory=dict)

    # Metadata
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None

    # Stats
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

class EvaluationComparison(BaseModel):
    """Comparison between two evaluation runs."""
    run1_id: str
    run2_id: str
    run1_version: int
    run2_version: int

    metric_deltas: dict[str, float]  # metric -> delta
    winner: str | None = None  # "run1", "run2", or None for tie
    improvement_pct: float = 0.0

    better_on: list[str] = Field(default_factory=list)
    worse_on: list[str] = Field(default_factory=list)
    case_by_case: list[dict[str, Any]] = Field(default_factory=list)
```

### Step 2: Implement Metrics

```python
# src/promptflow/evaluation/metrics.py
from typing import Any, Callable
import re

class Metric:
    """Base class for evaluation metrics."""

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        scorer: Callable[[str, str | None, dict], float | bool | str],
        description: str = "",
    ):
        self.name = name
        self.type = metric_type
        self.scorer = scorer
        self.description = description

    def score(
        self,
        output: str,
        expected: str | None = None,
        context: dict[str, Any] = None,
    ) -> float | bool | str:
        """Score an output."""
        return self.scorer(output, expected, context or {})

# Built-in metrics
def exact_match_scorer(output: str, expected: str | None, context: dict) -> bool:
    """Exact string match."""
    if expected is None:
        return False
    return output.strip() == expected.strip()

def contains_scorer(output: str, expected: str | None, context: dict) -> bool:
    """Check if output contains expected substring."""
    if expected is None:
        return False
    return expected.lower() in output.lower()

def length_check_scorer(output: str, expected: str | None, context: dict) -> bool:
    """Check if output length is within bounds."""
    min_len = context.get("min_length", 0)
    max_len = context.get("max_length", float('inf'))
    length = len(output)
    return min_len <= length <= max_len

def regex_match_scorer(output: str, expected: str | None, context: dict) -> bool:
    """Check if output matches regex pattern."""
    pattern = context.get("pattern")
    if not pattern:
        return False
    return bool(re.search(pattern, output))

def json_valid_scorer(output: str, expected: str | None, context: dict) -> bool:
    """Check if output is valid JSON."""
    import json
    try:
        json.loads(output)
        return True
    except:
        return False

def levenshtein_similarity_scorer(output: str, expected: str | None, context: dict) -> float:
    """Compute normalized Levenshtein similarity."""
    if expected is None:
        return 0.0

    # Simple Levenshtein implementation
    if len(output) == 0:
        return 0.0 if len(expected) > 0 else 1.0
    if len(expected) == 0:
        return 0.0

    # Create matrix
    d = [[0] * (len(expected) + 1) for _ in range(len(output) + 1)]

    for i in range(len(output) + 1):
        d[i][0] = i
    for j in range(len(expected) + 1):
        d[0][j] = j

    for i in range(1, len(output) + 1):
        for j in range(1, len(expected) + 1):
            cost = 0 if output[i-1] == expected[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )

    distance = d[len(output)][len(expected)]
    max_len = max(len(output), len(expected))
    return 1.0 - (distance / max_len)

# Pre-configured metrics
EXACT_MATCH = Metric(
    name="exact_match",
    metric_type=MetricType.BOOLEAN,
    scorer=exact_match_scorer,
    description="Exact string match with expected output"
)

CONTAINS = Metric(
    name="contains",
    metric_type=MetricType.BOOLEAN,
    scorer=contains_scorer,
    description="Output contains expected substring"
)

LENGTH_CHECK = Metric(
    name="length_check",
    metric_type=MetricType.BOOLEAN,
    scorer=length_check_scorer,
    description="Output length within bounds"
)

REGEX_MATCH = Metric(
    name="regex_match",
    metric_type=MetricType.BOOLEAN,
    scorer=regex_match_scorer,
    description="Output matches regex pattern"
)

JSON_VALID = Metric(
    name="json_valid",
    metric_type=MetricType.BOOLEAN,
    scorer=json_valid_scorer,
    description="Output is valid JSON"
)

LEVENSHTEIN_SIMILARITY = Metric(
    name="levenshtein_similarity",
    metric_type=MetricType.NUMERIC,
    scorer=levenshtein_similarity_scorer,
    description="Normalized Levenshtein similarity"
)

# Registry of built-in metrics
BUILTIN_METRICS = {
    "exact_match": EXACT_MATCH,
    "contains": CONTAINS,
    "length_check": LENGTH_CHECK,
    "regex_match": REGEX_MATCH,
    "json_valid": JSON_VALID,
    "levenshtein_similarity": LEVENSHTEIN_SIMILARITY,
}
```

### Step 3: Evaluation Runner

```python
# src/promptflow/evaluation/runner.py
import time
import uuid
from typing import Any
from promptflow.core.registry import PromptRegistry
from promptflow.evaluation.models import (
    EvaluationRun,
    EvaluationResult,
    RunStatus,
    TestSet,
)
from promptflow.evaluation.metrics import BUILTIN_METRICS, Metric

class EvaluationRunner:
    """Executes evaluation runs."""

    def __init__(self, registry: PromptRegistry):
        self.registry = registry

    def evaluate(
        self,
        prompt_name: str,
        test_set: TestSet | str,
        version: int | None = None,
        alias: str | None = None,
        project: str | None = None,
        metrics: list[str | Metric] | None = None,
        model_id: str | None = None,
    ) -> EvaluationRun:
        """
        Run evaluation on a prompt.

        Args:
            prompt_name: Name of prompt to evaluate
            test_set: TestSet object or test set ID
            version: Specific version to evaluate
            alias: Alias to evaluate (e.g., "prod")
            project: Project name
            metrics: List of metric names or Metric objects
            model_id: Model to use for actual LLM testing (optional)

        Returns:
            EvaluationRun with results
        """
        # Get prompt
        prompt = self.registry.get_prompt(prompt_name, project)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        # Get version
        if alias:
            prompt_version = prompt.get_by_alias(alias)
            version_num = prompt.aliases.get(alias)
        elif version:
            prompt_version = prompt.get_version(version)
            version_num = version
        else:
            prompt_version = prompt.get_version()
            version_num = prompt.latest_version

        if not prompt_version:
            raise ValueError(f"Version not found")

        # Get test set
        if isinstance(test_set, str):
            test_set = self._load_test_set(test_set)

        # Parse metrics
        metric_objects = self._parse_metrics(metrics or ["exact_match"])

        # Create run
        run = EvaluationRun(
            id=uuid.uuid4().hex[:12],
            prompt_id=prompt.id,
            prompt_version=version_num,
            test_set_id=test_set.id,
            test_set_name=test_set.name,
            model_id=model_id,
            metrics=[m.name for m in metric_objects],
            total_cases=len(test_set.test_cases),
            status=RunStatus.RUNNING,
        )

        # Run evaluation
        try:
            for test_case in test_set.test_cases:
                result = self._evaluate_case(
                    prompt_version,
                    test_case,
                    metric_objects,
                    model_id,
                )
                run.results.append(result)

                # Update stats
                if result.error is None:
                    run.passed_cases += 1
                else:
                    run.failed_cases += 1

            # Compute aggregate scores
            run.aggregate_scores = self._aggregate_scores(run.results)
            run.avg_latency_ms = sum(r.latency_ms for r in run.results) / len(run.results)
            run.total_tokens = sum(r.token_count or 0 for r in run.results)

            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            run.completed_at = datetime.now(timezone.utc)

        # Save run (extend storage backend to support this)
        self._save_run(run)

        return run

    def _evaluate_case(
        self,
        prompt_version,
        test_case,
        metrics: list[Metric],
        model_id: str | None,
    ) -> EvaluationResult:
        """Evaluate a single test case."""
        result = EvaluationResult(
            test_case_id=test_case.id,
            input_vars=test_case.input_vars,
            rendered_prompt="",
            expected_output=test_case.expected_output,
        )

        try:
            # Render prompt
            start = time.time()
            rendered = prompt_version.render(test_case.input_vars)
            result.rendered_prompt = rendered

            # If model_id provided, actually call the LLM
            if model_id:
                output, tokens = self._call_model(model_id, rendered)
                result.model_output = output
                result.token_count = tokens
            else:
                # Without model, just check rendering worked
                result.model_output = rendered

            result.latency_ms = (time.time() - start) * 1000

            # Score with metrics
            for metric in metrics:
                score = metric.score(
                    result.model_output,
                    test_case.expected_output,
                    test_case.metadata,
                )
                result.scores[metric.name] = score

        except Exception as e:
            result.error = str(e)

        return result

    def _parse_metrics(self, metrics: list[str | Metric]) -> list[Metric]:
        """Parse metric specifications."""
        parsed = []
        for m in metrics:
            if isinstance(m, Metric):
                parsed.append(m)
            elif isinstance(m, str):
                if m in BUILTIN_METRICS:
                    parsed.append(BUILTIN_METRICS[m])
                else:
                    raise ValueError(f"Unknown metric: {m}")
        return parsed

    def _aggregate_scores(self, results: list[EvaluationResult]) -> dict[str, float]:
        """Aggregate scores across all results."""
        if not results:
            return {}

        aggregated = {}
        metric_names = results[0].scores.keys()

        for metric in metric_names:
            scores = [
                r.scores[metric]
                for r in results
                if metric in r.scores and r.error is None
            ]

            if scores:
                # Convert booleans to 0/1 for averaging
                numeric_scores = [
                    float(s) if isinstance(s, (bool, int, float)) else 0.0
                    for s in scores
                ]
                aggregated[metric] = sum(numeric_scores) / len(numeric_scores)

        return aggregated

    def _call_model(self, model_id: str, prompt: str) -> tuple[str, int]:
        """Call an LLM model (to be implemented)."""
        # TODO: Implement actual model calling
        # For Bedrock:
        # - Parse model_id
        # - Call bedrock.converse()
        # - Return output and token count
        raise NotImplementedError("Model calling not yet implemented")

    def _load_test_set(self, test_set_id: str) -> TestSet:
        """Load test set from storage."""
        # TODO: Extend storage backend to support test sets
        raise NotImplementedError("Test set loading not yet implemented")

    def _save_run(self, run: EvaluationRun):
        """Save evaluation run."""
        # TODO: Extend storage backend to support evaluation runs
        pass
```

### Step 4: Add to Registry

```python
# Update src/promptflow/core/registry.py
from promptflow.evaluation.runner import EvaluationRunner
from promptflow.evaluation.models import TestSet, EvaluationRun

class PromptRegistry:
    # ... existing methods ...

    def create_test_set(
        self,
        name: str,
        test_cases: list[dict[str, Any]],
        project: str | None = None,
        description: str | None = None,
    ) -> TestSet:
        """Create a test set."""
        from promptflow.evaluation.models import TestCase

        project = project or self.default_project

        parsed_cases = [
            TestCase(
                id=str(i),
                input_vars=tc["input_vars"],
                expected_output=tc.get("expected_output"),
                metadata=tc.get("metadata", {}),
                tags=tc.get("tags", []),
            )
            for i, tc in enumerate(test_cases)
        ]

        test_set = TestSet(
            id=_generate_id(),
            name=name,
            description=description,
            project=project,
            test_cases=parsed_cases,
        )

        # Save to storage
        self.storage.save_test_set(test_set)
        return test_set

    def evaluate(
        self,
        prompt_name: str,
        test_set: str | TestSet,
        version: int | None = None,
        alias: str | None = None,
        metrics: list[str] | None = None,
        model_id: str | None = None,
    ) -> EvaluationRun:
        """Evaluate a prompt."""
        runner = EvaluationRunner(self)
        return runner.evaluate(
            prompt_name=prompt_name,
            test_set=test_set,
            version=version,
            alias=alias,
            project=self.default_project,
            metrics=metrics,
            model_id=model_id,
        )

    def get_evaluation_run(self, run_id: str) -> EvaluationRun | None:
        """Get an evaluation run by ID."""
        return self.storage.get_evaluation_run(run_id)

    def list_evaluation_runs(
        self,
        prompt_name: str | None = None,
        project: str | None = None,
        limit: int = 10,
    ) -> list[EvaluationRun]:
        """List evaluation runs."""
        return self.storage.list_evaluation_runs(
            prompt_name=prompt_name,
            project=project or self.default_project,
            limit=limit,
        )
```

### Step 5: CLI Commands

```python
# Add to src/promptflow/cli.py
@app.command()
def create_test_set(
    name: str = typer.Argument(..., help="Test set name"),
    file: str = typer.Option(..., "--file", "-f", help="JSON file with test cases"),
    project: str = typer.Option(None, "--project", "-p"),
):
    """Create a test set from a JSON file."""
    import json

    with open(file) as f:
        test_cases = json.load(f)

    registry = get_registry()
    test_set = registry.create_test_set(
        name=name,
        test_cases=test_cases,
        project=project,
    )

    console.print(f"✓ Created test set: {test_set.name} ({len(test_set.test_cases)} cases)")

@app.command()
def evaluate(
    prompt_name: str = typer.Argument(..., help="Prompt to evaluate"),
    test_set: str = typer.Option(..., "--test-set", "-t"),
    version: int = typer.Option(None, "--version", "-v"),
    alias: str = typer.Option(None, "--alias", "-a"),
    metrics: str = typer.Option("exact_match", "--metrics", "-m", help="Comma-separated metrics"),
    model: str = typer.Option(None, "--model"),
):
    """Evaluate a prompt version."""
    registry = get_registry()

    metric_list = [m.strip() for m in metrics.split(",")]

    console.print(f"Running evaluation on {prompt_name}...")

    run = registry.evaluate(
        prompt_name=prompt_name,
        test_set=test_set,
        version=version,
        alias=alias,
        metrics=metric_list,
        model_id=model,
    )

    # Display results
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Status: {run.status}")
    console.print(f"  Passed: {run.passed_cases}/{run.total_cases}")
    console.print(f"  Avg Latency: {run.avg_latency_ms:.2f}ms")

    console.print("\n[bold]Scores:[/bold]")
    for metric, score in run.aggregate_scores.items():
        console.print(f"  {metric}: {score:.3f}")
```

## Usage Examples

### Example 1: Create and Run Evaluation

```bash
# 1. Create test set
cat > test_cases.json << EOF
[
  {
    "input_vars": {"text": "The cat sat on the mat."},
    "expected_output": "A cat sitting on a mat."
  },
  {
    "input_vars": {"text": "Python is a programming language."},
    "expected_output": "Python: a programming language."
  }
]
EOF

promptflow create-test-set summarization_tests --file test_cases.json

# 2. Run evaluation
promptflow evaluate summarizer \
  --test-set summarization_tests \
  --alias prod \
  --metrics "levenshtein_similarity,length_check"

# 3. View results
promptflow show-evaluation <run-id>
```

### Example 2: Python API

```python
from promptflow import PromptRegistry
from promptflow.evaluation.models import TestCase

# Initialize
registry = PromptRegistry()

# Create test set
test_set = registry.create_test_set(
    name="greeting_tests",
    test_cases=[
        {
            "input_vars": {"name": "Alice", "style": "formal"},
            "expected_output": "Good day, Alice.",
        },
        {
            "input_vars": {"name": "Bob", "style": "casual"},
            "expected_output": "Hey Bob!",
        },
    ]
)

# Run evaluation
results = registry.evaluate(
    prompt_name="greeting",
    test_set=test_set,
    version=2,
    metrics=["exact_match", "levenshtein_similarity"],
)

# Check results
print(f"Score: {results.aggregate_scores['exact_match']:.2%}")
print(f"Similarity: {results.aggregate_scores['levenshtein_similarity']:.2%}")

# View individual results
for result in results.results:
    print(f"Input: {result.input_vars}")
    print(f"Output: {result.model_output}")
    print(f"Expected: {result.expected_output}")
    print(f"Scores: {result.scores}")
    print()
```

## Next Steps

1. **Extend Storage Backend**: Add methods for test sets and evaluation runs
2. **Add Advanced Metrics**: Cosine similarity, BLEU, ROUGE, LLM-as-judge
3. **Dashboard Integration**: Add evaluation results page to Streamlit app
4. **Model Integration**: Implement actual LLM calling for Bedrock/OpenAI
5. **Comparison Tools**: Build version comparison and A/B testing

## Testing

```python
# tests/test_evaluation.py
import pytest
from promptflow import PromptRegistry
from promptflow.evaluation.models import TestCase, TestSet

def test_basic_evaluation():
    registry = PromptRegistry()

    # Register prompt
    registry.register(
        name="test_prompt",
        template="Hello {name}!"
    )

    # Create test set
    test_set = TestSet(
        id="test1",
        name="Test Set 1",
        test_cases=[
            TestCase(
                id="1",
                input_vars={"name": "World"},
                expected_output="Hello World!"
            )
        ]
    )

    # Evaluate
    result = registry.evaluate(
        "test_prompt",
        test_set=test_set,
        metrics=["exact_match"]
    )

    assert result.status == "completed"
    assert result.passed_cases == 1
    assert result.aggregate_scores["exact_match"] == 1.0
```

This implementation provides a solid foundation for the evaluation system that can be extended with more advanced features over time.
