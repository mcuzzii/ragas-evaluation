from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rag import SimpleKeywordRetriever, ExampleRAG

load_dotenv()

DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]

samples = [
    {"query": "What is Ragas 0.3?", "grading_notes": "- Ragas 0.3 is a library for evaluating LLM applications."},
    {"query": "How to install Ragas?", "grading_notes": "- install from source  - install from pip using ragas[examples]"},
    {"query": "What are the main features of Ragas?", "grading_notes": "organised around - experiments - datasets - metrics."}
]
dataset = Dataset(
    name="rag_system_dataset",
    backend="local/csv",
    root_dir="evals"
)
for sample in samples:
    dataset.append(sample)
dataset.save()

my_metric = DiscreteMetric(
    name="correctness",
    prompt = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

openai_client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL")
)

# Create RAG client
test_rag = ExampleRAG(
    llm_client=openai_client,
    model_name="openai/gpt-oss-20b:free",
    retriever=SimpleKeywordRetriever(),
    logdir="evals/logs"
)

# Add documents (this will be traced)
test_rag.add_documents(DOCUMENTS)

# Create LLM judge
llm_judge = llm_factory("openai/gpt-oss-120b:free", client=openai_client)

@experiment()
async def run_experiment(row, rag_client, judge):
    response = rag_client.query(row["query"])

    score = my_metric.score(
        llm=judge,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"]
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view

result = asyncio.run(run_experiment.arun(dataset, rag_client=test_rag, judge=llm_judge))
pd.DataFrame(result).to_csv("evals/experiments/rag_system_results.csv")