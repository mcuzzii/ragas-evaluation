import google.genai as genai
from ragas.llms import llm_factory
from dotenv import load_dotenv
import pandas as pd
import os
from ragas.metrics import discrete_metric
from ragas.metrics.result import MetricResult
from ragas import Dataset, experiment
import asyncio

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Create dataset
dataset = Dataset(
    name="sentiment_dataset",
    backend="local/csv",
    root_dir="evals"
)
samples = [
    {"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    {"text": "It was an average film, nothing special.", "label": "positive"},
    {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"}
]
for sample in samples:
    dataset.append(sample)
dataset.save()

@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def sentiment_metric(prediction: str, actual: str):
    """Calculate accuracy of the prediction."""
    return MetricResult(value="pass", reason="") if prediction == actual else MetricResult(value="fail", reason="")

def generate_sentiment(text, cli, model_name):
    prompt = (
        f"Analyze the sentiment of the following text. "
        f"You must respond with exactly one word: either 'positive' or 'negative' in all lowercase. "
        f"Do not include any punctuation, explanation, or extra text.\n\n"
        f"Text:\n{text}"
    )
    response = cli.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response.text.strip().lower()

@experiment()
async def run_experiment(row, cli, model_name):
    response = generate_sentiment(row["text"], cli=cli, model_name=model_name)
    score = sentiment_metric.score(
        prediction=response,
        actual=row["label"]
    )
    experiment_view = {
        **row,
        "response": response,
        "score": score.value,
    }
    return experiment_view

# Run with specific parameters
results = asyncio.run(run_experiment.arun(dataset, cli=client, model_name="gemini-2.5-flash"))
pd.DataFrame(results).to_csv("evals/experiments/sentiment_results.csv")