📘 llm-lab

A lightweight, notebook-first framework for running reproducible LLM experiments, comparing multiple models, and evaluating custom metrics across JSONL/CSV datasets.

🚀 Features

Multi-model evaluation — compare OpenAI models in one experiment

CSV + JSONL support — ideal for business analysts & product teams

Custom metrics — create your own quality checks with a one-line decorator

Reproducible runs — every run stored in a SQLite database

Built-in comparison tools — leaderboard, per-run summaries, charts

Notebook-first design — built to be used directly in Jupyter

📦 Installation

For now (local development):

pip install -e .


PyPI publishing will come later.

📄 Dataset Formats

llm-lab supports:

CSV files

JSONL (one example per line)

JSON lists

Each example must contain:

expected_output — the reference answer (used by metrics)

Other fields can be anything you want to use inside your prompt.

Example .csv
question,context,expected_output
"What is 2+2?", "", "4"
"Capital of France?", "", "Paris"

🧪 Quickstart
from llm_lab import Experiment, run_experiment, compare_models

exp = Experiment(
    name="qa_eval",
    dataset_path="data/qa_sample.csv",
    prompt_template="Q: {question}\nA:",
    model_names=["gpt-4o-mini", "gpt-3.5-turbo"],
    metrics=["exact_match"],
    model_params={"temperature": 0.0},
)

results = run_experiment(exp, max_examples=20)

compare_models(results)

📊 Comparing Models

Leaderboard:

from llm_lab import show_leaderboard
show_leaderboard(metric="exact_match")


Detailed run summary:

from llm_lab import summarize_run
summarize_run(results[0].run_id)


Plot metric across models:

from llm_lab import plot_model_metric
plot_model_metric(results, metric="exact_match")

🧩 Custom Metrics

You can add your own metrics easily.

from llm_lab import register_metric

@register_metric
def contains_expected(example, output):
    expected = example["expected_output"].lower()
    return {"contains_expected": float(expected in output.lower())}


Then include it in your experiment:

metrics=["exact_match", "contains_expected"]

🧰 CSV Normalization Utility

Business analysts often have arbitrary column names (Ideal_Answer, target, etc).
Use the helper to normalize your raw CSV into an llm-lab-compatible dataset.

from llm_lab import prepare_csv_for_llm_lab

prepare_csv_for_llm_lab(
    src_path="raw/support_eval_raw.csv",
    dst_path="data/support_eval.csv",
    expected_col="Ideal_Answer",
)


Now support_eval.csv can be used directly.

🗂 Project Structure
llm-lab/
│
├── data/
│   └── qa_sample.csv
├── notebooks/
│   └── 01_quickstart.ipynb
├── src/
│   └── llm_lab/
│       ├── experiment.py
│       ├── metrics.py
│       ├── model_client.py
│       ├── storage.py
│       ├── utils.py
│       ├── analysis.py
│       └── __init__.py
├── pyproject.toml
└── README.md

🧠 Philosophy

llm-lab is designed with three principles:

Minimalism — no boilerplate, no YAML configs, no heavy framework

Reproducibility — all experiments are logged in SQLite

Accessibility — analysts and PMs should be able to use it with zero ML background

It aims to be the pytest + scikit-learn of LLM evaluation—
simple, composable, and transparent.

🤝 Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss what you'd like to change.

📄 License

MIT License.