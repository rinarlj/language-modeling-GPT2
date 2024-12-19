import os
import wandb
from data_processing import prepare_datasets
from model_setup import setup_model_and_tokenizer
from training import load_datasets, train_model
from text_generation import generate_text
from evaluation import evaluate_model

wandb.login()

wandb.init(project="gpt2-malagasy", name="fine-tuning-gpt2")

output_dir = "./gpt2-malagasy"
train_path, test_path = "train_dataset.txt", "test_dataset.txt"

train_path, test_path = prepare_datasets("/content/extracted/rakitra/rakitra.csv", train_path, test_path)

model, tokenizer = setup_model_and_tokenizer()

train_dataset, test_dataset = load_datasets(train_path, test_path, tokenizer)

training_results = train_model(model, tokenizer, train_dataset, test_dataset, output_dir)
metrics = evaluate_model("./gpt2-malagasy", test_dataset)
wandb.log(metrics)
print(training_results["metrics"])


