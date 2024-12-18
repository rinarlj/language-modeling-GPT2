import os
from data_processing import prepare_datasets
from model_setup import setup_model_and_tokenizer
from training import load_datasets, train_model
from text_generation import generate_text



output_dir = "./gpt2-malagasy"
train_path, test_path = "./data/train_dataset.txt", "./data/test_dataset.txt"

train_path, test_path = prepare_datasets("/content/extracted/rakitra/rakitra.csv", train_path, test_path)

model, tokenizer = setup_model_and_tokenizer()

train_dataset, test_dataset = load_datasets(train_path, test_path, tokenizer)

training_results = train_model(model, tokenizer, train_dataset, test_dataset, output_dir)

print(training_results["metrics"])


