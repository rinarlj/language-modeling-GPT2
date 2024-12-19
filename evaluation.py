from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

def evaluate_model(output_dir, test_dataset):
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=16,
            logging_dir="./logs",
            report_to="none"
        ),
        eval_dataset=test_dataset,
    )
    metrics = trainer.evaluate()
    print("=== Résultats de l'évaluation ===")
    print(f"Perte d'évaluation : {metrics['eval_loss']}")
    return metrics
