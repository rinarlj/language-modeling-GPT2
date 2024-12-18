from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def load_datasets(train_path, test_path, tokenizer):
    dataset = load_dataset("text", data_files={"train": train_path, "test": test_path})
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])
    return dataset["train"], dataset["test"]

def train_model(model, tokenizer, train_dataset, test_dataset, output_dir="./gpt2-malagasy"):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_steps=100,
        save_steps=500,
        warmup_steps=200,
        prediction_loss_only=False,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none"  
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    train_result = trainer.train()
    
    metrics = trainer.evaluate()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return {"train_result": train_result, "metrics": metrics}
