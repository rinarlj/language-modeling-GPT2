from transformers import pipeline

def generate_text(prompt, model_dir='./gpt2-malagasy', max_length=800):
    text_generator = pipeline(
        'text-generation', 
        model=model_dir, 
        tokenizer=model_dir, 
        config={'max_length': max_length}
    )
    return text_generator(prompt)[0]['generated_text']
