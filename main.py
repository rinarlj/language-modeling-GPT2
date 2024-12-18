import os
from data_processing import prepare_datasets
from model_setup import setup_model_and_tokenizer
from training import load_datasets, train_model
from text_generation import generate_text

def main():
    # 1. Définir les chemins des fichiers
    csv_path = "./data/rakitra.csv"  # Adapter le chemin selon ton environnement
    output_dir = "./gpt2-malagasy"
    train_path, test_path = "./data/train_dataset.txt", "./data/test_dataset.txt"
    
    # 2. Préparer les données
    print("Préparation des jeux de données...")
    train_path, test_path = prepare_datasets(csv_path, train_path, test_path)
    
    # 3. Configurer le modèle et le tokenizer
    print("Configuration du modèle et du tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # 4. Charger les datasets
    print("Chargement des datasets...")
    train_dataset, test_dataset = load_datasets(train_path, test_path, tokenizer)
    
    # 5. Entraîner le modèle
    print("Début de l'entraînement...")
    train_model(model, tokenizer, train_dataset, test_dataset, output_dir)
    
    # 6. Générer du texte
    print("Génération de texte...")
    prompt = "Misy fomba mahomby"
    result = generate_text(prompt, output_dir)
    print(f"Texte généré : {result}")

if __name__ == "__main__":
    main()
