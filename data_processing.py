import pandas as pd
from sklearn.model_selection import train_test_split

def build_text_files(data_csv, dest_path):
    all_text = " ".join(data_csv['text'].dropna().astype(str))
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(all_text)

def prepare_datasets(csv_path, train_path='train_dataset.txt', test_path='test_dataset.txt', test_size=0.15):
    df = pd.read_csv(csv_path)
    train, test = train_test_split(df, test_size=test_size)
    
    build_text_files(train, train_path)
    build_text_files(test, test_path)
    
    return train_path, test_path
