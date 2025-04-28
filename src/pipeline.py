# src/pipeline.py

from data_loader import load_data
from preprocessing import wrangle
from save_model import save_model
from train_model import train_model


def main(path: str = "./data/raw/train.csv"):
    # Step 1: Load data
    train_df = load_data(path)

    # Step 2: Preprocess data
    train_df_Preprocessed = wrangle(train_df)

    # Step 3: Train model
    predictor = train_model(train_df_Preprocessed)

    # Step 4: Save model using pickle
    save_model(predictor, "models/predictor.pkl")


if __name__ == "__main__":
    main()
