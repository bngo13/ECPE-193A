# General Imports
import pandas as pd
import numpy as np
import glob
import time

# Model Utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Models (cuML, fast on GPU)
from cuml.neighbors import KNeighborsClassifier as KNN
from cuml.ensemble import RandomForestClassifier as RandomForest

# Scoring
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Visual Scoring
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


def main():
    # Read CSV
    master_df = read_csv()
    target = master_df.columns[8]

    # Get features from CSV
    features = [col for col in master_df.columns if col != target]

    # Label Data
    le = label_dataset(master_df, target)

    # Split dataset and create scaled alternatives
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = split_dataset(master_df, features, target, 0.3, 42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models that do not need scaling
    print("\n-- Training Phase --\n")
    print("Training non scaled models...")
    start = time.time()
    rf_model = RandomForest(n_estimators=100, max_depth=20, max_features='sqrt')
    rf_model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int32))
    end = time.time()
    print(f"RF: {end-start}")
    print()

    # Train models that need scaling
    print("Training scaled models...")
    start = time.time()
    knn_model = KNN(n_neighbors=5, metric='euclidean')
    knn_model.fit(X_train_scaled.astype(np.float32), y_train.values.astype(np.int32))
    end = time.time()
    print(f"KNN: {end - start}")
    print()

    # Test non scaled models
    print("\n-- Testing Phase --\n")
    print("Testing non scaled models...")
    start = time.time()
    test_dataset([rf_model], X_test.values, y_test.values, ["Random Forest"], le)
    end = time.time()
    print(f"RF Train: {end - start}")
    print()

    # Test scaled models
    print("Testing scaled models...")
    start = time.time()
    test_dataset([knn_model], X_test_scaled, y_test.values, ["KNN"], le)
    end = time.time()
    print(f"KNN Train: {end - start}")

    plt.show()


def read_csv():
    # Load CSV files and read them into a CSV DF
    csv_files = glob.glob("./*.csv")
    if len(csv_files) == 0:
        raise Exception("Error: No files found in current folder for parsing")
    
    csv_dfs = [pd.read_csv(file) for file in csv_files]

    print("\n-- Exploratory Data Analysis Phase --\n")
    processed_dfs = []
    for df in csv_dfs:
        df = df[df.columns[:11]]
        df.columns = ["filename", "xcenter", "ycenter", "area", "eccentricity", "angle", "perimeter", "Hue", "Saturation", "Class", "Temperature"]
        processed_dfs.append(df)

    main_df = pd.concat(processed_dfs)
    newdf = main_df.drop('filename', axis=1)
    print(f"Shape of data before cleaning {main_df.shape}")
    
    newdf = newdf.drop_duplicates()
    print(f"Shape of data after dropping duplicates {newdf.shape}")

    newdf = newdf[newdf["Class"].isin(["head", "torso", "arm", "leg", "car engine", "Tire", "noise"])]
    print(f"Shape of data after cleaning classes {newdf.shape}")

    # Remove rows with non-numeric values
    for col in ["xcenter", "ycenter", "area", "eccentricity", "angle", "perimeter", "Hue", "Saturation", "Temperature"]:
        newdf = newdf[pd.to_numeric(newdf[col], errors="coerce").notnull()]
    
    newdf = newdf.dropna()
    print(f"Shape of data after removing non number values from columns {newdf.shape}")
    return newdf


def split_dataset(df, features, target, test_size, random_state):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def label_dataset(df, target):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    return le


def test_dataset(models, X_test, y_test, model_names, le):
    for model, name in zip(models, model_names):
        print(f"--Testing {name}--")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\tAccuracy: {accuracy:.2f}")
        print(f"\tRecall: {recall:.2f}")
        print(f"\tPrecision: {precision:.2f}")
        print(f"\tF1: {f1:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        cm_display.plot(cmap=plt.cm.Blues)
        cm_display.ax_.set_title(name)
        plt.title(f"{name} Confusion Matrix")
        print()


if __name__ == "__main__":
    main()

