# General Imports
import pandas as pd
import numpy as np
import glob
import time

# Model Utils
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Models
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RandomForest

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
    features = []
    for feature in master_df.columns:
        if feature == target:
            continue
        features.append(feature)

    # Label Data
    le = label_dataset(master_df, target)

    # Create Correlation Matrix
    corr_mat = master_df.corr()
    print(f"Correlation features sorted by highest correlation to lowest correlation:")
    print(f"{corr_mat['Class'].sort_values(ascending=False)}")

    # Show Correlation Matrix
    print("Close the correlation matrix window to continue...")
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.show()

    # Split dataset and create scaled alternatives
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = split_dataset(master_df, features, target, 0.3, 42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Train models that do not need scaling
    print("\n-- Training Phase --\n")
    print("Training non scaled models...")
    random_forest_trained = train_random_forest(np.copy(X_train), np.copy(y_train))
    print()

    # Train models that need scaling
    print("Training scaled models...")
    knn_trained = train_knn(np.copy(X_train_scaled), np.copy(y_train))
    print()

    # Test non scaled models
    print("\n-- Testing Phase --\n")
    print("Testing non scaled models...")
    test_dataset([random_forest_trained], X_test, y_test, ["Random Forest"], le)
    print()

    # Test scaled models
    print("Testing scaled models...")
    test_dataset([knn_trained], X_test_scaled, y_test, ["KNN"], le)

    plt.show()

def read_csv():
    # Load CSV files and read them into a CSV DF
    csv_files = glob.glob("./*.csv")
    csv_dfs = [pd.read_csv(file) for file in csv_files]

    if len(csv_files) == 0:
        raise Exception("Error: No files found in current folder for parsing")

    # Preprocessing Time!!!
    print("\n-- Exploratory Data Analysis Phase --\n")
    processed_dfs = []
    for df in csv_dfs:
        # Only keep 11 columns
        df = df[df.columns[:11]]
        
        # Force column names
        df.columns = ["filename", "xcenter", "ycenter", "area", "eccentricity", "angle", "perimeter", "Hue", "Saturation", "Class", "Temperature"]

        processed_dfs.append(df)
    
    main_df = pd.DataFrame()
    for df in processed_dfs:
        main_df = pd.concat([main_df, df])
    newdf = main_df.drop('filename', axis=1)
    print(f"Shape of data before cleaning {main_df.shape}")
    
    # Post concat cleanup
    newdf = newdf.drop_duplicates()
    print(f"Shape of data after dropping duplicates {newdf.shape}")

    # Filter out entries that don't have a valid class
    newdf = newdf[newdf["Class"].isin(["head", "torso", "arm", "leg", "car engine", "Tire", "noise"])]
    print(f"Shape of data after cleaning classes {newdf.shape}")

    # NaN non number values
    newdf = newdf[pd.to_numeric(newdf["xcenter"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["ycenter"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["area"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["eccentricity"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["angle"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["perimeter"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["Hue"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["Saturation"], errors="coerce").notnull()]
    newdf = newdf[pd.to_numeric(newdf["Temperature"], errors="coerce").notnull()]

    # Drop all NaN
    newdf = newdf.dropna(how='any')
    print(f"Shape of data after removing non number values from columns {newdf.shape}")
    return newdf

def split_dataset(df, features, target, test_size, random_state):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def label_dataset(df, target):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    return le

def train_knn(X_train, y_train):
    print("-> Training KNN <-")
    start = time.time()
    knn = KNN()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21], # How many neighbors to consider
        'weights': ['uniform', 'distance'], # Weights of points may be dependent on the distance from the point
        'metric': ['euclidean', 'manhattan'], # What distance algorithm to use
    }

    model = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    model.fit(X_train, y_train)
    print(f"\tBest KNN hyperparameters: {model.best_params_}")
    end = time.time()
    print(f"\tKNN Time: {end - start}")

    return model.best_estimator_

def train_random_forest(X_train, y_train):
    print("-> Training Random Forest <-")
    start = time.time()
    rf = RandomForest()

    param_grid = {
        'n_estimators': [50, 100, 200], # Number of estimations to run in total
        'max_depth': [15, 20, 25, 30], # Max depth of the tree
        'max_features': ['sqrt', 'log2'], # Limit a given split to use a subset of the original features
    }
    
    model = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\tBest Random Forest hyperparameters:", model.best_params_)
    end = time.time()
    print(f"\tRF Time: {end - start}")
    return model.best_estimator_

def test_dataset(model, X_test, y_test, model_name, le):
    for current_model, current_name in zip(model, model_name):
        print(f"--Testing {current_name}--")
        y_pred = current_model.predict(np.copy(X_test))

        # Accuracy
        accuracy = accuracy_score(y_test,  y_pred)
        print(f"\tAccuracy: {accuracy:.2f}")

        # Recall
        recall = recall_score(y_test,  y_pred, average="weighted")
        print(f"\tRecall: {recall:.2f}")

        # Precision
        precision = precision_score(y_test, y_pred, average="weighted")
        print(f"\tPrecision: {precision:.2f}")

        # F1
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"\tF1: {f1:.2f}")

        # Visualize Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(le.inverse_transform(y_test)))
        cm_display.plot(cmap=plt.cm.Blues)
        cm_display.ax_.set_title(current_name)
        plt.title(f"{current_name} Confusion Matrix")

        print()

if __name__ == "__main__":
    main()

