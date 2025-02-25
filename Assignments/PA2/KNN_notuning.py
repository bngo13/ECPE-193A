# General Imports
import pandas as pd
import numpy as np
import glob

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

    # Split dataset and create scaled alternatives
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test, le = split_dataset(master_df, features, target, 0.3, 42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Train models that do not need scaling
    print("Training non scaled models...")
    decision_tree_trained = train_decision_tree(np.copy(X_train), np.copy(y_train))
    naive_bayes_trained = train_naive_bayes(np.copy(X_train), np.copy(y_train))
    random_forest_trained = train_random_forest(np.copy(X_train), np.copy(y_train))
    print()

    # Train models that need scaling
    print("Training scaled models...")
    knn_trained = train_knn(np.copy(X_train_scaled), np.copy(y_train))
    svm_trained = train_svm(np.copy(X_train_scaled), np.copy(y_train))
    print()

    # Test non scaled models
    print("Testing non scaled models...")
    test_dataset([decision_tree_trained, naive_bayes_trained, random_forest_trained], X_test, y_test, ["Decision Tree", "Naive Bayes", "Random Forest"])
    print()

    # Test scaled models
    print("Testing scaled models...")
    test_dataset([svm_trained, knn_trained], X_test_scaled, y_test, ["SVM", "KNN"])

    plt.show()

def read_csv():
    # Load CSV files and read them into a CSV DF
    csv_files = glob.glob("./*.csv")
    csv_dfs = [pd.read_csv(file) for file in csv_files]

    # Preprocessing Time!!!
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

    # Post concat cleanup
    newdf = main_df.drop('filename', axis=1)
    newdf = newdf.drop_duplicates()

    # Filter out entries that don't have a valid class
    newdf = newdf[newdf["Class"].isin(["head", "torso", "arm", "leg", "car engine", "Tire", "noise"])]

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
    return newdf

def split_dataset(df, features, target, test_size, random_state):
    # Comment here:
    X = df[features]
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])  # Encode target variable
    y = df[target]
    # Comment here:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, le

def train_decision_tree(X_train, y_train):
    print("-> Training Decision Trees <-")
    dt = DecisionTree()

    param_grid = {
        'criterion': ['gini', 'entropy'],  # Criterion to measure the quality of a split
        'max_depth': [10, 20, 30, 40, 50],  # Maximum depth of the tree
        'splitter': ['best', 'random']  # Strategy to choose the split at each node
    }

    model = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"\tBest Decision Tree hyperparameters: {model.best_params_}")
    return model.best_estimator_

def train_svm(X_train, y_train):
    print("-> Training SVM <-")
    svm = SVM()

    param_grid = {
        'C': [100, 125, 150], # How close the line gets to the values
        'kernel': ['linear', 'poly', 'rbf'], # Type of data fitting. RBF = circle
        'degree': [1, 2, 3], # How many degrees for the lines
    }

    model = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    model.fit(X_train, y_train)

    print(f"\tBest SVM hyperparameters: {model.best_params_}")
    return model.best_estimator_

def train_naive_bayes(X_train, y_train):
    print("-> Training Naive Bayes <-")
    nb = NaiveBayes()

    param_grid = {
        'var_smoothing': [1e-11, 1e-10, 1e-9] # Smooth the curve to allow for numbers further from distribution mean to be accounted for
    }

    model = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    model.fit(X_train, y_train)

    print(f"\tBest Naive Bayes hyperparameters: {model.best_params_}")
    return model.best_estimator_

def train_knn(X_train, y_train):
    print("-> Training KNN <-")
    knn = KNN()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21], # How many neighbors to consider
        'weights': ['uniform', 'distance'], # Weights of points may be dependent on the distance from the point
        'metric': ['euclidean', 'manhattan'], # What distance algorithm to use
    }

    model = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    model.fit(X_train, y_train)
    print(f"\tBest KNN hyperparameters: {model.best_params_}")

    return model.best_estimator_

def train_random_forest(X_train, y_train):
    print("-> Training Random Forest <-")
    rf = RandomForest()

    param_grid = {
        'n_estimators': [50, 100, 200], # Number of estimations to run in total
        'max_depth': [15, 20, 25, 30], # Max depth of the tree
        'max_features': ['sqrt', 'log2'], # Limit a given split to use a subset of the original features
    }
    
    model = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\tBest Random Forest hyperparameters:", model.best_params_)
    return model.best_estimator_

def test_dataset(model, X_test, y_test, model_name):
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

        # Visualize
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        cm_display.plot(cmap=plt.cm.Blues)
        cm_display.ax_.set_title(current_name)
        plt.title(f"{current_name} Confusion Matrix")

        print()

if __name__ == "__main__":
    main()

