import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split  # Import GridSearchCV
from sklearn.metrics import accuracy_score  # Import accuracy score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder  # Encoding the target variable

def main():
    # Comment here:
    df = pd.read_csv('cancer.csv')

    # Comment here:
    newdf = df.dropna()
    print("Shape after dropping NAs:", newdf.shape)  # Debugging shape

    # Comment here:
    newdf = newdf.drop('id', axis=1)

    # Comment here. Will all instances be dropped?
    newdf = newdf.drop_duplicates()
    print("Shape after dropping duplicates:", newdf.shape)  # Debugging shape

    # Comment here
    target = newdf.columns[0]

    # Comment here: 
    features = []
    for feature in newdf.columns[1:]:
        features.append(feature)
    print("Features:", features)  # Debugging features list
    
    # Comment here:
    train_and_evaluate_knn(newdf, features, target, 0.3, 42)

def train_and_evaluate_knn(df, features, target, test_size, random_state):
    # Comment here:
    X = df[features]
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])  # Encode target variable
    y = df[target]

    # Comment here:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Build a pipeline: scaling followed by KNN classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'knn__n_neighbors': list(range(1, 31)),           # Number of neighbors from 1 to 30
        'knn__weights': ['uniform', 'distance'],          # Weight function options
        'knn__p': [1, 2],                                 # 1: Manhattan, 2: Euclidean
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Initialize GridSearchCV with 5-fold cross-validation and parallel processing
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters: %s", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Comment here
    y_pred = best_model.predict(X_test.values)  # Make predictions on the test set
    print("Comparing predictions")

    # Comment here: What's the definition of accuracy?
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Map encoded predictions back to original labels
    y_pred_original = le.inverse_transform(y_pred)

    # Create a DataFrame to compare predictions and actual values
    result_df = pd.DataFrame({'Predicted': y_pred_original, 'Actual': le.inverse_transform(y_test)})

    # Print each row of result_df with match/mismatch indication
    for index, row in result_df.iterrows():
        match_status = "Match" if row['Predicted'] == row['Actual'] else "Mismatch"
        print(f"Row {index}: Predicted = {row['Predicted']}, Actual = {row['Actual']}, {match_status}")

if __name__ == "__main__":
    main()

