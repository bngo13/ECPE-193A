import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import argparse

training_file = ""
testing_file = ""

def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-file', type=str, required=True)
    parser.add_argument('--testing-file', type=str, required=True)

    args = parser.parse_args()

    global training_file, testing_file
    training_file = args.training_file
    testing_file = args.testing_file

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y);

    print(f"Coefficients: {model.coef_}")
    print(f"Intercepts: {model.intercept_}")

    return model

def fit_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def predict_with_model(model, featureval):
    prediction = model.predict(featureval)
    return prediction

def get_conv_data(df):
    y = df[["conv_time"]]
    X = df[["convolution_flops", "convolution_bytes"]]
    return (X, y)

def get_cov_data(df):
    y = df[["cov_time"]]
    X = df[["covariance_flops", "covariance_bytes"]]
    return (X, y)

def get_htod_data(df):
    y = df[["h2dtime"]]
    X = df[["htod_bytes"]]
    return (X, y)

def get_dtoh_data(df):
    y = df[["d2htime"]]
    X = df[["dtoh_bytes"]]
    return (X, y)

def get_models(df):
     # Convolution
    X, y = get_conv_data(df)
    conv = fit_model(X, y)

    # Covariance
    X, y = get_cov_data(df)
    cov = fit_model(X, y)

    # Host to Device
    X, y = get_htod_data(df)
    htod = fit_model(X, y)

    # Device to Host
    X, y = get_dtoh_data(df)
    dtoh = fit_model(X, y)
    return (conv, cov, htod, dtoh)

def show_model_summary(models):
    for model in models:
        print(model.summary())

def predict_models(models, df):
    for i, (model, func) in enumerate(zip(models, [get_conv_data, get_cov_data, get_htod_data, get_dtoh_data])):
        X, y = func(df)
        X = sm.add_constant(X)
        y_pred = model.predict(X)
        y_pred = y_pred.to_frame(y.columns[0])
        print(f"-- Model {i + 1} --")
        print(f"Actual Values: ")
        for i in y.values: print(i[0])
        print(f"Predicted Values:")
        for i in y_pred.values: print(i[0])

def main():
    args()

    # Get information
    training_df = pd.read_csv(training_file)
    testing_df = pd.read_csv(testing_file)
    training_df = training_df.dropna()

    # Get Models and Info
    models = get_models(training_df)
    show_model_summary(models)
    
    # Test Models
    predict_models(models, testing_df)

if __name__ == "__main__":
    main()
