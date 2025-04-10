import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y);

    print(f"Coefficients: {model.coef_}")
    print(f"Intercepts: {model.intercept_}")

    return model

def model_stats(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    print(model.params)

def predict_with_model(model, featureval):
    prediction = model.predict(featureval)
    return prediction

def get_conv_data(df):
    y = df["conv_time"]
    X = df[["convolution_flops", "convolution_bytes"]]
    return (X, y)

def get_cov_data(df):
    y = df["cov_time"]
    X = df[["covariance_flops", "covariance_bytes"]]
    return (X, y)

def get_htod_data(df):
    y = df["h2dtime"]
    X = df["htod_bytes"]
    return (X, y)

def get_dtoh_data(df):
    y = df["d2htime"]
    X = df["dtoh_bytes"]
    return (X, y)

def main():
    df = pd.read_csv("training.csv")
    df = df.dropna()
    X, y = get_dtoh_data(df)
    model_stats(X, y)

if __name__ == "__main__":
    main()
