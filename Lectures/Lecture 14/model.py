import pandas as pd
import statsmodels.api as sm

def single():
    data = pd.read_csv('children.csv')
    y = data['WGT']

    X = data[[' HGT', ' AGE']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    summary = model.summary()
    return summary

def double():
    data = pd.read_csv('children.csv')
    y = data['WGT']

    X = data[[' HGT', ' AGE']]
    X['AGE2'] = data[' AGE']**2
    X['AGE3'] = data[' AGE']**3
    X['AGE4'] = data[' AGE']**4
    X['AGE5'] = data[' AGE']**5
    X['AGE6'] = data[' AGE']**6
    X['AGE7'] = data[' AGE']**7
    X['AGE8'] = data[' AGE']**8
    X['AGE9'] = data[' AGE']**9
    X['AGE10'] = data[' AGE']**10

    X['HGT2'] = data[' HGT']**2
    X['HGT3'] = data[' HGT']**3
    X['HGT4'] = data[' HGT']**4
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    summary = model.summary()
    return summary

def main():
    one_var = single()
    two_var = double()

    print("-- Single Var --")
    print(one_var)

    print("-- Double Var --")
    print(two_var)
    
    pass

if __name__ == "__main__":
    main()