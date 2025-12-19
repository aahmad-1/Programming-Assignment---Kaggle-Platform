import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

def load_and_preprocess(include_g1g2=True, data_path='./student_data/student-mat.csv'):

    math_df = pd.read_csv(data_path)
    
    if include_g1g2:
        X = math_df.drop(columns=['G3'])
    else:
        X = math_df.drop(columns=['G1', 'G2', 'G3'])
    
    y = math_df['G3']
    
    binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                   'schoolsup', 'famsup', 'paid', 'activities', 
                   'nursery', 'higher', 'internet', 'romantic']
    
    nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
    
    for col in binary_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    print("Testing with G1 and G2...")
    X_train, X_test, y_train, y_test, factors = load_and_preprocess(include_g1g2=True)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of factors: {len(factors)}")