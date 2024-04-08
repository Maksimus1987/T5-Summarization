from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def prepare_and_train_model(train_data_path):
    data = pd.read_csv(train_data_path)
    X = data['Day'].values.reshape(-1, 1)
    y = data['Temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')
    return model

model = prepare_and_train_model('train/preprocessed_train.csv')

def test_model(model, test_data_path):
    data = pd.read_csv(test_data_path)
    X = data['Day'].values.reshape(-1, 1)
    y = data['Temperature']

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f'Test MSE: {mse}')

test_model(model, 'test/preprocessed_test.csv')