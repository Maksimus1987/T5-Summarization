import numpy as np
import os
import pandas as pd

def create_data(anomaly=False, noise=False):
    np.random.seed(42)
    days = np.arange(1, 366)
    base_temperature = 15
    temperature_variation = 5
    if anomaly:
        anomaly_day = np.random.randint(1, 365)
        anomaly_temperature = base_temperature + np.random.uniform(-temperature_variation, temperature_variation)
        temperatures = np.where(days == anomaly_day, anomaly_temperature, base_temperature + np.random.normal(0, temperature_variation, len(days)))
    else:
        temperatures = base_temperature + np.random.normal(0, temperature_variation, len(days))

    if noise:
        noise_factor = np.random.uniform(0.1, 0.5)
        temperatures += np.random.normal(0, noise_factor, len(days))

    return pd.DataFrame({'Day': days, 'Temperature': temperatures})

def save_data(data, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    data.to_csv(os.path.join(folder, filename), index=False)

# Создание и сохранение данных
for i in range(5):
    data = create_data(anomaly=i % 2 == 0, noise=i % 3 == 0)
    if i < 3:
        save_data(data, 'train', f'data_{i}.csv')
    else:
        save_data(data, 'test', f'data_{i}.csv')
