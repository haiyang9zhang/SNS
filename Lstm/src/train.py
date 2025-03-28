from data.processor import WeatherDataProcessor
from models.predictor import MultiWeatherPredictor
from src.config import MODEL_CONFIG, DATA_CONFIG
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime


def setup_training_directories():

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return model_dir, log_dir


def plot_training_history(history, feature_name, log_dir):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{feature_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_rmse'], label='RMSE')
    plt.plot(history['val_mae'], label='MAE')
    plt.title(f'{feature_name} Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{feature_name}_training_history.png'))
    plt.close()


def train_model():

    print("\n=== Weather prediction model training ===")
    model_dir, log_dir = setup_training_directories()
    data_processor = WeatherDataProcessor()

    with tqdm(total=2, desc="Data loading and processing") as pbar:
        print("\n1. Load raw data...")
        data_processor.load_data(DATA_CONFIG['history_file'], DATA_CONFIG['weather_file'])
        pbar.update(1)

        print("\n2. Prepare training data...")
        features = [
            'min_temp', 'max_temp', 'rain', 'humidity', 'cloud_cover',
            'wind_speed', 'wind_direction_numerical', 'pressure', 'visibility'
        ]
        pbar.update(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUse equipment: {device}")

    predictor = MultiWeatherPredictor(
        sequence_length=MODEL_CONFIG['sequence_length'],
        hidden_size=MODEL_CONFIG['hidden_size']
    )

    for feature in features:
        print(f"\nTraining characteristics: {feature}")
        try:
            feature_data = data_processor.prepare_feature_data(feature)
            if feature_data is None or len(feature_data) < MODEL_CONFIG['sequence_length'] + 1:
                print(f"{feature} Not enough data. Skip training")
                continue

            X, y = [], []
            for i in range(len(feature_data) - MODEL_CONFIG['sequence_length']):
                X.append(feature_data[i:(i + MODEL_CONFIG['sequence_length'])])
                y.append(feature_data[i + MODEL_CONFIG['sequence_length']])

            X = np.array(X)
            y = np.array(y)

            X = np.array([
                data_processor.scalers[feature].transform(x.reshape(-1, 1)).ravel()
                for x in X
            ])
            y = data_processor.scalers[feature].transform(y.reshape(-1, 1)).ravel()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=MODEL_CONFIG['validation_split'], random_state=42
            )

            print(f"Training set size: {len(X_train)} ")
            print(f"Test set size: {len(X_test)} ")

            X_train = torch.FloatTensor(X_train.reshape(-1, MODEL_CONFIG['sequence_length'], 1)).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_test = torch.FloatTensor(X_test.reshape(-1, MODEL_CONFIG['sequence_length'], 1)).to(device)
            y_test = torch.FloatTensor(y_test).to(device)

            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=MODEL_CONFIG['batch_size'],
                shuffle=True
            )

            if feature not in predictor.models:
                predictor._get_model(feature)

            optimizer = predictor.optimizers[feature]
            criterion = nn.MSELoss()

            history = {
                'train_loss': [],
                'val_loss': [],
                'val_rmse': [],
                'val_mae': [],
                'val_r2': []
            }

            best_val_loss = float('inf')
            early_stopping_counter = 0
            total_batches = len(train_loader) * MODEL_CONFIG['epochs']
            progress_bar = tqdm(total=total_batches, desc=f"{feature} 训练进度")

            for epoch in range(MODEL_CONFIG['epochs']):
                predictor.models[feature].train()
                total_loss = 0
                batch_count = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = predictor.models[feature](batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1
                    progress_bar.update(1)

                avg_train_loss = total_loss / batch_count

                predictor.models[feature].eval()
                with torch.no_grad():
                    val_outputs = predictor.models[feature](X_test).cpu().numpy().squeeze()
                    y_test_np = y_test.cpu().numpy()

                    val_outputs = data_processor.scalers[feature].inverse_transform(
                        val_outputs.reshape(-1, 1)
                    ).squeeze()
                    y_test_np = data_processor.scalers[feature].inverse_transform(
                        y_test_np.reshape(-1, 1)
                    ).squeeze()

                    val_loss = criterion(
                        torch.tensor(val_outputs),
                        torch.tensor(y_test_np)
                    ).item()
                    rmse = np.sqrt(np.mean((y_test_np - val_outputs) ** 2))
                    mae = np.mean(np.abs(y_test_np - val_outputs))
                    r2 = 1 - np.sum((y_test_np - val_outputs) ** 2) / np.sum(
                        (y_test_np - np.mean(y_test_np)) ** 2
                    )

                    history['train_loss'].append(avg_train_loss)
                    history['val_loss'].append(val_loss)
                    history['val_rmse'].append(rmse)
                    history['val_mae'].append(mae)
                    history['val_r2'].append(r2)

                    progress_bar.set_postfix({
                        'epoch': f"{epoch + 1}/{MODEL_CONFIG['epochs']}",
                        'train_loss': f"{avg_train_loss:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'rmse': f"{rmse:.4f}"
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stopping_counter = 0
                        predictor.save_model(feature, os.path.join(model_dir, f'{feature}_model.pth'))
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= MODEL_CONFIG['early_stopping']:
                        print(f"\n早停：{epoch + 1} 轮后无改善")
                        break

            progress_bar.close()
            plot_training_history(history, feature, log_dir)

            print(f"\n{feature} Model training completed:")
            print(f"Best proof loss: {best_val_loss:.4f}")
            print(f"finally RMSE: {rmse:.4f}")
            print(f"finally MAE: {mae:.4f}")
            print(f"finally R²: {r2:.4f}")

        except Exception as e:
            print(f"\nTraining characteristics {feature} Error: {str(e)}")
            continue

        time.sleep(1)


if __name__ == '__main__':
    try:
        start_time = time.time()
        train_model()
        end_time = time.time()
        print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} 分钟")
    except Exception as e:
        print(f"\nErrors during training: {str(e)}")