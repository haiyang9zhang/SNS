import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import src.config as config

class WeatherLSTM(nn.Module):
    def __init__(self, sequence_length, hidden_size=64):
        super(WeatherLSTM, self).__init__()
        self.sequence_length = sequence_length


        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)


        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out[:, -1, :]


        lstm_out = self.bn1(lstm_out)


        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class MultiWeatherPredictor:
    def __init__(self, sequence_length=7, hidden_size=64):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.criterion = nn.MSELoss()
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')

    def predict(self, feature, sequence):

        model = self._get_model(feature)
        model.eval()
        with torch.no_grad():
            sequence = torch.FloatTensor(sequence.reshape(1, self.sequence_length, 1)).to(self.device)
            prediction = model(sequence)
            return prediction.cpu().numpy()[0][0]

    def predict_multiple_days(self, feature, sequence, days_ahead):

        model = self._get_model(feature)
        model.eval()
        predictions = []
        current_sequence = sequence.copy()

        try:
            with torch.no_grad():
                for _ in range(days_ahead):

                    sequence_tensor = torch.FloatTensor(
                        current_sequence.reshape(1, self.sequence_length, 1)
                    ).to(self.device)


                    next_day_prediction = model(sequence_tensor).cpu().numpy()[0][0]
                    predictions.append(next_day_prediction)


                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = next_day_prediction

            return predictions

        except Exception as e:
            print(f"Multiple days prediction failed: {str(e)}")
            raise e

    def _get_model(self, feature_name):

        if feature_name not in self.models:
            self.models[feature_name] = WeatherLSTM(
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size
            ).to(self.device)
            self.optimizers[feature_name] = torch.optim.Adam(
                self.models[feature_name].parameters(),
                lr=config.MODEL_CONFIG['learning_rate']
            )
        return self.models[feature_name]

    def train(self, feature_name, X, y, epochs=100, batch_size=32, validation_split=0.2, progress_callback=None):

        model = self._get_model(feature_name)
        optimizer = self.optimizers[feature_name]


        split_idx = int(len(X) * (1 - validation_split))
        train_X = X[:split_idx]
        train_y = y[:split_idx]
        val_X = X[split_idx:]
        val_y = y[split_idx:]


        train_X = torch.FloatTensor(train_X.reshape(-1, self.sequence_length, 1)).to(self.device)
        train_y = torch.FloatTensor(train_y).to(self.device)
        val_X = torch.FloatTensor(val_X.reshape(-1, self.sequence_length, 1)).to(self.device)
        val_y = torch.FloatTensor(val_y).to(self.device)


        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': []
        }


        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0

            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if progress_callback:
                    progress_callback(feature_name, batch_count, epoch, loss.item())


            model.eval()
            with torch.no_grad():
                val_outputs = model(val_X).cpu().numpy().squeeze()
                val_y_np = val_y.cpu().numpy()

                val_loss = self.criterion(torch.tensor(val_outputs), torch.tensor(val_y_np)).item()
                val_mse = mean_squared_error(val_y_np, val_outputs)
                val_mae = mean_absolute_error(val_y_np, val_outputs)
                val_r2 = r2_score(val_y_np, val_outputs)

                history['val_loss'].append(val_loss)
                history['val_mse'].append(val_mse)
                history['val_mae'].append(val_mae)
                history['val_r2'].append(val_r2)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(feature_name, f'best_{feature_name}_model.pth')

        return history

    def train_all_features(self, X_dict, y_dict, epochs=100, batch_size=32, progress_callback=None):

        results = {}
        for feature_name in X_dict.keys():
            print(f"\nTraining model for {feature_name}...")
            history = self.train(
                feature_name,
                X_dict[feature_name],
                y_dict[feature_name],
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=progress_callback
            )
            results[feature_name] = history
            # 保存最终模型
            self.save_model(feature_name, f'final_{feature_name}_model.pth')
        return results


    def predict_all(self, sequences):

        predictions = {}
        for feature_name, sequence in sequences.items():
            if feature_name in self.models:
                predictions[feature_name] = self.predict(feature_name, sequence)
        return predictions

    def evaluate(self, feature, X_test, y_test):

        model = self._get_model(feature)
        model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_test).cpu().numpy().squeeze()

            metrics = {
                'rmse': np.sqrt(np.mean((y_test - predictions) ** 2)),
                'mae': np.mean(np.abs(y_test - predictions)),
                'r2': 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            }
            return metrics

    def save_model(self, feature, filename):

        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, filename)

        torch.save({
            'model_state_dict': self.models[feature].state_dict(),
            'optimizer_state_dict': self.optimizers[feature].state_dict(),
            'hidden_size': self.hidden_size,  # 保存 hidden_size
            'sequence_length': self.sequence_length
        }, filepath)

    def load_saved_model(self, feature, filename):

        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            model = WeatherLSTM(
                sequence_length=checkpoint['sequence_length'],
                hidden_size=checkpoint['hidden_size']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.models[feature] = model

            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizers[feature] = optimizer
        else:
            raise FileNotFoundError(f"No saved model found for feature: {feature}")

