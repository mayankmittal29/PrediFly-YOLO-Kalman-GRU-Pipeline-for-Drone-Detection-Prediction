import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import cv2

class DronePathDataset(Dataset):
    """Enhanced dataset for drone path data with augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and np.random.random() < 0.5:
            # Apply random noise augmentation
            noise = torch.randn_like(x) * 0.03
            x = x + noise
            
        return x, y

class CNNGRUAttention(nn.Module):
    """
    Hybrid CNN-GRU model with attention for drone path prediction
    
    Architecture:
    1. Conv1D layers to extract spatial features
    2. GRU layers to process temporal sequence
    3. Attention mechanism to focus on important timesteps
    4. Dense layers for prediction refinement
    """
    def __init__(self, input_size, seq_length, hidden_size=128, num_layers=2, 
                 cnn_filters=[64, 128], dropout=0.3, output_size=2):
        super(CNNGRUAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # CNN layers for spatial feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_size
        
        for filters in cnn_filters:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size=3, padding=1),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Conv1d(filters, filters, kernel_size=3, padding=1),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                )
            )
            in_channels = filters
        
        # Bidirectional GRU for temporal features
        self.gru = nn.GRU(
            input_size=cnn_filters[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # * 2 for bidirectional
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_size)
        
        # Uncertainty estimation (predict mean and variance)
        self.uncertainty = nn.Linear(64, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input for CNN: [batch, seq_len, features] -> [batch, features, seq_len]
        x_cnn = x.permute(0, 2, 1)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        
        # Reshape back for GRU: [batch, features, seq_len] -> [batch, seq_len, features]
        x_gru = x_cnn.permute(0, 2, 1)
        
        # Apply GRU
        gru_out, _ = self.gru(x_gru)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(gru_out).squeeze(-1), dim=1)
        attention_weights = attention_weights.unsqueeze(2)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Fully connected layers with residual connections
        fc1_out = self.fc1(context_vector)
        fc1_out = self.bn1(fc1_out)
        fc1_out = F.relu(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)
        fc2_out = F.relu(fc2_out)
        
        # Final prediction
        position_output = self.fc3(fc2_out)
        
        # Uncertainty estimation (log variance)
        uncertainty = torch.exp(self.uncertainty(fc2_out))
        
        return position_output, uncertainty


def prepare_sequences(df, seq_length=30, stride=1, features=None):
    """Prepare sequences for training with enhanced feature engineering"""
    if features is None:
        # Default features
        features = ['x', 'y', 'smoothed_velocity', 'height_meters']
    
    # Add derived features for better prediction
    if 'smoothed_velocity' in df.columns:
        # Calculate acceleration (change in velocity)
        df['acceleration'] = df['smoothed_velocity'].diff().fillna(0)
        if 'acceleration' not in features:
            features.append('acceleration')
    
    if 'x' in df.columns and 'y' in df.columns:
        # Calculate direction vectors
        df['dx'] = df['x'].diff().fillna(0)
        df['dy'] = df['y'].diff().fillna(0)
        
        # Calculate angle
        df['angle'] = np.arctan2(df['dy'], df['dx'])
        
        # Calculate curvature (change in angle)
        df['curvature'] = df['angle'].diff().fillna(0)
        
        if 'dx' not in features:
            features.append('dx')
        if 'dy' not in features:
            features.append('dy')
        if 'angle' not in features:
            features.append('angle')
        if 'curvature' not in features:
            features.append('curvature')
    
    print(f"Using features: {features}")
    
    # Scale features to [0, 1] range for better training
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    
    # Create sequences with stride for data augmentation
    X, y = [], []
    for i in range(0, len(df_scaled) - seq_length, stride):
        # Input sequence
        X.append(df_scaled.iloc[i:i+seq_length].values)
        
        # Target is next position
        target_idx = i + seq_length
        y.append(df[['x', 'y']].iloc[target_idx].values)
    
    return np.array(X), np.array(y), scaler, features


def train_model(model, train_loader, val_loader, device, epochs=100, patience=15):
    """Train the CNN-GRU model with advanced techniques"""
    # Loss function: combination of MSE and uncertainty-aware loss
    def gaussian_nll_loss(pred, target, variance):
        """Gaussian negative log likelihood loss with uncertainty"""
        return 0.5 * (torch.log(variance) + (pred - target)**2 / variance).sum(dim=1).mean()
    
    # Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=patience//3, factor=0.5, verbose=True
    )
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_model = None
    early_stop_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            predictions, uncertainties = model(inputs)
            
            # Compute loss
            mse_loss = F.mse_loss(predictions, targets)
            uncertainty_loss = gaussian_nll_loss(predictions, targets, uncertainties + 1e-6)
            loss = mse_loss + 0.1 * uncertainty_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions, uncertainties = model(inputs)
                
                mse_loss = F.mse_loss(predictions, targets)
                uncertainty_loss = gaussian_nll_loss(predictions, targets, uncertainties + 1e-6)
                loss = mse_loss + 0.1 * uncertainty_loss
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model and check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Load best model
    model.load_state_dict(best_model)
    # Save the model (add these lines)
    torch.save(model.state_dict(), '/kaggle/working/drone_path_model_frogjump.pth')
    print("Model saved to drone_path_model.pth")
    return model, train_losses, val_losses


def predict_with_uncertainty(model, df, scaler, features, seq_length=30, future_steps=120, num_samples=20):
    """Predict future frames with uncertainty estimation using Monte Carlo sampling"""
    # Get initial sequence
    initial_seq = df[features].iloc[:seq_length].values
    initial_seq_scaled = scaler.transform(initial_seq)
    
    # Prepare for prediction
    device = next(model.parameters()).device
    current_seq = torch.tensor(initial_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Containers for predictions and uncertainties
    all_predictions = []
    all_uncertainties = []
    
    # Predict for each future step
    model.eval()
    
    with torch.no_grad():
        for _ in range(future_steps):
            # Generate multiple predictions using Monte Carlo dropout
            model.train()  # Enable dropout for MC sampling
            sample_predictions = []
            
            for _ in range(num_samples):
                pred, uncertainty = model(current_seq)
                sample_predictions.append(pred[0].cpu().numpy())
            
            # Compute mean and uncertainty from samples
            sample_predictions = np.array(sample_predictions)
            mean_prediction = np.mean(sample_predictions, axis=0)
            prediction_std = np.std(sample_predictions, axis=0)
            
            # Create full feature vector (we need to add other features beyond position)
            frame_idx = len(all_predictions) + seq_length
            if frame_idx < len(df):
                next_features = df[features].iloc[frame_idx].values.copy()
                next_features_scaled = scaler.transform([next_features])[0]
                
                # Replace position with our prediction
                next_features_scaled[0] = mean_prediction[0]  # x
                next_features_scaled[1] = mean_prediction[1]  # y
                
                # Add to predictions and uncertainties
                all_predictions.append(mean_prediction)
                all_uncertainties.append(prediction_std)
                
                # Update sequence
                current_seq = torch.cat([
                    current_seq[:, 1:, :],
                    torch.tensor(next_features_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                ], dim=1)
            else:
                break
    
    # Inverse transform predictions
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    
    return all_predictions, all_uncertainties


def ensemble_prediction(models, df, scaler, features, seq_length=30, future_steps=120):
    """Use an ensemble of models for more robust prediction"""
    # Get initial sequence
    initial_seq = df[features].iloc[:seq_length].values
    initial_seq_scaled = scaler.transform(initial_seq)
    
    # Containers for all model predictions
    all_model_predictions = []
    
    # Get predictions from each model
    for model in models:
        device = next(model.parameters()).device
        current_seq = torch.tensor(initial_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Container for this model's predictions
        model_predictions = []
        
        # Predict future steps
        model.eval()
        with torch.no_grad():
            for _ in range(future_steps):
                # Predict next position
                pred, _ = model(current_seq)
                next_pos = pred[0].cpu().numpy()
                
                # Create full feature vector
                frame_idx = len(model_predictions) + seq_length
                if frame_idx < len(df):
                    next_features = df[features].iloc[frame_idx].values.copy()
                    next_features_scaled = scaler.transform([next_features])[0]
                    
                    # Replace position with our prediction
                    next_features_scaled[0] = next_pos[0]  # x
                    next_features_scaled[1] = next_pos[1]  # y
                    
                    # Add to predictions
                    model_predictions.append(next_pos)
                    
                    # Update sequence
                    current_seq = torch.cat([
                        current_seq[:, 1:, :],
                        torch.tensor(next_features_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    ], dim=1)
                else:
                    break
        
        all_model_predictions.append(np.array(model_predictions))
    
    # Average predictions across models
    ensemble_predictions = np.mean(all_model_predictions, axis=0)
    # Calculate uncertainty as standard deviation across models
    ensemble_uncertainties = np.std(all_model_predictions, axis=0)
    
    return ensemble_predictions, ensemble_uncertainties


def create_comparison_video(input_video, output_video, df, predicted_positions, 
                           uncertainties, training_end_frame, seq_length):
    """Create a video showing actual vs predicted paths with uncertainty visualization"""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Extract paths for visualization
    training_path = df[['x', 'y']].iloc[:training_end_frame].values
    actual_test_path = df[['x', 'y']].iloc[training_end_frame:].values
    predicted_path = predicted_positions
    
    # Calculate errors for visualization
    errors = np.zeros(len(predicted_path))
    for i in range(min(len(predicted_path), len(actual_test_path))):
        errors[i] = np.sqrt(((actual_test_path[i] - predicted_path[i]) ** 2).sum())
    
    max_error = max(errors) if len(errors) > 0 else 1
    
    # Process each frame
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for visualization
        viz_frame = frame.copy()
        
        # Add semi-transparent overlay for better visibility
        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height-80), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, viz_frame, 0.3, 0, viz_frame)
        
        # Draw the training path (blue)
        for i in range(len(training_path)-1):
            pt1 = (int(training_path[i][0]), int(training_path[i][1]))
            pt2 = (int(training_path[i+1][0]), int(training_path[i+1][1]))
            cv2.line(viz_frame, pt1, pt2, (255, 0, 0), 2)  # Blue color
        
        # Draw the actual test path (green)
        for i in range(len(actual_test_path)-1):
            pt1 = (int(actual_test_path[i][0]), int(actual_test_path[i][1]))
            pt2 = (int(actual_test_path[i+1][0]), int(actual_test_path[i+1][1]))
            cv2.line(viz_frame, pt1, pt2, (0, 255, 0), 2)  # Green color
        
        # Draw the predicted path with confidence intervals
        for i in range(len(predicted_path)-1):
            pt1 = (int(predicted_path[i][0]), int(predicted_path[i][1]))
            pt2 = (int(predicted_path[i+1][0]), int(predicted_path[i+1][1]))
            
            # Color based on error (red to yellow gradient)
            if i < len(errors):
                error_ratio = min(errors[i] / max_error, 1.0)
                color = (0, int(255 * (1-error_ratio)), int(255 * error_ratio))  # Yellow (low error) to Red (high error)
            else:
                color = (0, 0, 255)  # Default red
                
            cv2.line(viz_frame, pt1, pt2, color, 2)
            
            # Draw uncertainty ellipses if available
            if i < len(uncertainties):
                # Calculate ellipse parameters
                uncertainty_x = uncertainties[i][0] * 3  # 3-sigma for 99.7% confidence
                uncertainty_y = uncertainties[i][1] * 3
                
                cv2.ellipse(viz_frame, 
                           center=pt1,
                           axes=(int(max(5, uncertainty_x * 20)), int(max(5, uncertainty_y * 20))),
                           angle=0,
                           startAngle=0,
                           endAngle=360,
                           color=(255, 0, 255),  # Magenta
                           thickness=1)
            
        # Draw current drone position
        if frame_idx < len(df):
            current_x, current_y = int(df.iloc[frame_idx]['x']), int(df.iloc[frame_idx]['y'])
            cv2.circle(viz_frame, (current_x, current_y), 12, (255, 255, 255), -1)  # White circle
            cv2.circle(viz_frame, (current_x, current_y), 8, (0, 0, 0), -1)  # Black inner circle
            
            # Add label for current phase
            phase_text = "TRAINING PHASE" if frame_idx < training_end_frame else "PREDICTION PHASE"
            cv2.putText(viz_frame, phase_text, (width - 300, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Add timestamp
            if 'timestamp' in df.columns:
                timestamp = df.iloc[frame_idx]['timestamp']
                cv2.putText(viz_frame, f"Time: {timestamp:.2f}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # If in prediction phase, show error and uncertainty
            if frame_idx >= training_end_frame and frame_idx - training_end_frame < len(predicted_positions):
                pred_idx = frame_idx - training_end_frame
                pred_x, pred_y = int(predicted_positions[pred_idx][0]), int(predicted_positions[pred_idx][1])
                
                # Calculate and display error
                error = np.sqrt((current_x - pred_x)**2 + (current_y - pred_y)**2)
                
                # Color code based on error magnitude
                if error < 20:
                    error_color = (0, 255, 0)  # Green for low error
                elif error < 50:
                    error_color = (0, 255, 255)  # Yellow for medium error
                else:
                    error_color = (0, 0, 255)  # Red for high error
                    
                cv2.putText(viz_frame, f"Error: {error:.1f} px", (10, height - 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, error_color, 2)
                
                # Show uncertainty if available
                if pred_idx < len(uncertainties):
                    unc_x, unc_y = uncertainties[pred_idx]
                    cv2.putText(viz_frame, f"Uncertainty: {np.mean([unc_x, unc_y]):.3f}", 
                               (10, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                # Draw line between actual and predicted
                cv2.line(viz_frame, (current_x, current_y), (pred_x, pred_y), 
                         error_color, 2, cv2.LINE_AA)
                
                # Draw predicted position
                cv2.circle(viz_frame, (pred_x, pred_y), 10, (0, 0, 255), -1)
        
        # Add legend
        cv2.putText(viz_frame, "Training Path", (10, height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Blue
        cv2.putText(viz_frame, "Actual Test Path", (200, height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green
        cv2.putText(viz_frame, "Predicted Path", (400, height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red
        
        # Write frame
        out.write(viz_frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Comparison video saved to {output_video}")
    return True


def calculate_comprehensive_metrics(actual_positions, predicted_positions, uncertainties=None):
    """
    Calculate comprehensive error metrics for path prediction
    
    Parameters:
        actual_positions: numpy array of actual positions (x, y)
        predicted_positions: numpy array of predicted positions (x, y)
        uncertainties: optional numpy array of uncertainties for each prediction
        
    Returns:
        Dictionary of error metrics
    """
    # Ensure arrays are the same length
    min_len = min(len(actual_positions), len(predicted_positions))
    actual = actual_positions[:min_len]
    predicted = predicted_positions[:min_len]
    
    # Calculate euclidean distance errors
    euclidean_errors = np.sqrt(np.sum((actual - predicted) ** 2, axis=1))
    
    # Calculate component errors
    x_errors = actual[:, 0] - predicted[:, 0]
    y_errors = actual[:, 1] - predicted[:, 1]
    
    # Basic statistics
    metrics = {
        'mean_error': np.mean(euclidean_errors),
        'median_error': np.median(euclidean_errors),
        'max_error': np.max(euclidean_errors),
        'min_error': np.min(euclidean_errors),
        'std_error': np.std(euclidean_errors),
        'rmse': np.sqrt(np.mean(euclidean_errors**2)),
        'mae_x': np.mean(np.abs(x_errors)),
        'mae_y': np.mean(np.abs(y_errors)),
        'rmse_x': np.sqrt(np.mean(x_errors**2)),
        'rmse_y': np.sqrt(np.mean(y_errors**2)),
        'total_cumulative_error': np.sum(euclidean_errors),
        'euclidean_errors': euclidean_errors,
        'x_errors': x_errors,
        'y_errors': y_errors
    }
    
    # Error percentiles
    for p in [50, 75, 90, 95, 99]:
        metrics[f'percentile_{p}'] = np.percentile(euclidean_errors, p)
    
    # Calculate error growth rate (slope of error over time)
    if min_len > 2:
        frames = np.arange(min_len)
        error_slope = np.polyfit(frames, euclidean_errors, 1)[0]
        metrics['error_growth_rate'] = error_slope
    
    # Calculate cumulative error over time
    metrics['cumulative_error'] = np.cumsum(euclidean_errors)
    
    # Calculate normalized error (by path length)
    path_lengths = np.sqrt(np.sum(np.diff(actual, axis=0)**2, axis=1))
    total_path_length = np.sum(path_lengths)
    if total_path_length > 0:
        metrics['normalized_error'] = metrics['total_cumulative_error'] / total_path_length
    
    # Calculate direction accuracy
    if min_len > 1:
        actual_directions = np.arctan2(np.diff(actual[:, 1]), np.diff(actual[:, 0]))
        pred_directions = np.arctan2(np.diff(predicted[:, 1]), np.diff(predicted[:, 0]))
        
        # Convert to degrees and ensure positive angles
        actual_degrees = np.degrees(actual_directions) % 360
        pred_degrees = np.degrees(pred_directions) % 360
        
        # Calculate angular error (accounting for 360 degree wrap-around)
        angular_errors = np.minimum(np.abs(actual_degrees - pred_degrees), 
                                   360 - np.abs(actual_degrees - pred_degrees))
        
        metrics['mean_angular_error'] = np.mean(angular_errors)
        metrics['angular_errors'] = angular_errors
    
    # Uncertainty evaluation if provided
    if uncertainties is not None:
        # Limit uncertainties to same length
        uncertainties = uncertainties[:min_len]
        
        # Average uncertainty
        metrics['mean_uncertainty'] = np.mean(np.mean(uncertainties, axis=1))
        
        # Uncertainty calibration (correlation between error and uncertainty)
        if min_len > 2:
            metrics['uncertainty_correlation'] = np.corrcoef(
                np.mean(uncertainties, axis=1), euclidean_errors)[0, 1]
            
            # Calculate what percentage of actual positions fall within uncertainty bounds
            # Using 2-sigma (95% confidence) uncertainty bounds
            in_bounds_count = 0
            for i in range(min_len):
                if euclidean_errors[i] <= 2 * np.mean(uncertainties[i]):
                    in_bounds_count += 1
            
            metrics['calibration_score'] = in_bounds_count / min_len
    
    return metrics


def visualize_error_analysis(actual_positions, predicted_positions, uncertainties=None, 
                             metrics=None, df=None, training_end_frame=None, save_path=None):
    """
    Create comprehensive visualizations of prediction errors
    
    Parameters:
        actual_positions: numpy array of actual positions (x, y)
        predicted_positions: numpy array of predicted positions (x, y)
        uncertainties: optional numpy array of uncertainties for each prediction
        metrics: dictionary of error metrics (if already calculated)
        df: optional dataframe with original data for additional context
        training_end_frame: index where training data ends and prediction begins
        save_path: path to save the figures
    """
    # Calculate metrics if not provided
    if metrics is None:
        metrics = calculate_comprehensive_metrics(actual_positions, predicted_positions, uncertainties)
    
    # Create a multi-plot figure for comprehensive error analysis
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('CNN-GRU Drone Path Prediction Error Analysis', fontsize=20)
    
    grid = plt.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Prediction vs Actual Path Comparison
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(actual_positions[:, 0], actual_positions[:, 1], 'g-', label='Actual Path', linewidth=2)
    ax1.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'r-', label='Predicted Path', linewidth=2)
    
    # Draw uncertainty ellipses if available
    if uncertainties is not None:
        for i in range(0, len(predicted_positions), max(1, len(predicted_positions)//20)):  # Draw every 5% of points
            x, y = predicted_positions[i]
            unc_x, unc_y = uncertainties[i]
            
            # Create ellipse (2-sigma)
            ellipse = plt.matplotlib.patches.Ellipse(
                (x, y), width=4*unc_x, height=4*unc_y, 
                edgecolor='purple', facecolor='none', alpha=0.5
            )
            ax1.add_patch(ellipse)
    
    # Add connecting lines between actual and predicted points
    for i in range(0, len(actual_positions), max(1, len(actual_positions)//10)):
        ax1.plot([actual_positions[i, 0], predicted_positions[i, 0]], 
                [actual_positions[i, 1], predicted_positions[i, 1]], 
                'k-', alpha=0.3)
    
    ax1.set_title('Path Comparison')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Error Over Time
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.plot(metrics['euclidean_errors'], 'b-', linewidth=2)
    
    # Add trend line
    frames = np.arange(len(metrics['euclidean_errors']))
    if len(frames) > 1:
        z = np.polyfit(frames, metrics['euclidean_errors'], 1)
        p = np.poly1d(z)
        ax2.plot(frames, p(frames), "r--", linewidth=2, 
               label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
    
    ax2.axhline(y=metrics['mean_error'], color='g', linestyle='--', 
               label=f'Mean: {metrics["mean_error"]:.2f}')
    ax2.axhline(y=metrics['median_error'], color='orange', linestyle='--', 
               label=f'Median: {metrics["median_error"]:.2f}')
    
    ax2.set_title('Error Over Time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Error (pixels)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Error Distribution Histogram
    ax3 = fig.add_subplot(grid[0, 2])
    sns.histplot(metrics['euclidean_errors'], kde=True, ax=ax3, bins=20, color='purple')
    
    ax3.axvline(x=metrics['mean_error'], color='r', linestyle='--', 
               label=f'Mean: {metrics["mean_error"]:.2f}')
    ax3.axvline(x=metrics['median_error'], color='g', linestyle='--', 
               label=f'Median: {metrics["median_error"]:.2f}')
    
    ax3.set_title('Error Distribution')
    ax3.set_xlabel('Error (pixels)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Cumulative Error
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.plot(metrics['cumulative_error'], 'g-', linewidth=2)
    ax4.set_title('Cumulative Error')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Cumulative Error (pixels)')
    ax4.grid(True)
    
    # 5. X and Y Error Components
    ax5 = fig.add_subplot(grid[1, 1])
    ax5.plot(metrics['x_errors'], 'r-', label='X Error', alpha=0.7)
    ax5.plot(metrics['y_errors'], 'b-', label='Y Error', alpha=0.7)
    ax5.set_title('X and Y Error Components')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Error (pixels)')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Error vs Velocity (if available)
    ax6 = fig.add_subplot(grid[1, 2])
    if df is not None and 'smoothed_velocity' in df.columns:
        # Get velocities for the prediction period
        if training_end_frame is not None:
            velocities = df['smoothed_velocity'].iloc[training_end_frame:training_end_frame+len(metrics['euclidean_errors'])].values
            velocities = velocities[:len(metrics['euclidean_errors'])]
            
            ax6.scatter(velocities, metrics['euclidean_errors'], alpha=0.7, c='purple')
            
            # Add trend line
            if len(velocities) > 1:
                z = np.polyfit(velocities, metrics['euclidean_errors'], 1)
                p = np.poly1d(z)
                sorted_vel = np.sort(velocities)
                ax6.plot(sorted_vel, p(sorted_vel), "r--", linewidth=2, 
                        label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
            
            ax6.set_title('Error vs. Velocity')
            ax6.set_xlabel('Velocity')
            ax6.set_ylabel('Error (pixels)')
            ax6.legend()
            ax6.grid(True)
        else:
            ax6.text(0.5, 0.5, 'Training end frame not provided', 
                    horizontalalignment='center', verticalalignment='center')
    else:
        ax6.text(0.5, 0.5, 'Velocity data not available', 
                horizontalalignment='center', verticalalignment='center')
    
    # 7. Angular Error Distribution (if calculated)
    ax7 = fig.add_subplot(grid[2, 0])
    if 'angular_errors' in metrics:
        sns.histplot(metrics['angular_errors'], kde=True, ax=ax7, bins=20, color='orange')
        ax7.axvline(x=metrics['mean_angular_error'], color='r', linestyle='--', 
                   label=f'Mean: {metrics["mean_angular_error"]:.2f}°')
        ax7.set_title('Angular Error Distribution')
        ax7.set_xlabel('Angular Error (degrees)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, 'Angular error not calculated', 
                horizontalalignment='center', verticalalignment='center')
    
    # 8. Error Metrics Summary
    ax8 = fig.add_subplot(grid[2, 1])
    
    # Select key metrics for display
    key_metrics = {
        'Mean Error': metrics['mean_error'],
        'Median Error': metrics['median_error'],
        'RMSE': metrics['rmse'],
        'RMSE X': metrics['rmse_x'],
        'RMSE Y': metrics['rmse_y'],
        'Error StdDev': metrics['std_error'],
        'P90 Error': metrics['percentile_90']
    }
    
    # Add uncertainty metrics if available
    if 'mean_uncertainty' in metrics:
        key_metrics['Mean Uncertainty'] = metrics['mean_uncertainty']
    if 'uncertainty_correlation' in metrics:
        key_metrics['Unc-Error Corr'] = metrics['uncertainty_correlation']
    if 'calibration_score' in metrics:
        key_metrics['Calibration'] = metrics['calibration_score']
    
    # Create bar chart
    bars = ax8.bar(range(len(key_metrics)), list(key_metrics.values()), color='teal')
    ax8.set_xticks(range(len(key_metrics)))
    ax8.set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
    ax8.set_title('Error Metrics Summary')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 9. 2D Error Heatmap
    ax9 = fig.add_subplot(grid[2, 2])
    
    # Create a 2D histogram using X and Y errors
    h = ax9.hist2d(metrics['x_errors'], metrics['y_errors'], bins=30, cmap='hot')
    plt.colorbar(h[3], ax=ax9)
    
    # Add origin lines
    ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax9.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add error ellipse (1-sigma and 2-sigma)
    from matplotlib.patches import Ellipse
    std_x = np.std(metrics['x_errors'])
    std_y = np.std(metrics['y_errors'])
    mean_x = np.mean(metrics['x_errors'])
    mean_y = np.mean(metrics['y_errors'])
    
    ellipse1 = Ellipse((mean_x, mean_y), width=2*std_x, height=2*std_y, 
                     edgecolor='blue', facecolor='none', label='1σ')
    ellipse2 = Ellipse((mean_x, mean_y), width=4*std_x, height=4*std_y, 
                     edgecolor='green', facecolor='none', label='2σ', linestyle='--')
    
    ax9.add_patch(ellipse1)
    ax9.add_patch(ellipse2)
    ax9.set_title('2D Error Distribution')
    ax9.set_xlabel('X Error (pixels)')
    ax9.set_ylabel('Y Error (pixels)')
    ax9.legend()
    
    # Adjust layout and save if requested
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Error analysis visualization saved to: {save_path}")
    
    plt.show()
    
    # Additional error growth analysis plot
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['euclidean_errors'], 'b-', label='Error', linewidth=2)
    
    # Calculate moving average for smooth visualization
    window_size = max(3, len(metrics['euclidean_errors']) // 10)
    moving_avg = np.convolve(metrics['euclidean_errors'], 
                           np.ones(window_size)/window_size, 
                           mode='valid')
    
    # Pad the beginning of the moving average
    padding = len(metrics['euclidean_errors']) - len(moving_avg)
    moving_avg = np.pad(moving_avg, (padding, 0), 'edge')
    
    plt.plot(moving_avg, 'r-', label=f'Moving Avg (window={window_size})', linewidth=2)
    
    # Add uncertainty bands if available
    if 'std_error' in metrics:
        plt.fill_between(
            np.arange(len(metrics['euclidean_errors'])),
            moving_avg - metrics['std_error'],
            moving_avg + metrics['std_error'],
            color='gray', alpha=0.3, label='±1σ Band'
        )
    
    plt.title('Error Growth Analysis')
    plt.xlabel('Frame')
    plt.ylabel('Error (pixels)')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        error_growth_path = save_path.replace('.png', '_error_growth.png')
        plt.savefig(error_growth_path)
        print(f"Error growth analysis saved to: {error_growth_path}")
    
    plt.show()
    
    # Create heatmap of prediction errors on the path
    plt.figure(figsize=(12, 10))
    
    # Plot paths
    plt.plot(actual_positions[:, 0], actual_positions[:, 1], 'g-', alpha=0.7, linewidth=2, label='Actual Path')
    plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'r-', alpha=0.7, linewidth=2, label='Predicted Path')
    
    # Create a colormap for error magnitude
    cmap = plt.cm.jet
    norm = plt.Normalize(0, np.max(metrics['euclidean_errors']))
    
    # Draw error points with color based on magnitude
    for i in range(len(metrics['euclidean_errors'])):
        plt.plot(
            [actual_positions[i, 0], predicted_positions[i, 0]],
            [actual_positions[i, 1], predicted_positions[i, 1]],
            color=cmap(norm(metrics['euclidean_errors'][i])),
            alpha=0.6,
            linewidth=2
        )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Error Magnitude (pixels)')
    
    plt.title('Spatial Distribution of Prediction Errors')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        error_heatmap_path = save_path.replace('.png', '_error_heatmap.png')
        plt.savefig(error_heatmap_path)
        print(f"Error heatmap saved to: {error_heatmap_path}")
    
    plt.show()
    
    return metrics


def visualize_attention_weights(model, test_sequence, feature_names=None, save_path=None):
    """
    Visualize attention weights from the model to understand which parts of the
    sequence the model is focusing on for its predictions
    
    Parameters:
        model: trained model with attention mechanism
        test_sequence: a sample input sequence to visualize attention for
        feature_names: list of feature names for better visualization
        save_path: path to save the figure
    """
    # Set model to evaluation mode
    model.eval()
    
    # Process the sequence
    device = next(model.parameters()).device
    with torch.no_grad():
        if not isinstance(test_sequence, torch.Tensor):
            test_sequence = torch.tensor(test_sequence, dtype=torch.float32)
        
        if len(test_sequence.shape) == 2:
            # Add batch dimension if missing
            test_sequence = test_sequence.unsqueeze(0)
            
        # Move to device
        test_sequence = test_sequence.to(device)
        
        # Forward pass to get attention weights
        # Use a hook to capture attention weights from the forward pass
        attention_weights = []
        
        def get_activation(name):
            def hook(model, input, output):
                attention_weights.append(output.detach().cpu().numpy())
            return hook
        
        # Assuming model.attention layer outputs attention weights
        # You might need to modify this based on the actual model architecture
        hook_handle = model.attention[-1].register_forward_hook(get_activation('attention'))
        
        # Forward pass
        model(test_sequence)
        
        # Remove the hook
        hook_handle.remove()
    
    # Check if we captured attention weights
    if not attention_weights:
        print("Could not capture attention weights. The model architecture might need modification.")
        return
    
    # Get the attention weights (might need adjustment based on actual output format)
    weights = attention_weights[0]
    if len(weights.shape) > 2:
        weights = weights.squeeze()
    
    # If it's still multi-dimensional, take the mean across appropriate dimensions
    if len(weights.shape) > 1:
        weights = weights.mean(axis=0)
    
    # Create a heatmap visualization
    plt.figure(figsize=(12, 6))
    
    # If feature names are provided, use them
    if feature_names is not None:
        plt.title(f'Attention Weights Across Sequence (Features: {", ".join(feature_names)})')
    else:
        plt.title('Attention Weights Across Sequence')
    
    plt.plot(weights, 'r-', linewidth=2)
    plt.fill_between(np.arange(len(weights)), 0, weights, alpha=0.3, color='red')
    plt.xlabel('Sequence Position')
    plt.ylabel('Attention Weight')
    plt.grid(True)
    
    # Highlight the positions with highest attention
    top_k = 3
    top_indices = np.argsort(weights)[-top_k:]
    for i in top_indices:
        plt.annotate(f'Position {i}', 
                   xy=(i, weights[i]), 
                   xytext=(i, weights[i] + 0.05),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   ha='center')
    
    if save_path:
        attention_path = save_path.replace('.png', '_attention.png')
        plt.savefig(attention_path)
        print(f"Attention visualization saved to: {attention_path}")
    
    plt.show()
    
    # Also create a 2D heatmap if the model's attention is over the entire sequence
    try:
        # This assumes the attention mechanism outputs weights over the entire sequence
        # for each position in the sequence
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights[0].squeeze(), cmap='viridis')
        plt.title('Attention Heatmap')
        plt.xlabel('Sequence Position')
        plt.ylabel('Attention Weight')
        
        if save_path:
            attention_heatmap_path = save_path.replace('.png', '_attention_heatmap.png')
            plt.savefig(attention_heatmap_path)
            print(f"Attention heatmap saved to: {attention_heatmap_path}")
        
        plt.show()
    except:
        print("Could not create 2D attention heatmap.")
    
    return weights


def main():
    # Parameters
    input_video = "/kaggle/working/seconds_14_to_22_frogjump.mp4"
    tracking_data_path = "/kaggle/working/drone_tracking_data_frogjump.csv"
    output_video = "cnn_gru_prediction_comparison_frogjump.mp4"
    error_analysis_path = "error_analysis_frogjump.png"
    
    # CNN-GRU parameters
    seq_length = 15
    batch_size = 32
    epochs = 100
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tracking data
    df = pd.read_csv(tracking_data_path)
    
    # Calculate frames for training/prediction split (first 4 seconds for training)
    fps = 30
    frames_per_segment = int(4 * fps)
    training_end_frame = frames_per_segment
    max_prediction_frames = frames_per_segment  # Predict next 4 seconds
    
    print(f"Total frames: {len(df)}")
    print(f"Training on frames 0-{training_end_frame} (first 4 seconds)")
    print(f"Predicting on frames {training_end_frame+1}-{training_end_frame+max_prediction_frames} (next 4 seconds)")
    
    # Prepare sequences with a smaller stride for more training data
    stride = 2
    X, y, scaler, features = prepare_sequences(df, seq_length=seq_length, stride=stride)
    
    # Split data for training and validation (80/20 of the training data)
    train_indices = []
    val_indices = []
    
    for i in range(0, len(X)):
        end_frame = i * stride + seq_length
        if end_frame < training_end_frame:
            if np.random.random() < 0.8:  # 80% for training
                train_indices.append(i)
            else:  # 20% for validation
                val_indices.append(i)
    
    train_X, train_y = X[train_indices], y[train_indices]
    val_X, val_y = X[val_indices], y[val_indices]
    
    # Create datasets and dataloaders with augmentation
    train_dataset = DronePathDataset(train_X, train_y, augment=True)
    val_dataset = DronePathDataset(val_X, val_y, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create CNN-GRU model
    model = CNNGRUAttention(
        input_size=len(features),
        seq_length=seq_length,
        hidden_size=128,
        num_layers=3,
        cnn_filters=[64, 128],
        dropout=0.3,
        output_size=2
    ).to(device)
    
    # Train model
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        patience=15
    )
    
    # Create ensemble models (optional)
    num_models = 3
    ensemble_models = []
    
    # Train ensemble models with different initializations
    for i in range(num_models):
        print(f"\nTraining ensemble model {i+1}/{num_models}")
        ensemble_model = CNNGRUAttention(
            input_size=len(features),
            seq_length=seq_length,
            hidden_size=128,
            num_layers=3,
            cnn_filters=[64, 128],
            dropout=0.3,
            output_size=2
        ).to(device)
        
        # Train with fewer epochs for ensemble members
        ensemble_model, _, _ = train_model(
            model=ensemble_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs // 2,
            patience=10
        )
        
        ensemble_models.append(ensemble_model)
    
    # Generate predictions with uncertainty
    future_steps = min(max_prediction_frames, len(df) - training_end_frame)
    print(f"Predicting {future_steps} future frames with uncertainty...")
    
    # Choose between single model with MC dropout or ensemble
    use_ensemble = True
    
    if use_ensemble:
        # Ensemble prediction
        predicted_positions, uncertainties = ensemble_prediction(
            models=[trained_model] + ensemble_models,
            df=df,
            scaler=scaler,
            features=features,
            seq_length=seq_length,
            future_steps=future_steps
        )
    else:
        # Single model prediction with Monte Carlo dropout
        predicted_positions, uncertainties = predict_with_uncertainty(
            model=trained_model,
            df=df,
            scaler=scaler,
            features=features,
            seq_length=seq_length,
            future_steps=future_steps,
            num_samples=20
        )
    
    # Calculate prediction errors
    actual_positions = df[['x', 'y']].iloc[training_end_frame:training_end_frame+len(predicted_positions)].values
    
    # Calculate comprehensive metrics and visualize results
    metrics = calculate_comprehensive_metrics(actual_positions, predicted_positions, uncertainties)
    
    # Display summary of key metrics
    print("\n===== PREDICTION ERROR METRICS =====")
    print(f"Mean Distance Error: {metrics['mean_error']:.2f} pixels")
    print(f"Median Distance Error: {metrics['median_error']:.2f} pixels")
    print(f"Maximum Distance Error: {metrics['max_error']:.2f} pixels")
    print(f"RMSE: {metrics['rmse']:.2f} pixels")
    if 'error_growth_rate' in metrics:
        print(f"Error Growth Rate: {metrics['error_growth_rate']:.4f} pixels/frame")
    if 'normalized_error' in metrics:
        print(f"Normalized Error: {metrics['normalized_error']:.4f}")
    if 'mean_angular_error' in metrics:
        print(f"Mean Angular Error: {metrics['mean_angular_error']:.2f} degrees")
    if 'uncertainty_correlation' in metrics:
        print(f"Uncertainty-Error Correlation: {metrics['uncertainty_correlation']:.4f}")
    if 'calibration_score' in metrics:
        print(f"Calibration Score: {metrics['calibration_score']:.4f}")
    print("====================================\n")
    
    # Create detailed visualizations
    visualize_error_analysis(
        actual_positions=actual_positions,
        predicted_positions=predicted_positions,
        uncertainties=uncertainties,
        metrics=metrics,
        df=df,
        training_end_frame=training_end_frame,
        save_path=error_analysis_path
    )
    
    # Visualize attention weights if possible
    if len(X) > 0:
        test_sequence = X[-1]  # Use last training sequence
        try:
            visualize_attention_weights(
                model=trained_model,
                test_sequence=test_sequence,
                feature_names=features,
                save_path=error_analysis_path.replace('.png', '_attention.png')
            )
        except Exception as e:
            print(f"Could not visualize attention weights: {str(e)}")
    
    # Create comparison video with uncertainty visualization
    create_comparison_video(
        input_video=input_video,
        output_video=output_video,
        df=df,
        predicted_positions=predicted_positions,
        uncertainties=uncertainties,
        training_end_frame=training_end_frame,
        seq_length=seq_length
    )
    
    # Save results to CSV for further analysis
    results_df = pd.DataFrame({
        'frame': range(training_end_frame, training_end_frame + len(predicted_positions)),
        'actual_x': actual_positions[:, 0],
        'actual_y': actual_positions[:, 1],
        'predicted_x': predicted_positions[:, 0],
        'predicted_y': predicted_positions[:, 1],
        'error': metrics['euclidean_errors'],
        'uncertainty_x': [u[0] for u in uncertainties],
        'uncertainty_y': [u[1] for u in uncertainties]
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("Prediction results saved to prediction_results.csv")
    
    print("Done!")


if __name__ == "__main__":
    main()