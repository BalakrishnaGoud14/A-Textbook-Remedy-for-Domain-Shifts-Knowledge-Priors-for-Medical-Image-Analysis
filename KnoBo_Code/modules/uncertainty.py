"""
Uncertainty Quantification for Concept Bottleneck Models
Implements Monte Carlo Dropout and prediction confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class MCDropoutCBM(nn.Module):
    """CBM with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64], dropout_rate=0.3):
        super(MCDropoutCBM, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        return self.network(x)
    
    def enable_dropout(self):
        """Enable dropout at test time for MC sampling"""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_mc_dropout_cbm(X_train, y_train, X_val, y_val, num_classes,
                        num_epochs=100, batch_size=64, learning_rate=0.001,
                        hidden_dims=[128, 64], dropout_rate=0.3, device='cpu'):
    """
    Train CBM with dropout for uncertainty estimation
    
    Returns:
        model: trained MCDropoutCBM
        best_val_acc: best validation accuracy
    """
    
    # Create datasets
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MCDropoutCBM(X_train.shape[1], num_classes, hidden_dims, dropout_rate).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for concepts, labels in train_loader:
            concepts, labels = concepts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(concepts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for concepts, labels in val_loader:
                concepts, labels = concepts.to(device), labels.to(device)
                outputs = model(concepts)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds) * 100
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, best_val_acc


def mc_dropout_predict(model, X, num_samples=30, batch_size=64, device='cpu'):
    """
    Make predictions with uncertainty estimation using Monte Carlo Dropout
    
    Args:
        model: trained MCDropoutCBM
        X: input concepts (N, num_concepts)
        num_samples: number of MC dropout samples
        batch_size: batch size
        device: computation device
    
    Returns:
        mean_probs: (N, num_classes) - mean predicted probabilities
        std_probs: (N, num_classes) - std of predicted probabilities  
        predictions: (N,) - final predictions (argmax of mean_probs)
        uncertainty: (N,) - prediction uncertainty (entropy or std)
    """
    
    dataset = TensorDataset(torch.FloatTensor(X))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    model.enable_dropout()  # Enable dropout for MC sampling
    
    all_predictions = []
    
    # MC sampling
    for _ in range(num_samples):
        batch_predictions = []
        with torch.no_grad():
            for (concepts,) in dataloader:
                concepts = concepts.to(device)
                outputs = model(concepts)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu().numpy())
        all_predictions.append(np.concatenate(batch_predictions, axis=0))
    
    # Stack predictions: (num_samples, N, num_classes)
    all_predictions = np.array(all_predictions)
    
    # Compute statistics
    mean_probs = all_predictions.mean(axis=0)  # (N, num_classes)
    std_probs = all_predictions.std(axis=0)    # (N, num_classes)
    
    # Final predictions
    predictions = np.argmax(mean_probs, axis=1)
    
    # Uncertainty metrics
    # 1. Predictive entropy
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)
    
    # 2. Standard deviation of predicted class
    predicted_class_std = std_probs[np.arange(len(predictions)), predictions]
    
    # 3. Variation ratio (1 - max_prob)
    variation_ratio = 1 - mean_probs.max(axis=1)
    
    uncertainty = {
        'entropy': entropy,
        'predicted_std': predicted_class_std,
        'variation_ratio': variation_ratio,
        'mean_std': std_probs.mean(axis=1)
    }
    
    return mean_probs, std_probs, predictions, uncertainty


def calibration_metrics(y_true, mean_probs, predictions):
    """
    Compute calibration metrics
    
    Returns:
        expected_calibration_error: ECE
        max_calibration_error: MCE
        accuracy: accuracy
        confidence: mean confidence
    """
    
    max_probs = mean_probs.max(axis=1)
    accuracy = accuracy_score(y_true, predictions)
    confidence = max_probs.mean()
    
    # ECE calculation
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(max_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = (predictions[mask] == y_true[mask]).mean()
            bin_confidence = max_probs[mask].mean()
            bin_error = abs(bin_accuracy - bin_confidence)
            
            ece += (mask.sum() / len(y_true)) * bin_error
            mce = max(mce, bin_error)
    
    return {
        'ece': ece,
        'mce': mce,
        'accuracy': accuracy,
        'confidence': confidence
    }


def identify_uncertain_samples(uncertainty, threshold_percentile=90):
    """
    Identify highly uncertain samples for human review
    
    Args:
        uncertainty: dict with uncertainty metrics
        threshold_percentile: percentile threshold for flagging uncertain samples
    
    Returns:
        uncertain_indices: indices of uncertain samples
        uncertainty_scores: combined uncertainty scores
    """
    
    # Combine multiple uncertainty metrics (normalized)
    entropy_norm = (uncertainty['entropy'] - uncertainty['entropy'].min()) / \
                   (uncertainty['entropy'].max() - uncertainty['entropy'].min() + 1e-10)
    
    variation_norm = uncertainty['variation_ratio']
    
    # Combined score
    uncertainty_scores = (entropy_norm + variation_norm) / 2
    
    # Threshold
    threshold = np.percentile(uncertainty_scores, threshold_percentile)
    uncertain_indices = np.where(uncertainty_scores >= threshold)[0]
    
    return uncertain_indices, uncertainty_scores


if __name__ == "__main__":
    print("Testing MC Dropout Uncertainty Quantification...")
    
    # Dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)
    X_test = np.random.randn(200, 50)
    y_test = np.random.randint(0, 3, 200)
    
    # Train
    print("\nTraining MCDropout CBM...")
    model, val_acc = train_mc_dropout_cbm(
        X_train, y_train, X_val, y_val,
        num_classes=3, num_epochs=50
    )
    print(f"\nBest Validation Accuracy: {val_acc:.2f}%")
    
    # Predict with uncertainty
    print("\nMaking predictions with uncertainty estimation...")
    mean_probs, std_probs, predictions, uncertainty = mc_dropout_predict(
        model, X_test, num_samples=30
    )
    
    # Calibration
    calib = calibration_metrics(y_test, mean_probs, predictions)
    print(f"\nCalibration Metrics:")
    print(f"  Accuracy: {calib['accuracy']*100:.2f}%")
    print(f"  Confidence: {calib['confidence']:.4f}")
    print(f"  ECE: {calib['ece']:.4f}")
    print(f"  MCE: {calib['mce']:.4f}")
    
    # Identify uncertain samples
    uncertain_idx, unc_scores = identify_uncertain_samples(uncertainty, threshold_percentile=90)
    print(f"\nIdentified {len(uncertain_idx)} highly uncertain samples (top 10%)")
    print(f"These samples should be reviewed by human experts")
    
    # Show examples
    print("\nTop 5 Most Uncertain Predictions:")
    top_uncertain = np.argsort(unc_scores)[::-1][:5]
    for i, idx in enumerate(top_uncertain, 1):
        print(f"{i}. Sample {idx}:")
        print(f"   Predicted: {predictions[idx]}, True: {y_test[idx]}")
        print(f"   Confidence: {mean_probs[idx].max():.4f}")
        print(f"   Uncertainty Score: {unc_scores[idx]:.4f}")
        print(f"   Entropy: {uncertainty['entropy'][idx]:.4f}")
