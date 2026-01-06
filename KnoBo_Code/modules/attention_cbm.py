"""
Concept Bottleneck Model with Attention Mechanism
Learns to weight medical concepts by importance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ConceptAttentionLayer(nn.Module):
    """Learnable attention over concept activations"""
    
    def __init__(self, num_concepts, hidden_dim=64):
        super(ConceptAttentionLayer, self).__init__()
        self.num_concepts = num_concepts
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_concepts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, concepts):
        """
        Args:
            concepts: (batch_size, num_concepts) - concept activations
        Returns:
            weighted_concepts: (batch_size, num_concepts) - attention-weighted concepts
            attention_weights: (batch_size, num_concepts) - attention scores
        """
        attention_weights = self.attention(concepts)
        weighted_concepts = concepts * attention_weights
        return weighted_concepts, attention_weights


class AttentionCBM(nn.Module):
    """Concept Bottleneck Model with Attention"""
    
    def __init__(self, num_concepts, num_classes, hidden_dim=64):
        super(AttentionCBM, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # Attention layer
        self.attention = ConceptAttentionLayer(num_concepts, hidden_dim)
        
        # Classifier on weighted concepts
        self.classifier = nn.Linear(num_concepts, num_classes)
        
    def forward(self, concepts):
        """
        Args:
            concepts: (batch_size, num_concepts)
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, num_concepts)
        """
        weighted_concepts, attention_weights = self.attention(concepts)
        logits = self.classifier(weighted_concepts)
        return logits, attention_weights


class ConceptDataset(Dataset):
    """Dataset for concept-based learning"""
    
    def __init__(self, concepts, labels):
        self.concepts = torch.FloatTensor(concepts)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.concepts[idx], self.labels[idx]


def train_attention_cbm(X_train, y_train, X_val, y_val, num_classes, 
                       num_epochs=100, batch_size=64, learning_rate=0.001,
                       hidden_dim=64, device='cpu'):
    """
    Train CBM with attention mechanism
    
    Args:
        X_train: (N, num_concepts) - training concept activations
        y_train: (N,) - training labels
        X_val: (M, num_concepts) - validation concept activations
        y_val: (M,) - validation labels
        num_classes: number of output classes
        num_epochs: training epochs
        batch_size: batch size
        learning_rate: learning rate
        hidden_dim: hidden dimension for attention network
        device: 'cpu' or 'cuda'
    
    Returns:
        model: trained AttentionCBM
        best_val_acc: best validation accuracy
        attention_weights: learned attention weights on validation set
    """
    
    # Create datasets
    train_dataset = ConceptDataset(X_train, y_train)
    val_dataset = ConceptDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_concepts = X_train.shape[1]
    model = AttentionCBM(num_concepts, num_classes, hidden_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
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
            logits, _ = model(concepts)
            loss = criterion(logits, labels)
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
                logits, _ = model(concepts)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds) * 100
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Get attention weights on validation set
    model.eval()
    all_attention_weights = []
    with torch.no_grad():
        for concepts, _ in val_loader:
            concepts = concepts.to(device)
            _, attention_weights = model(concepts)
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    attention_weights = np.concatenate(all_attention_weights, axis=0)
    
    return model, best_val_acc, attention_weights


def evaluate_attention_cbm(model, X_test, y_test, batch_size=64, device='cpu'):
    """
    Evaluate trained attention CBM
    
    Returns:
        accuracy: test accuracy
        attention_weights: attention weights on test set
        predictions: model predictions
    """
    test_dataset = ConceptDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_true = []
    all_attention_weights = []
    
    with torch.no_grad():
        for concepts, labels in test_loader:
            concepts, labels = concepts.to(device), labels.to(device)
            logits, attention_weights = model(concepts)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    accuracy = accuracy_score(all_true, all_preds) * 100
    attention_weights = np.concatenate(all_attention_weights, axis=0)
    
    return accuracy, attention_weights, np.array(all_preds)


def get_top_concepts(attention_weights, concept_names=None, top_k=10):
    """
    Get top-k most important concepts based on average attention weights
    
    Args:
        attention_weights: (N, num_concepts) - attention weights
        concept_names: list of concept names (optional)
        top_k: number of top concepts to return
    
    Returns:
        top_concepts: list of (concept_idx, avg_attention) or (concept_name, avg_attention)
    """
    avg_attention = attention_weights.mean(axis=0)
    top_indices = np.argsort(avg_attention)[::-1][:top_k]
    
    if concept_names is not None:
        top_concepts = [(concept_names[idx], avg_attention[idx]) for idx in top_indices]
    else:
        top_concepts = [(idx, avg_attention[idx]) for idx in top_indices]
    
    return top_concepts


if __name__ == "__main__":
    # Example usage
    print("Testing Attention CBM...")
    
    # Dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)
    X_test = np.random.randn(200, 50)
    y_test = np.random.randint(0, 3, 200)
    
    # Train
    model, val_acc, val_attention = train_attention_cbm(
        X_train, y_train, X_val, y_val, 
        num_classes=3, num_epochs=50
    )
    
    print(f"\nBest Validation Accuracy: {val_acc:.2f}%")
    
    # Evaluate
    test_acc, test_attention, preds = evaluate_attention_cbm(model, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Get top concepts
    top_concepts = get_top_concepts(test_attention, top_k=10)
    print("\nTop 10 Most Important Concepts:")
    for idx, (concept_idx, weight) in enumerate(top_concepts, 1):
        print(f"{idx}. Concept {concept_idx}: {weight:.4f}")
