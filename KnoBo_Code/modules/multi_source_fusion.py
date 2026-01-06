"""
Multi-Source Knowledge Fusion for KnoBo
Combines concept predictions from multiple medical knowledge sources
(PubMed, StatPearls, Wikipedia, Textbooks)
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from modules.utils import load_features
import os


KNOWLEDGE_SOURCES = ['PubMed', 'StatPearls', 'Wikipedia', 'Textbooks']


def load_concept_classifiers(modality, source, num_features):
    """
    Load pre-trained concept classifiers from a knowledge source
    
    Args:
        modality: 'xray' or 'skin'
        source: knowledge source name
        num_features: number of concepts
    
    Returns:
        classifier_list: list of trained concept classifiers
    """
    classifier_dir = f"./data/grounding_functions/{modality}/{modality}_binary_classifiers_{num_features}_{source}"
    
    if not os.path.exists(classifier_dir):
        print(f"Warning: Classifiers not found at {classifier_dir}")
        return None
    
    classifier_list = []
    for i in range(num_features):
        classifier_path = os.path.join(classifier_dir, f"binary_classifier_{i}.pkl")
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
                classifier_list.append(classifier)
        else:
            print(f"Warning: Classifier {i} not found for {source}")
            return None
    
    return classifier_list


def extract_concepts_from_source(X, classifier_list, normalize=True):
    """
    Extract concept activations using classifiers from a source
    
    Args:
        X: (N, feature_dim) - input features
        classifier_list: list of concept classifiers
        normalize: whether to normalize input
    
    Returns:
        concepts: (N, num_concepts) - concept activations
    """
    if classifier_list is None:
        return None
    
    concepts = []
    
    for classifier in classifier_list:
        # Get probability of positive class
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            probs = classifier.predict_proba(X_scaled)[:, 1]
        else:
            probs = classifier.predict_proba(X)[:, 1]
        concepts.append(probs)
    
    return np.column_stack(concepts)


def multi_source_fusion(concept_dict, fusion_method='average', weights=None):
    """
    Fuse concept predictions from multiple sources
    
    Args:
        concept_dict: dict mapping source_name -> concepts array (N, num_concepts)
        fusion_method: 'average', 'weighted', 'max', 'voting', 'learned'
        weights: dict mapping source_name -> weight (for weighted fusion)
    
    Returns:
        fused_concepts: (N, num_concepts) - fused concept activations
    """
    
    # Filter out None values
    valid_sources = {k: v for k, v in concept_dict.items() if v is not None}
    
    if len(valid_sources) == 0:
        raise ValueError("No valid concept sources available")
    
    if len(valid_sources) == 1:
        return list(valid_sources.values())[0]
    
    # Stack concepts from all sources
    concept_stack = np.array(list(valid_sources.values()))  # (num_sources, N, num_concepts)
    
    if fusion_method == 'average':
        # Simple average
        fused_concepts = concept_stack.mean(axis=0)
    
    elif fusion_method == 'weighted':
        # Weighted average
        if weights is None:
            weights = {source: 1.0 / len(valid_sources) for source in valid_sources}
        
        weight_array = np.array([weights.get(source, 1.0) for source in valid_sources.keys()])
        weight_array = weight_array / weight_array.sum()  # Normalize
        weight_array = weight_array[:, np.newaxis, np.newaxis]  # (num_sources, 1, 1)
        
        fused_concepts = (concept_stack * weight_array).sum(axis=0)
    
    elif fusion_method == 'max':
        # Max pooling across sources
        fused_concepts = concept_stack.max(axis=0)
    
    elif fusion_method == 'voting':
        # Majority voting (threshold at 0.5)
        binary_concepts = (concept_stack > 0.5).astype(float)
        fused_concepts = binary_concepts.mean(axis=0)
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    return fused_concepts


def load_and_fuse_concepts(dataset_name, modality, model_name, num_features,
                           sources=KNOWLEDGE_SOURCES, fusion_method='average',
                           weights=None, shots='all', normalize=True):
    """
    Load features, extract concepts from multiple sources, and fuse them
    
    Args:
        dataset_name: name of dataset
        modality: 'xray' or 'skin'
        model_name: name of vision model
        num_features: number of concepts
        sources: list of knowledge sources to use
        fusion_method: how to fuse concepts
        weights: weights for weighted fusion
        shots: number of shots ('all' or int)
        normalize: normalize features
    
    Returns:
        fused_train: fused training concepts
        fused_val: fused validation concepts  
        fused_ood: fused OOD concepts
        y_train, y_val, y_ood: labels
        available_sources: list of sources that were successfully loaded
    """
    
    # Load features
    label2index = torch.load(f"./data/features/{model_name}/{dataset_name}_label.pt")
    X_train, y_train, X_val, y_val, X_ood, y_ood = load_features(
        f"./data/features/{model_name}/{dataset_name}",
        label2index, shots, normalize
    )
    
    # Extract concepts from each source
    concept_dict_train = {}
    concept_dict_val = {}
    concept_dict_ood = {}
    available_sources = []
    
    for source in sources:
        print(f"Loading {source} classifiers...")
        classifier_list = load_concept_classifiers(modality, source, num_features)
        
        if classifier_list is not None:
            print(f"Extracting concepts from {source}...")
            train_concepts = extract_concepts_from_source(X_train, classifier_list, normalize)
            val_concepts = extract_concepts_from_source(X_val, classifier_list, normalize)
            ood_concepts = extract_concepts_from_source(X_ood, classifier_list, normalize)
            
            if train_concepts is not None:
                concept_dict_train[source] = train_concepts
                concept_dict_val[source] = val_concepts
                concept_dict_ood[source] = ood_concepts
                available_sources.append(source)
                print(f"✓ {source} concepts extracted successfully")
        else:
            print(f"✗ Skipping {source} - classifiers not available")
    
    if len(available_sources) == 0:
        raise ValueError("No concept sources available!")
    
    print(f"\nFusing concepts from {len(available_sources)} sources using {fusion_method} method...")
    
    # Fuse concepts
    fused_train = multi_source_fusion(concept_dict_train, fusion_method, weights)
    fused_val = multi_source_fusion(concept_dict_val, fusion_method, weights)
    fused_ood = multi_source_fusion(concept_dict_ood, fusion_method, weights)
    
    return fused_train, fused_val, fused_ood, y_train, y_val, y_ood, available_sources


def train_on_fused_concepts(fused_train, y_train, fused_val, y_val, fused_ood, y_ood,
                            num_epochs=200, learning_rate=0.001, batch_size=64):
    """
    Train classifier on fused concepts
    
    Returns:
        val_acc, ood_acc, gap, average_acc
    """
    from modules.cbm import get_results
    
    # Convert to pandas DataFrames (expected by get_results)
    df_train = pd.DataFrame(fused_train)
    df_val = pd.DataFrame(fused_val)
    df_ood = pd.DataFrame(fused_ood)
    
    # Get unique labels for label mapping
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_ood]))
    label2index = {label: idx for idx, label in enumerate(unique_labels)}
    
    val_acc, ood_acc, gap, average_acc, _ = get_results(
        None,  # args not needed
        label2index,
        None,  # classifier_list not needed for fused concepts
        df_train, y_train,
        df_val, y_val,
        df_ood, y_ood,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    return val_acc, ood_acc, gap, average_acc


if __name__ == "__main__":
    print("Testing Multi-Source Concept Fusion...")
    
    # Example: Try to fuse concepts for a dataset
    dataset_name = "COVID-QU"
    modality = "xray"
    model_name = "whyxrayclip"
    num_features = 150
    
    try:
        fused_train, fused_val, fused_ood, y_train, y_val, y_ood, sources = load_and_fuse_concepts(
            dataset_name=dataset_name,
            modality=modality,
            model_name=model_name,
            num_features=num_features,
            sources=['PubMed', 'StatPearls', 'Wikipedia', 'Textbooks'],
            fusion_method='average'
        )
        
        print(f"\n✓ Successfully fused concepts from sources: {sources}")
        print(f"  Training concepts shape: {fused_train.shape}")
        print(f"  Validation concepts shape: {fused_val.shape}")
        print(f"  OOD concepts shape: {fused_ood.shape}")
        
        # Train and evaluate
        print("\nTraining on fused concepts...")
        val_acc, ood_acc, gap, avg = train_on_fused_concepts(
            fused_train, y_train, fused_val, y_val, fused_ood, y_ood
        )
        
        print(f"\nResults with Multi-Source Fusion:")
        print(f"  Ind Acc: {val_acc:.2f}%")
        print(f"  OOD Acc: {ood_acc:.2f}%")
        print(f"  Gap: {gap:.2f}%")
        print(f"  Average: {avg:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Multi-source fusion requires concept classifiers")
        print("from multiple knowledge sources to be available in:")
        print(f"  data/grounding_functions/{modality}/")
