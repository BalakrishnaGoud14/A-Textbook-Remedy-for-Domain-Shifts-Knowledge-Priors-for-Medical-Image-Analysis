"""
Hyperparameter Optimization for KnoBo using Optuna
Optimizes learning rate, regularization, and other parameters
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
from modules.utils import load_features
import pickle
import os


def objective_cbm(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for CBM hyperparameter optimization
    
    Optimizes:
    - C: regularization parameter
    - solver: optimization algorithm
    - penalty: regularization type
    
    Args:
        trial: Optuna trial object
        X_train, y_train: training concepts and labels
        X_val, y_val: validation concepts and labels
    
    Returns:
        validation_accuracy: accuracy to maximize
    """
    
    # Hyperparameters to optimize
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear'])
    
    if solver == 'liblinear':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    elif solver == 'saga':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none'])
    else:  # lbfgs
        penalty = 'l2'
    
    # L1 ratio for elasticnet
    l1_ratio = None
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    try:
        model = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty,
            l1_ratio=l1_ratio,
            max_iter=200,
            multi_class='multinomial',
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
        
    except Exception as e:
        # Invalid hyperparameter combination
        return 0.0


def optimize_cbm_hyperparameters(X_train, y_train, X_val, y_val, 
                                 n_trials=100, timeout=3600):
    """
    Run hyperparameter optimization for CBM
    
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        n_trials: number of Optuna trials
        timeout: optimization timeout in seconds
    
    Returns:
        best_params: dict of best hyperparameters
        study: Optuna study object
    """
    
    print("Starting Hyperparameter Optimization...")
    print("=" * 60)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective_cbm(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Show top 5 trials
    print(f"\nTop 5 Trials:")
    df_trials = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(df_trials[['number', 'value', 'params_C', 'params_solver', 'params_penalty']].to_string(index=False))
    
    return study.best_params, study


def optimize_dataset(dataset_name, modality, model_name, num_features,
                    bottleneck='PubMed', n_trials=50):
    """
    Optimize hyperparameters for a specific dataset
    
    Args:
        dataset_name: name of dataset
        modality: 'xray' or 'skin'
        model_name: vision model name
        num_features: number of concepts
        bottleneck: knowledge source
        n_trials: number of optimization trials
    
    Returns:
        best_params: optimized hyperparameters
        results: performance with best params
    """
    
    print(f"\n{'#' * 80}")
    print(f"Optimizing Hyperparameters for: {dataset_name}")
    print(f"{'#' * 80}\n")
    
    # Load features
    label2index = torch.load(f"./data/features/{model_name}/{dataset_name}_label.pt")
    X_train, y_train, X_val, y_val, X_ood, y_ood = load_features(
        f"./data/features/{model_name}/{dataset_name}",
        label2index, 'all', True
    )
    
    # Load concept classifiers
    classifier_dir = f"./data/grounding_functions/{modality}/{modality}_binary_classifiers_{num_features}_{bottleneck}"
    
    classifier_list = []
    for i in range(num_features):
        classifier_path = os.path.join(classifier_dir, f"binary_classifier_{i}.pkl")
        with open(classifier_path, 'rb') as f:
            classifier_list.append(pickle.load(f))
    
    # Extract concepts
    train_concepts = []
    val_concepts = []
    ood_concepts = []
    
    for classifier in classifier_list:
        train_concepts.append(classifier.predict_proba(X_train)[:, 1])
        val_concepts.append(classifier.predict_proba(X_val)[:, 1])
        ood_concepts.append(classifier.predict_proba(X_ood)[:, 1])
    
    train_concepts = np.column_stack(train_concepts)
    val_concepts = np.column_stack(val_concepts)
    ood_concepts = np.column_stack(ood_concepts)
    
    # Optimize
    best_params, study = optimize_cbm_hyperparameters(
        train_concepts, y_train,
        val_concepts, y_val,
        n_trials=n_trials
    )
    
    # Train final model with best params
    print("\nTraining final model with best parameters...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_concepts)
    val_scaled = scaler.transform(val_concepts)
    ood_scaled = scaler.transform(ood_concepts)
    
    final_model = LogisticRegression(
        **best_params,
        max_iter=200,
        multi_class='multinomial',
        random_state=42
    )
    final_model.fit(train_scaled, y_train)
    
    # Evaluate
    val_acc = accuracy_score(y_val, final_model.predict(val_scaled)) * 100
    ood_acc = accuracy_score(y_ood, final_model.predict(ood_scaled)) * 100
    gap = abs(val_acc - ood_acc)
    avg = (val_acc + ood_acc) / 2
    
    results = {
        'dataset': dataset_name,
        'val_acc': val_acc,
        'ood_acc': ood_acc,
        'gap': gap,
        'average': avg,
        'best_params': best_params
    }
    
    print(f"\nFinal Results:")
    print(f"  Val Acc: {val_acc:.2f}%")
    print(f"  OOD Acc: {ood_acc:.2f}%")
    print(f"  Gap: {gap:.2f}%")
    print(f"  Average: {avg:.2f}%")
    
    return best_params, results, study


def save_optimization_results(results_list, filename='hyperparameter_optimization_results.csv'):
    """Save optimization results to CSV"""
    df = pd.DataFrame(results_list)
    df.to_csv(f"./data/results/{filename}", index=False)
    print(f"\nResults saved to: ./data/results/{filename}")


if __name__ == "__main__":
    print("Hyperparameter Optimization Module")
    print("=" * 80)
    print("\nNote: Install optuna with: pip install optuna")
    print("\nThis module will optimize:")
    print("  - C (regularization strength)")
    print("  - solver (optimization algorithm)")
    print("  - penalty (regularization type)")
    print("  - l1_ratio (for elasticnet penalty)")
    print("\nExample usage:")
    print("""
    from modules.hyperparameter_optimization import optimize_dataset
    
    best_params, results, study = optimize_dataset(
        dataset_name='COVID-QU',
        modality='xray',
        model_name='whyxrayclip',
        num_features=150,
        bottleneck='PubMed',
        n_trials=50
    )
    """)
