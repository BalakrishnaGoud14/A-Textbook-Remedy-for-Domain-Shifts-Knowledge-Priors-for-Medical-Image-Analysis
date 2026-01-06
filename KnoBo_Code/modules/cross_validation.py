"""
K-Fold Cross-Validation for KnoBo
Provides robust performance estimates with confidence intervals
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
from scipy import stats
from modules.utils import load_features


def cross_validate_cbm(X, y, concepts, num_folds=5, num_epochs=200, 
                       learning_rate=0.001, C=1.0, random_state=42):
    """
    Perform k-fold cross-validation on concept-based model
    
    Args:
        X: (N, feature_dim) - raw features (not used if concepts provided)
        y: (N,) - labels
        concepts: (N, num_concepts) - concept activations
        num_folds: number of folds
        num_epochs: training epochs per fold
        learning_rate: learning rate
        C: regularization parameter
        random_state: random seed
    
    Returns:
        cv_results: dict with CV results
    """
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    fold_accuracies = []
    fold_details = []
    
    print(f"Running {num_folds}-Fold Cross-Validation...")
    print("=" * 60)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(concepts, y), 1):
        print(f"\nFold {fold_idx}/{num_folds}")
        print("-" * 60)
        
        # Split data
        X_train_fold = concepts[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = concepts[val_idx]
        y_val_fold = y[val_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        # Train logistic regression
        model = LogisticRegression(
            C=C,
            max_iter=num_epochs,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=random_state,
            verbose=0
        )
        
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        y_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred) * 100
        
        fold_accuracies.append(accuracy)
        fold_details.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'accuracy': accuracy
        })
        
        print(f"  Train size: {len(train_idx)}")
        print(f"  Val size: {len(val_idx)}")
        print(f"  Accuracy: {accuracy:.2f}%")
    
    # Compute statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    # 95% confidence interval
    confidence_level = 0.95
    degrees_of_freedom = num_folds - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error = t_value * (std_acc / np.sqrt(num_folds))
    ci_lower = mean_acc - margin_of_error
    ci_upper = mean_acc + margin_of_error
    
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"Min: {min(fold_accuracies):.2f}%")
    print(f"Max: {max(fold_accuracies):.2f}%")
    
    cv_results = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'fold_accuracies': fold_accuracies,
        'fold_details': pd.DataFrame(fold_details)
    }
    
    return cv_results


def cross_validate_ood_generalization(X_id, y_id, concepts_id,
                                      X_ood, y_ood, concepts_ood,
                                      num_folds=5, num_epochs=200,
                                      learning_rate=0.001, C=1.0,
                                      random_state=42):
    """
    Cross-validate with separate OOD test set
    
    Args:
        X_id, y_id, concepts_id: in-distribution data
        X_ood, y_ood, concepts_ood: out-of-distribution data
        num_folds: number of folds
        
    Returns:
        cv_results: dict with results including OOD performance
    """
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    print(f"Running {num_folds}-Fold Cross-Validation with OOD Evaluation...")
    print("=" * 60)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(concepts_id, y_id), 1):
        print(f"\nFold {fold_idx}/{num_folds}")
        print("-" * 60)
        
        # Split in-distribution data
        X_train_fold = concepts_id[train_idx]
        y_train_fold = y_id[train_idx]
        X_val_fold = concepts_id[val_idx]
        y_val_fold = y_id[val_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        X_ood_fold = scaler.transform(concepts_ood)
        
        # Train
        model = LogisticRegression(
            C=C,
            max_iter=num_epochs,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=random_state,
            verbose=0
        )
        
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on in-distribution validation
        y_pred_val = model.predict(X_val_fold)
        val_acc = accuracy_score(y_val_fold, y_pred_val) * 100
        
        # Evaluate on OOD test
        y_pred_ood = model.predict(X_ood_fold)
        ood_acc = accuracy_score(y_ood, y_pred_ood) * 100
        
        gap = abs(val_acc - ood_acc)
        avg = (val_acc + ood_acc) / 2
        
        fold_results.append({
            'fold': fold_idx,
            'val_acc': val_acc,
            'ood_acc': ood_acc,
            'gap': gap,
            'average': avg
        })
        
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  OOD Acc: {ood_acc:.2f}%")
        print(f"  Gap: {gap:.2f}%")
    
    # Compute statistics
    df_results = pd.DataFrame(fold_results)
    
    summary = {
        'val_acc': {
            'mean': df_results['val_acc'].mean(),
            'std': df_results['val_acc'].std(),
            'ci_lower': df_results['val_acc'].mean() - 1.96 * df_results['val_acc'].std() / np.sqrt(num_folds),
            'ci_upper': df_results['val_acc'].mean() + 1.96 * df_results['val_acc'].std() / np.sqrt(num_folds)
        },
        'ood_acc': {
            'mean': df_results['ood_acc'].mean(),
            'std': df_results['ood_acc'].std(),
            'ci_lower': df_results['ood_acc'].mean() - 1.96 * df_results['ood_acc'].std() / np.sqrt(num_folds),
            'ci_upper': df_results['ood_acc'].mean() + 1.96 * df_results['ood_acc'].std() / np.sqrt(num_folds)
        },
        'gap': {
            'mean': df_results['gap'].mean(),
            'std': df_results['gap'].std(),
            'ci_lower': df_results['gap'].mean() - 1.96 * df_results['gap'].std() / np.sqrt(num_folds),
            'ci_upper': df_results['gap'].mean() + 1.96 * df_results['gap'].std() / np.sqrt(num_folds)
        }
    }
    
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    print(f"Val Acc: {summary['val_acc']['mean']:.2f}% ± {summary['val_acc']['std']:.2f}%")
    print(f"  95% CI: [{summary['val_acc']['ci_lower']:.2f}%, {summary['val_acc']['ci_upper']:.2f}%]")
    print(f"OOD Acc: {summary['ood_acc']['mean']:.2f}% ± {summary['ood_acc']['std']:.2f}%")
    print(f"  95% CI: [{summary['ood_acc']['ci_lower']:.2f}%, {summary['ood_acc']['ci_upper']:.2f}%]")
    print(f"Gap: {summary['gap']['mean']:.2f}% ± {summary['gap']['std']:.2f}%")
    print(f"  95% CI: [{summary['gap']['ci_lower']:.2f}%, {summary['gap']['ci_upper']:.2f}%]")
    
    cv_results = {
        'summary': summary,
        'fold_results': df_results,
        'num_folds': num_folds
    }
    
    return cv_results


def compare_models_cv(concepts_dict, y_id, y_ood, concepts_ood_dict,
                     model_names, num_folds=5, num_epochs=200):
    """
    Compare multiple models using cross-validation with statistical testing
    
    Args:
        concepts_dict: dict mapping model_name -> in-distribution concepts
        y_id: in-distribution labels
        y_ood: OOD labels
        concepts_ood_dict: dict mapping model_name -> OOD concepts
        model_names: list of model names to compare
        num_folds: number of CV folds
        
    Returns:
        comparison_results: DataFrame with comparison statistics
    """
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'#' * 80}")
        print(f"Evaluating: {model_name}")
        print(f"{'#' * 80}")
        
        cv_results = cross_validate_ood_generalization(
            None, y_id, concepts_dict[model_name],
            None, y_ood, concepts_ood_dict[model_name],
            num_folds=num_folds,
            num_epochs=num_epochs
        )
        
        all_results[model_name] = cv_results
    
    # Create comparison table
    comparison_data = []
    for model_name in model_names:
        summary = all_results[model_name]['summary']
        comparison_data.append({
            'Model': model_name,
            'Val Acc': f"{summary['val_acc']['mean']:.2f} ± {summary['val_acc']['std']:.2f}",
            'OOD Acc': f"{summary['ood_acc']['mean']:.2f} ± {summary['ood_acc']['std']:.2f}",
            'Gap': f"{summary['gap']['mean']:.2f} ± {summary['gap']['std']:.2f}",
            'Val 95% CI': f"[{summary['val_acc']['ci_lower']:.2f}, {summary['val_acc']['ci_upper']:.2f}]",
            'OOD 95% CI': f"[{summary['ood_acc']['ci_lower']:.2f}, {summary['ood_acc']['ci_upper']:.2f}]"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    
    return {
        'comparison_table': comparison_df,
        'detailed_results': all_results
    }


if __name__ == "__main__":
    print("Testing Cross-Validation Module...")
    
    # Dummy data
    np.random.seed(42)
    N_id = 1000
    N_ood = 200
    num_concepts = 50
    num_classes = 3
    
    X_id = np.random.randn(N_id, 768)
    concepts_id = np.random.rand(N_id, num_concepts)
    y_id = np.random.randint(0, num_classes, N_id)
    
    X_ood = np.random.randn(N_ood, 768)
    concepts_ood = np.random.rand(N_ood, num_concepts)
    y_ood = np.random.randint(0, num_classes, N_ood)
    
    # Test basic CV
    print("\n" + "#" * 80)
    print("Test 1: Basic Cross-Validation")
    print("#" * 80)
    cv_results = cross_validate_cbm(X_id, y_id, concepts_id, num_folds=5)
    
    # Test OOD CV
    print("\n" + "#" * 80)
    print("Test 2: Cross-Validation with OOD Evaluation")
    print("#" * 80)
    cv_ood_results = cross_validate_ood_generalization(
        X_id, y_id, concepts_id,
        X_ood, y_ood, concepts_ood,
        num_folds=5
    )
    
    print("\n✓ Cross-validation module tested successfully!")
