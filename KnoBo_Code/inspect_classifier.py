"""
Inspect Grounding Function Classifier
This script loads and visualizes what a grounding function looks like internally
"""

import pickle
import numpy as np
import os

# Find the first available grounding function
grounding_dir = "data/grounding_functions/xray"

# Get first directory
subdirs = [d for d in os.listdir(grounding_dir) if os.path.isdir(os.path.join(grounding_dir, d))]
if not subdirs:
    print("No grounding functions found!")
    exit()

first_concept_dir = subdirs[0]
concept_path = os.path.join(grounding_dir, first_concept_dir)

# Find the .p (pickle) file
pickle_files = [f for f in os.listdir(concept_path) if f.endswith('.p')]
if not pickle_files:
    print(f"No pickle file found in {concept_path}")
    exit()

pickle_file = pickle_files[0]
concept_name = pickle_file.replace('.p', '')

print("=" * 80)
print("INSPECTING GROUNDING FUNCTION CLASSIFIER")
print("=" * 80)
print(f"\nConcept: {concept_name}")
print(f"Location: {concept_path}\n")

# Load the classifier
model_path = os.path.join(concept_path, pickle_file)
with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

print("-" * 80)
print("CLASSIFIER TYPE AND PROPERTIES")
print("-" * 80)
print(f"Model Type: {type(classifier).__name__}")
print(f"Model Class: {classifier.__class__}")

# For Logistic Regression, show detailed properties
if hasattr(classifier, 'coef_'):
    print(f"\n📊 Model Parameters:")
    print(f"   - Coefficients shape: {classifier.coef_.shape}")
    print(f"   - Number of features (input): {classifier.coef_.shape[1]}")
    print(f"   - Number of classes (output): {len(classifier.classes_)}")
    print(f"   - Classes: {classifier.classes_}")
    print(f"   - Intercept: {classifier.intercept_}")
    
    print(f"\n🔢 Weight Vector (first 20 dimensions):")
    weights = classifier.coef_[0]
    for i in range(min(20, len(weights))):
        print(f"   Dimension {i:3d}: {weights[i]:+.6f}")
    print(f"   ... ({len(weights) - 20} more dimensions)")
    
    print(f"\n📈 Weight Statistics:")
    print(f"   - Min weight: {weights.min():.6f}")
    print(f"   - Max weight: {weights.max():.6f}")
    print(f"   - Mean weight: {weights.mean():.6f}")
    print(f"   - Std weight: {weights.std():.6f}")
    print(f"   - Positive weights: {(weights > 0).sum()}")
    print(f"   - Negative weights: {(weights < 0).sum()}")
    print(f"   - Zero weights: {(weights == 0).sum()}")

if hasattr(classifier, 'n_iter_'):
    print(f"\n🔄 Training Info:")
    print(f"   - Iterations: {classifier.n_iter_}")

if hasattr(classifier, 'get_params'):
    print(f"\n⚙️ Model Hyperparameters:")
    params = classifier.get_params()
    for key, value in params.items():
        print(f"   - {key}: {value}")

# Load results file if exists
results_files = [f for f in os.listdir(concept_path) if f.endswith('_results.txt')]
if results_files:
    results_path = os.path.join(concept_path, results_files[0])
    with open(results_path, 'r') as f:
        results = f.read().strip()
    
    print(f"\n✅ Performance Metrics:")
    if ',' in results:
        train_acc, val_acc = results.split(',')
        print(f"   - Training Accuracy: {float(train_acc):.4f} ({float(train_acc)*100:.2f}%)")
        print(f"   - Validation Accuracy: {float(val_acc):.4f} ({float(val_acc)*100:.2f}%)")
    else:
        print(f"   - Accuracy: {results}")

print("\n" + "=" * 80)
print("HOW THIS CLASSIFIER WORKS")
print("=" * 80)
print("""
This is a Logistic Regression classifier that predicts:
   P(concept present | image embedding)

Decision Function:
   z = w₀·x₀ + w₁·x₁ + w₂·x₂ + ... + w₇₆₇·x₇₆₇ + b
   probability = 1 / (1 + e^(-z))

Where:
   - x = image embedding (768 dimensions)
   - w = learned weight vector (768 dimensions)
   - b = intercept/bias term
   - probability > 0.5 → predict YES (concept present)
   - probability < 0.5 → predict NO (concept absent)

Each weight w_i indicates:
   - Positive weight: Higher value in dimension i → concept more likely
   - Negative weight: Higher value in dimension i → concept less likely
   - Large magnitude: Strong indicator
   - Small magnitude: Weak indicator
""")

print("=" * 80)
print("EXAMPLE PREDICTION")
print("=" * 80)

# Create a dummy example
dummy_embedding = np.random.randn(768)
prediction_proba = classifier.predict_proba([dummy_embedding])[0]
prediction_class = classifier.predict([dummy_embedding])[0]

print(f"\nInput: Random 768-dimensional embedding")
print(f"   [0.234, -0.156, 0.872, ..., 0.123]  (768 numbers)")
print(f"\nOutput:")
print(f"   - Probability of 'NO' (class 0): {prediction_proba[0]:.4f}")
print(f"   - Probability of 'YES' (class 1): {prediction_proba[1]:.4f}")
print(f"   - Prediction: {'YES' if prediction_class == 1 else 'NO'}")
print(f"   - Confidence: {max(prediction_proba):.4f} ({max(prediction_proba)*100:.2f}%)")

print("\n" + "=" * 80)
print("FILE STRUCTURE")
print("=" * 80)
print(f"""
{concept_path}/
├── {pickle_file}                      ← Serialized LogisticRegression model
│                                      (Contains weight vector + bias)
│                                      Size: ~7 KB
└── {results_files[0] if results_files else 'results.txt'}  ← Performance metrics
                                       (Train/Val accuracy)
                                       Size: ~20 bytes

The .p file is a Python pickle that stores:
   1. Weight vector (768 floats) ≈ 6 KB
   2. Bias term (1 float) ≈ 8 bytes
   3. sklearn metadata ≈ 1 KB
""")

print("\n" + "=" * 80)
print(f"✅ Inspection Complete!")
print("=" * 80)
