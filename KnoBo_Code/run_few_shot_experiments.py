"""
Few-Shot Learning Experiments for KnoBo
Evaluates model performance with limited training samples (4, 8, 16-shot)
"""

import subprocess
import pandas as pd
import os

# Shot configurations to test
SHOT_CONFIGS = [4, 8, 16]
MODALITIES = ['xray', 'skin']
MODEL_NAMES = {
    'xray': 'whyxrayclip',
    'skin': 'whylesionclip'
}
NUM_FEATURES = {
    'xray': 150,
    'skin': 5
}

def run_few_shot_experiment(modality, shots):
    """Run KnoBo experiment with specified number of shots"""
    model_name = MODEL_NAMES[modality]
    num_features = NUM_FEATURES[modality]
    
    print(f"\n{'='*60}")
    print(f"Running {modality} - {shots}-shot experiment")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', 'modules/cbm.py',
        '--mode', 'binary',
        '--bottleneck', 'PubMed',
        '--number_of_features', str(num_features),
        '--add_prior', 'True',
        '--modality', modality,
        '--model_name', model_name,
        '--shots', str(shots)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {modality} {shots}-shot: {e}")
        print(e.stderr)
        return False

def main():
    """Run all few-shot experiments"""
    print("=" * 80)
    print("FEW-SHOT LEARNING EXPERIMENTS FOR KNOBO")
    print("=" * 80)
    
    results_summary = []
    
    for modality in MODALITIES:
        for shots in SHOT_CONFIGS:
            success = run_few_shot_experiment(modality, shots)
            results_summary.append({
                'modality': modality,
                'shots': shots,
                'status': 'Success' if success else 'Failed'
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))
    
    print("\nResults saved in data/results/ directory")
    print("Look for files matching pattern: *_{modality}_*_{shots}_*")

if __name__ == "__main__":
    main()
