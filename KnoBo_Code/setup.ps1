# Install environment for KnoBo (Windows PowerShell version)
Write-Host "Setting up KnoBo environment..."

# Create conda environment
Write-Host "Creating conda environment..."
conda create --name knowbo python=3.10 -y

# Activate environment and install requirements
Write-Host "Installing requirements..."
conda activate knowbo
pip install -r requirements.txt

# Create data directory if it doesn't exist
if (!(Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

# Download and extract features
Write-Host "Downloading features..."
Invoke-WebRequest -Uri "https://knowledge-bottlenecks.s3.amazonaws.com/features.zip" -OutFile "features.zip"
Expand-Archive -Path "features.zip" -DestinationPath "data" -Force

# Download and extract MIMIC data
Write-Host "Downloading MIMIC-CXR data..."
Invoke-WebRequest -Uri "https://knowledge-bottlenecks.s3.amazonaws.com/MIMIC-CXR.zip" -OutFile "MIMIC-CXR.zip"
if (!(Test-Path "data/datasets")) {
    New-Item -ItemType Directory -Path "data/datasets"
}
Expand-Archive -Path "MIMIC-CXR.zip" -DestinationPath "data/datasets" -Force

# Download and extract ISIC data
Write-Host "Downloading ISIC data..."
Invoke-WebRequest -Uri "https://knowledge-bottlenecks.s3.amazonaws.com/ISIC.zip" -OutFile "ISIC.zip"
Expand-Archive -Path "ISIC.zip" -DestinationPath "data/datasets" -Force

# Download and extract grounding functions
Write-Host "Downloading grounding functions..."
Invoke-WebRequest -Uri "https://knowledge-bottlenecks.s3.amazonaws.com/grounding_functions.zip" -OutFile "grounding_functions.zip"
Expand-Archive -Path "grounding_functions.zip" -DestinationPath "data" -Force

# Download and extract model weights
Write-Host "Downloading model weights..."
Invoke-WebRequest -Uri "https://knowledge-bottlenecks.s3.amazonaws.com/model_weights.zip" -OutFile "model_weights.zip"
Expand-Archive -Path "model_weights.zip" -DestinationPath "data" -Force

# Clean up zip files
Write-Host "Cleaning up temporary files..."
Remove-Item features.zip, MIMIC-CXR.zip, ISIC.zip, grounding_functions.zip, model_weights.zip -Force

Write-Host "Setup complete!"

