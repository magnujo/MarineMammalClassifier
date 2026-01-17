# MarineMammalClassifier

MarineMammalClassifier is a machine learning project developed as part of a Master’s thesis in Computer Science at Roskilde University. The project focuses on classifying marine mammals from bioacoustic recordings using convolutional neural networks (CNNs).

## Overview

The repository contains code for preprocessing underwater audio data, training deep learning models, and evaluating their performance on marine mammal classification tasks. The project is structured to support experimentation and reproducibility using a Conda-managed environment.

## Features

- Preprocessing of bioacoustic data (e.g. spectrogram generation)
- Convolutional neural network (CNN) models for classification
- Jupyter notebooks for exploration and visualization
- Modular Python codebase for experiments and reuse

## Repository Structure

```
.
├── MLModels/          # Neural network model definitions
├── Notebooks/         # Jupyter notebooks for experiments and analysis
├── Preprocessing/     # Audio preprocessing and feature extraction
├── Results/           # Experimental results and figures
├── Tests/             # Unit tests
├── Utils/             # Utility functions
├── main.py            # Main entry point for training and evaluation
├── environment.yaml   # Conda environment specification
└── README.md
```

## Installation

The project uses Conda for dependency management.

1. Clone the repository:
   ```bash
   git clone https://github.com/magnujo/MarineMammalClassifier.git
   cd MarineMammalClassifier
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate marine-mammal-classifier
   ```

## Usage

The main entry point for running experiments is `main.py`.

### Training

```bash
python main.py --mode train
```

### Evaluation

```bash
python main.py --mode evaluate --model path/to/model_checkpoint
```

Exact arguments may vary depending on your configuration and dataset.

## Results

Model outputs, plots, and evaluation metrics should be stored in the `Results/` directory. Additional analysis and visualizations are available in the notebooks under `Notebooks/`.

## Contributing

Contributions are welcome for research and educational purposes. Please fork the repository and submit a pull request with a clear description of your changes.

## License

This project is intended for academic and research use. Add a `LICENSE` file if you wish to specify formal licensing terms.
