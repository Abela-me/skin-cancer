# Skin Cancer Detection and Classification

A deep learning-based system for skin cancer detection and classification using Convolutional Neural Networks (CNN) and Probabilistic Graphical Models (PGM) with Bayesian Networks to model variable dependencies.

## Features

- **Deep Learning Classification**: CNN-based skin cancer detection
- **Bayesian Network Analysis**: PGM to model relationships between clinical variables
- **Image Processing**: Automated preprocessing and augmentation
- **Multi-class Classification**: Melanoma, Carcinoma, and Benign lesions

## Tech Stack

- **Deep Learning**: TensorFlow/Keras
- **Probabilistic Models**: pgmpy (Bayesian Networks)
- **Image Processing**: OpenCV, Pillow
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abela-me/skin-cancer.git
cd skin-cancer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
skin-cancer/
├── src/
│   ├── __init__.py
│   ├── cnn_model.py          # CNN classifier implementation
│   ├── pgm_model.py          # Bayesian network model
│   ├── data_loader.py        # Data loading and preprocessing
│   └── utils.py              # Utility functions
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Preprocessed images
├── models/                   # Trained model checkpoints
├── notebooks/
│   └── exploration.ipynb     # EDA and experiments
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

## Usage

### Training the CNN Model
```python
from src.cnn_model import SkinCancerCNN
from src.data_loader import load_and_preprocess_data

# Load data
X_train, y_train = load_and_preprocess_data('data/raw/')

# Create and train model
model = SkinCancerCNN()
model.train(X_train, y_train)
```

### Bayesian Network Analysis
```python
from src.pgm_model import SkinCancerBayesianNetwork

# Create network
bn = SkinCancerBayesianNetwork()
bn.add_variables(['age', 'skin_type', 'family_history', 'diagnosis'])
bn.fit(clinical_data)
```

## Dataset

The model is trained on skin lesion images. Recommended datasets:
- [ISIC Archive](https://www.isic-archive.com/)
- [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or collaborations, reach out via GitHub Issues.
