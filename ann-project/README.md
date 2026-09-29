# 🧠 ANN Project — Neural Network From Scratch (NumPy)

A complete, GitHub-ready Artificial Neural Network project. The network is built **from scratch with NumPy**
(forward pass, back-propagation, Adam/SGD, dropout, early stopping) and trained on the
Breast Cancer Wisconsin dataset (binary classification).

## Project structure

```
ann-project/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── setup.py
├── configs/
│   └── config.yaml          # hyper-parameters
├── data/
│   ├── raw/                 # original data (git-ignored)
│   └── processed/           # train/test splits
├── notebooks/
│   └── ANN_Project.ipynb    # full walkthrough
├── src/
│   ├── __init__.py
│   ├── config.py            # config loader
│   ├── data.py              # loading + preprocessing
│   ├── layers.py            # Dense layer, activations
│   ├── model.py             # ANN class (forward/backward/fit/predict)
│   ├── optimizers.py        # SGD, Adam
│   ├── metrics.py           # accuracy, precision, recall, F1, confusion matrix
│   ├── train.py             # training entry point
│   ├── evaluate.py          # evaluation entry point
│   └── utils.py             # seeding, plotting, saving
├── models/                  # saved weights (.npz)
├── reports/
│   └── figures/             # loss curves, confusion matrix
└── tests/
    ├── test_layers.py
    └── test_model.py
```

## Quick start

```bash
git clone https://github.com/<your-username>/ann-project.git
cd ann-project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train        # train the model
python -m src.evaluate     # evaluate on the test set
pytest -q                  # run tests
jupyter notebook notebooks/ANN_Project.ipynb
```

## Model

| Item | Value |
|------|-------|
| Architecture | 30 → 32 → 16 → 1 |
| Hidden activation | ReLU |
| Output activation | Sigmoid |
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Regularization | L2 + Dropout + Early stopping |

Typical test accuracy: **~96–98%**.

## Configuration
Edit `configs/config.yaml` to change layers, learning rate, epochs, etc.

## License
MIT
