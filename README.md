# Lung Cancer Detection — Deep CNN Experiments

This repository contains Jupyter notebooks and the dataset used to train and evaluate convolutional neural networks for classifying lung tissue images (normal vs. cancer subtypes). The focus is proof-of-concept experiments using several popular CNN backbones and transfer learning.

## Repository Contents
- Notebooks (model training & evaluation):
	- [Final_Lung_Cancer_DenseNet169.ipynb](Final_Lung_Cancer_DenseNet169.ipynb)
	- [Final_Lung_Cancer_EfficientNet_B3.ipynb](Final_Lung_Cancer_EfficientNet_B3.ipynb)
	- [Final_Lung_Cancer_MobileNetV2.ipynb](Final_Lung_Cancer_MobileNetV2.ipynb)
	- [Final_Lung_Cancer_ResNet152.ipynb](Final_Lung_Cancer_ResNet152.ipynb)
	- [Final_Lung_Cancer_VGG19.ipynb](Final_Lung_Cancer_VGG19.ipynb)

- Dataset: [Dataset_1](Dataset_1) (train / valid / test splits)

## Project Summary
- Problem: classify lung histology images into classes such as `normal`, `adenocarcinoma`, `large.cell.carcinoma`, and `squamous.cell.carcinoma`.
- Approach: transfer learning with pretrained CNN backbones, image augmentation, and standard supervised training loops implemented inside notebooks.
- Outputs: trained model weights, evaluation metrics (accuracy, precision, recall, confusion matrices), and plots saved by the notebooks.

## Dataset Structure
The dataset root is `Dataset_1/` and follows a standard folder-per-class layout under each split:

- `Dataset_1/train/<class>/*` — training images
- `Dataset_1/valid/<class>/*` — validation images
- `Dataset_1/test/<class>/*` — test images

Notes:
- Some class folders in `train` and `valid` contain descriptive suffixes (patient location, stage). The notebooks typically read folders by class name; ensure the expected class names match the folder names or adjust the data-loading cells.
- Images may have varying sizes and channels — preprocessing in the notebooks resizes and normalizes images.

## Environment & Requirements
Create an isolated Python environment (conda/venv). Example minimal requirements used across notebooks:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow>=2.6
keras
opencv-python
tqdm
```

To install with pip:

```bash
python -m pip install -r requirements.txt
```

GPU notes:
- For GPU training, install the GPU-enabled TensorFlow matching your CUDA/cuDNN versions (see TensorFlow official install docs). Running on GPU is strongly recommended for reasonable training times.

## Quick Start — Run a Notebook
1. Open the notebook you want to run (for example: [Final_Lung_Cancer_DenseNet169.ipynb](Final_Lung_Cancer_DenseNet169.ipynb)).
2. Update any dataset path variables at the top of the notebook if your dataset is in a different location.
3. Inspect and adjust hyperparameters (batch size, learning rate, image size) as needed.
4. Run cells sequentially. Optionally enable GPU in your notebook environment.

Example: run a notebook from the terminal (Linux/macOS/WSL/PowerShell) using `jupyter nbconvert` to execute (non-interactive):

```bash
jupyter nbconvert --to notebook --execute "Final_Lung_Cancer_DenseNet169.ipynb" --ExecutePreprocessor.timeout=600
```

## Reproducibility & Checkpoints
- Notebooks typically save model weights and training history to disk — check the notebook cells that call `model.save()` or `history.to_csv()`.
- To reproduce results, fix random seeds (notebooks may already do this) and log package versions (`pip freeze > pip-freeze.txt`).

## Evaluation
- Each notebook computes evaluation metrics on the validation and test sets: accuracy, precision, recall, F1, and confusion matrices.
- Use the saved model weights and the notebook evaluation cells to reproduce specific metrics and plots.

## Suggested Next Steps (automation and reproducibility)
- Generate a `requirements.txt` from your running environment: `pip freeze > requirements.txt`.
- Extract notebook training logic into a small script (e.g., `train.py`) to enable CLI training and easier CI.
- Add a `models/` folder to store saved checkpoints and include a small evaluation script (e.g., `evaluate.py`).

## Tips & Troubleshooting
- If images fail to load, verify the file permissions and that class folder names match.
- If training is slow or runs out of memory, reduce `batch_size` or `image_size`, or use mixed precision training (if supported by hardware).
- For imbalanced classes, consider class weighting, focal loss, or additional augmentation for minority classes.

## Contributing
- This repo is structured around notebooks; pull requests that add a reproducible script version (`train.py`, `evaluate.py`) are welcome.

## License
Add a license file if you plan to share this repository publicly. If unsure, ask and I can suggest a suitable license.

## Contact
If you'd like, I can:
- produce a `requirements.txt` from your environment
- create `train.py` and `evaluate.py` wrappers to run the experiments outside notebooks
- add a Dockerfile or conda environment for reproducible runs

Tell me which of the above you'd like next and I'll prepare it.
