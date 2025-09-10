# Fashion-MNIST LeNet-5 Classifier 

This repository is an end-to-end deep learning workflow: implementing and training the **LeNet-5 convolutional neural network** on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  

It applies key ML skills:
- **Model implementation** - LeNet-5 architecture written from scratch in Python (no high-level frameworks).  
- **Data handling** - Custom utilities to preprocess Fashion-MNIST images and labels.  
- **Training & evaluation** - Scripts for model training, saving parameters, and making predictions.  
- **Reproducibility** - Clear instructions and structured code for replicating experiments.  

---

## Contents

- **`train_lenet5.py`** - Training pipeline (data loading, training loop, evaluation).  
- **`predict_lenet5.py`** - Inference script for classifying new images.  
- **`lenet5_core.py`** - Core implementation of the LeNet-5 architecture.  
- **`utils_fashion_mnist.py`** - Data preprocessing and utility functions.  
- **`model_params_lenet5.json`** - Saved model parameters (weights).  
- **`Dataset/`** - Fashion-MNIST data (IDX format).  
- **`README.md`** - Project documentation (this file).  

---

## Project Workflow

1. **Dataset Preparation**  
   - Uses official Fashion-MNIST (10 categories of clothing/shoes).  
   - Loaded directly from raw IDX binary files in the `Dataset/` folder.  

2. **Model Implementation**  
   - LeNet-5 architecture: two convolutional layers, two subsampling layers, and fully connected layers (120 -> 84 -> 10).  
   - Implemented in pure Python for learning clarity.  

3. **Training**  
   - Run `train_lenet5.py` to train the model.  
   - Parameters saved to `model_params_lenet5.json`.  

4. **Inference**  
   - Run `predict_lenet5.py` with an input image.  
   - Image is resized to 28×28 grayscale and classified into one of 10 categories.  

---

## Running the Project

### 1. Install Requirements
This project uses only core Python libraries (`numpy`, `PIL`).  

```bash
pip install numpy pillow
