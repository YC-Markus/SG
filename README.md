# Super-Gaussian: Multi-Contrast MRI Super-Resolution with Superpixel-Guided 2D Gaussian Tokenization

This is a repository containing the official implementation of the paper: **Super-Gaussian: Multi-Contrast MRI Super-Resolution with Superpixel-Guided 2D Gaussian Tokenization**.

## ⚙️ Requirements

**Important:** To support the variable-length attention mechanism described in the paper, this implementation relies on **xformers**.

You may need to install it manually. Please refer to the [official xformers repository](https://github.com/facebookresearch/xformers) for detailed installation instructions.

## 🚀 Usage

To train the model, simply download this repository, navigate to the `SG_pub` directory, and run the training script.

```bash
cd SG_pub
python train.py

```
