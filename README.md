# ASTE Hotel Reviews Analysis

This repository contains the source code for our Aspect-Based Sentiment Triplet Extraction (ASTE) research project on hotel reviews.

## Project Structure
* `code/`: Contains all data processing, annotation, and model training scripts.
* `.gitignore`: Configured to exclude heavy local environments, datasets, and intermediate checkpoints.

## Model Weights
Because the trained model files (`model.safetensors`) exceed GitHub's standard file size limits, they are hosted externally under the **Releases** tab.

### How to use the model:
1. Navigate to the **Releases** section on the right side of this page.
2. Download the `v1.0.0` release assets (`model.safetensors`, config files, and tokenizer models).
3. Place the downloaded files inside a local directory named `model/` in your root folder.
4. Run the evaluation scripts inside the `code/` directory.
