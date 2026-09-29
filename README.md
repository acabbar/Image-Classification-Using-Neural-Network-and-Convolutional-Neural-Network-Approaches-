# TinyImageNet Model Comparison: MLP, CNN, and ResNet18

A comparative computer-vision study of representation learning on a fixed 15-class subset of TinyImageNet100. The project evaluates a fully connected neural network, a convolutional neural network trained from scratch, and a pretrained ResNet18 adapted through transfer learning. Results are also compared with three handcrafted-feature baselines evaluated on the same class subset.

## Results

| Method | Test accuracy |
|---|---:|
| ORB + Bag of Visual Words + LinearSVC | 7.26% |
| SIFT + Bag of Visual Words + LinearSVC | 22.00% |
| Fisher Vector + LinearSVC | 22.93% |
| Fully connected neural network | 40.80% |
| CNN trained from scratch | 58.40% |
| ResNet18 with transfer learning | 83.73% |

Moving from the strongest handcrafted baseline to the MLP improved accuracy by 17.87 percentage points. Preserving image structure with a CNN added 17.60 points, while pretrained ResNet18 added a further 25.33 points and produced the strongest result.

## Dataset setup

The experiment selects 15 fixed TinyImageNet100 classes so every method is evaluated on the same categories.

- 6,000 training images: 400 per class
- 1,500 test images: 100 per class
- 15 output classes identified by ImageNet WordNet IDs
- Training augmentation: random horizontal flips and rotations
- Separate preprocessing pipelines for the MLP, scratch CNN, and pretrained ResNet18

TinyImageNet100 is not stored in this repository. Obtain the dataset separately, place it at `data/TinyImageNet100_2026`, or update the configurable dataset path in the notebook. Verify the dataset licence and redistribution terms with its original provider.

## Models

### Fully connected network

Images are resized to 32 x 32, normalized, and flattened to 3,072 features. The network uses hidden layers of 512, 256, and 128 units with batch normalization, ReLU activations, and dropout.

### CNN trained from scratch

Images are resized to 64 x 64. Three convolutional blocks learn 32, 64, and 128 feature maps, followed by max pooling and a fully connected classifier.

### ResNet18 transfer learning

Images are resized to 224 x 224 and normalized with the standard pretrained ResNet statistics. A pretrained ResNet18 is fine-tuned after replacing its final layer with a 15-class output layer.

## Workflow

1. Select a fixed 15-class TinyImageNet subset.
2. Create deterministic per-class training and test partitions.
3. Train and evaluate the MLP for 25 epochs.
4. Train and evaluate the scratch CNN for 25 epochs.
5. Fine-tune and evaluate pretrained ResNet18 for 15 epochs.
6. Compare all methods using test accuracy and confusion matrices.
7. Save reusable PyTorch checkpoints.

## Repository contents

| File | Purpose |
|---|---|
| `tinyimagenet_model_comparison.ipynb` | Full data preparation, training, evaluation, plots, and interpretation |
| `model_nn_checkpoint.pth` | Saved fully connected network checkpoint |
| `model_cnn_checkpoint.pth` | Saved scratch-CNN checkpoint |
| `Report-CV Project 2.pdf` | Original project report retained as part of the project history |

The notebook also writes `model_resnet_checkpoint.pth` when the complete workflow is run, but that checkpoint is not currently included in the repository.

## Run locally

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    jupyter lab

On macOS or Linux, activate the environment with `source .venv/bin/activate`. Open `tinyimagenet_model_comparison.ipynb` and confirm that its dataset path points to your local TinyImageNet100 directory before running the cells.

## Limitations

- The comparison uses a selected 15-class subset rather than the complete dataset.
- The notebook uses a fixed first-400/last-100 per-class split instead of an official split or cross-validation.
- Training accuracy is recorded, but no independent validation split is used for model selection.
- The handcrafted baseline figures are imported from the related feature-engineering experiment rather than recomputed in this notebook.
- Results depend on pretrained weights, stochastic augmentation, hardware, and library versions.

## Key takeaway

The experiment demonstrates the value of spatial inductive bias and transfer learning. Flattening images for a dense network outperformed handcrafted descriptors, a CNN trained from scratch improved further, and pretrained ResNet18 achieved the best generalization by a large margin.
