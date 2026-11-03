# Handwritten Digit Recognizer (C#)

A small neural network built from scratch in C# that recognizes handwritten
digits from the MNIST dataset. It uses no machine learning libraries — just the
input, hidden, and output neuron classes in this repository.

## How it works

- Input: 1,024 values (32×32 pixel images).
- One hidden layer: 64 neurons by default.
- Output: 10 neurons, one for each digit.
- Training: backpropagation with mean squared error loss.

## Data

[MNIST](https://yann.lecun.com/exdb/mnist/) has 70,000 images of handwritten
digits (28×28 pixels, labels 0–9).

- 60,000 training images are split into **50,000 for training** and **10,000 for
  validation** (used to check progress during training).
- The official **10,000 test images** are used once at the end.
- Images are padded to 32×32 and scaled from 0–255 to 0–1.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Train

From the project folder:

```powershell
dotnet run
```

This runs 5 epochs with the tuned settings. It prints the loss and accuracy after
each epoch:

```text
Loading MNIST...
config: hidden=64 lr=0.3 hidden-lr=0.2 epochs=5 seed=1234
split: train=50000 validation=10000 test=10000
epoch  1  loss 0.0064  train 90.71%  val_loss 0.0041  val 94.62%    72.6s
epoch  2  loss 0.0038  train 95.05%  val_loss 0.0034  val 95.45%    68.6s
...
```

It saves:

- `weights/best.txt` — the best model based on validation results.
- `weights/epoch-N.txt` — the model after each epoch.
- `logs/*.csv` — training and test results.
- `logs/*.svg` — loss, accuracy, and confusion-matrix charts.

## Options

```powershell
dotnet run -- --hidden 128 --lr 0.15 --hidden-lr 0.12 --epochs 8 --seed 42
dotnet run -- --train-limit 10000 --val-limit 2000 --epochs 3 --skip-test
dotnet run -- --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--hidden <n>` | 64 | Number of hidden neurons |
| `--lr <x>` | 0.3 | Output-layer learning rate |
| `--hidden-lr <x>` | 0.2 | Hidden-layer learning rate |
| `--epochs <n>` | 5 | Number of training passes |
| `--seed <n>` | 1234 | Seed for the data split and weight initialization |
| `--train-limit <n>` | 50000 | Limit the training images |
| `--val-limit <n>` | 10000 | Limit the validation images |
| `--output-dir <path>` | `logs` | Where logs and charts go |
| `--weight-dir <path>` | `weights` | Where saved weights go |
| `--summary-csv <path>` | — | Add a one-line summary to a CSV |
| `--skip-test` | off | Skip the final test run while tuning |

Using the same seed and settings gives the same result.

## Hyperparameter tuning

`scripts/tune.ps1` tries several settings on a smaller subset of the data and
writes the results to `logs/experiments.csv`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/tune.ps1
```

| Hidden | Output lr | Hidden lr | Epochs | Best val loss | Best val accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.10 | 0.10 | 2 | 0.008256 | 88.0% |
| 64 | 0.10 | 0.10 | 2 | 0.007960 | 89.6% |
| 64 | 0.20 | 0.15 | 2 | 0.007259 | 90.5% |
| 64 | 0.30 | 0.20 | 2 | **0.006893** | **90.8%** |
| 128 | 0.10 | 0.10 | 2 | 0.007914 | 89.9% |
| 128 | 0.20 | 0.15 | 2 | 0.007410 | 90.6% |
| 64 | 0.20 | 0.15 | 1 | 0.009151 | 86.8% |
| 64 | 0.20 | 0.15 | 4 | 0.006216 | 92.0% |

The best setup was 64 hidden neurons with learning rates 0.3 and 0.2. A larger
hidden layer did not help and was slower. The final model uses the saved weights
with the best validation score.

## Results

Final settings: **64 hidden neurons, lr 0.3, hidden-lr 0.2, seed 1234, 5 epochs**.

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0064 | 90.71% | 0.0041 | 94.62% |
| 2 | 0.0038 | 95.05% | 0.0034 | 95.45% |
| 3 | 0.0032 | 96.09% | 0.0031 | 96.21% |
| 4 | 0.0029 | 96.67% | 0.0029 | 96.40% |
| 5 | **0.0026** | **96.92%** | **0.0027** | **96.65%** |

The loss goes down every epoch. On the test set, the model gets **96.03%
accuracy** with an average F1 score of **0.960**. Digits 5 and 8 are the hardest
(F1 0.942); digit 1 is the easiest (F1 0.982).

### Charts

![Training and validation loss](logs/loss-h64-lr0.3-hlr0.2-e5-s1234.svg)

![Training and validation accuracy](logs/accuracy-h64-lr0.3-hlr0.2-e5-s1234.svg)

![Test confusion matrix](logs/test-h64-lr0.3-hlr0.2-e5-s1234-confusion.svg)

The full logs are in `logs/`:

- `train-h64-lr0.3-hlr0.2-e5-s1234.csv` — loss and accuracy per epoch.
- `test-h64-lr0.3-hlr0.2-e5-s1234-metrics.csv` — results for each digit.
- `test-h64-lr0.3-hlr0.2-e5-s1234-confusion.csv` — confusion matrix.
- `experiments.csv` — tuning results and the final run.

## Drawing app

```powershell
dotnet run --project Gui
```

The app loads `weights/best.txt`, so train first. Draw a digit and press **Read**.

## Files

| Path | Purpose |
| --- | --- |
| `Program.cs` | Training loop, data split, evaluation, and logging |
| `NeuralNetwork.cs`, `InputNeuron.cs`, `HiddenNeuron.cs`, `OutputNeuron.cs` | The network |
| `DatasetLoader.cs`, `DigitNormalizer.cs` | Data loading and image preparation |
| `TrainingConfig.cs` | Command-line settings |
| `Metrics.cs`, `ChartWriter.cs` | Accuracy/f1 metrics and SVG charts |
| `scripts/tune.ps1` | Tuning script |
| `Gui/` | Drawing app |
| `data/` | MNIST files |
| `weights/` | Saved models |
| `logs/` | Training logs and charts |
