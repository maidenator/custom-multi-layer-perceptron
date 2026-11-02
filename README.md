# Custom Multi-Layer Perceptron

A handwritten-digit classifier written from scratch in C#. The network reads
MNIST images, trains with backpropagation, and can be used from a small
Avalonia drawing app.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Train the network

From the repository root:

```powershell
dotnet run
```

Training loads the MNIST files from `data/`, runs 5 epochs over the 60,000
training images, and prints accuracy per epoch. Checkpoints are written to
`weights/epoch-1.txt` through `weights/epoch-5.txt`. This can take several
minutes.

## Run the drawing app

```powershell
dotnet run --project Gui
```

The app loads `weights/epoch-5.txt`, so train first if that file is missing.
Draw a digit on the canvas and press **Read** to classify it.

## Project layout

| Path | Purpose |
| --- | --- |
| `Program.cs` | Console entry point: training loop and evaluation |
| `DatasetLoader.cs` | Reads MNIST IDX files, pads 28×28 images to 32×32 and normalizes them |
| `NeuralNetwork.cs` | Forward pass, backpropagation, and weight persistence |
| `InputNeuron.cs`, `HiddenNeuron.cs`, `OutputNeuron.cs` | Per-layer neuron implementations |
| `DigitNormalizer.cs` | Crops, scales, and centers hand-drawn input like MNIST does |
| `Gui/` | Avalonia desktop app for drawing and classifying digits |
| `data/` | MNIST training and test images/labels |
| `weights/` | Saved network weights, one file per epoch |

## Notes

- `dotnet build` at the repository root builds only the console app. The GUI is
  a separate project; use `--project Gui` or build inside `Gui/`.
- Build output (`bin/`, `obj/`) is ignored by git.
