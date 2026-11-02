using System;
using System.IO;

namespace Backpropagation {
	internal static class Program {
		private const int SampleCount = 60000;
		private const int Epochs = 5;
		private const double CorrectTarget = 0.9;
		private const double WrongTarget = 0.1;
		private const string WeightDirectory = "weights";

		private static void Main() {
			(double[][] trainInputs, int[] trainLabels) = DatasetLoader.Load(
				"data/train-images-idx3-ubyte",
				"data/train-labels-idx1-ubyte");

			(double[][] testInputs, int[] testLabels) = DatasetLoader.Load(
				"data/t10k-images-idx3-ubyte",
				"data/t10k-labels-idx1-ubyte");

			Directory.CreateDirectory(WeightDirectory);

			NeuralNetwork network = new NeuralNetwork();
			Random random = new Random(1234);
			int[] order = new int[SampleCount];
			for (int i = 0; i < order.Length; i++) {
				order[i] = i;
			}

			for (int epoch = 1; epoch <= Epochs; epoch++) {
				DateTime epochStart = DateTime.Now;
				Shuffle(order, random);
				int correct = 0;

				for (int position = 0; position < order.Length; position++) {
					int index = order[position];
					TrainOnSample(network, trainInputs[index], trainLabels[index]);

					if (Predict(network) == trainLabels[index]) {
						correct++;
					}
				}

				double testAccuracy = Evaluate(network, testInputs, testLabels);
				network.SaveWeights(Path.Combine(WeightDirectory, $"epoch-{epoch}.txt"));

				double seconds = (DateTime.Now - epochStart).TotalSeconds;
				Console.WriteLine($"epoch {epoch}  train {100.0 * correct / order.Length,5:F1}%  test {testAccuracy,5:F1}%  {seconds:F0}s");
			}
		}

		private static void TrainOnSample(NeuralNetwork network, double[] image, int label) {
			SetInputs(network, image);

			for (int outputIndex = 0; outputIndex < 10; outputIndex++) {
				network.SetDesiredOutput(outputIndex, outputIndex == label ? CorrectTarget : WrongTarget);
			}

			network.TrainStep();
		}

		private static double Evaluate(NeuralNetwork network, double[][] inputs, int[] labels) {
			int correct = 0;
			for (int index = 0; index < inputs.Length; index++) {
				SetInputs(network, inputs[index]);
				network.ForwardPass();

				if (Predict(network) == labels[index]) {
					correct++;
				}
			}
			return 100.0 * correct / inputs.Length;
		}

		private static void SetInputs(NeuralNetwork network, double[] image) {
			for (int inputIndex = 0; inputIndex < image.Length; inputIndex++) {
				network.SetInput(inputIndex, image[inputIndex]);
			}
		}

		private static int Predict(NeuralNetwork network) {
			int best = 0;
			for (int outputIndex = 1; outputIndex < 10; outputIndex++) {
				if (network.GetOutput(outputIndex) > network.GetOutput(best)) {
					best = outputIndex;
				}
			}
			return best;
		}

		private static void Shuffle(int[] order, Random random) {
			for (int i = order.Length - 1; i > 0; i--) {
				int j = random.Next(i + 1);
				(order[i], order[j]) = (order[j], order[i]);
			}
		}
	}
}
