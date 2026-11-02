using System;

namespace Backpropagation {
	internal static class Program {
		private static void Main() {
			(double[][] inputs, int[] labels) = DatasetLoader.Load(
				"data/t10k-images-idx3-ubyte",
				"data/t10k-labels-idx1-ubyte");

			int sampleIndex = 0;
			double[] image = inputs[sampleIndex];

			NeuralNetwork network = new NeuralNetwork();
			for (int inputIndex = 0; inputIndex < image.Length; inputIndex++) {
				network.SetInput(inputIndex, image[inputIndex]);
			}
			network.ForwardPass();

			Console.WriteLine($"Sample {sampleIndex}, actual label: {labels[sampleIndex]}");
			for (int outputIndex = 0; outputIndex < 10; outputIndex++) {
				Console.WriteLine($"  {outputIndex}: {network.GetOutput(outputIndex):F4}");
			}

			int predicted = 0;
			for (int outputIndex = 1; outputIndex < 10; outputIndex++) {
				if (network.GetOutput(outputIndex) > network.GetOutput(predicted)) {
					predicted = outputIndex;
				}
			}

			Console.WriteLine($"Predicted: {predicted} ({(predicted == labels[sampleIndex] ? "correct" : "wrong")})");
		}
	}
}
