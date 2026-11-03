using System;

namespace Backpropagation {
	/// <summary>
	/// A neuron in the input layer. It stores one input value and the weights
	/// that connect it to each hidden neuron.
	/// </summary>
	public class InputNeuron {
		private readonly double[] weights;
		private readonly Random random;
		private double input;

		public InputNeuron(int weightCount, Random random) {
			input = 0;
			weights = new double[weightCount];
			this.random = random;
			RandomizeWeights();
		}

		private double RandomWeight() {
			return random.Next(-1000, 1000) / 2000.0;
		}

		private void RandomizeWeights() {
			for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++) {
				weights[weightIndex] = RandomWeight();
			}
		}

		// Update the weight between this input neuron and one hidden neuron.
		public void UpdateWeight(int hiddenIndex, double error, double learningRate) {
			double scaledError = error * learningRate;
			weights[hiddenIndex] += scaledError * input;
		}

		public double GetInput() {
			return input;
		}

		public void SetInput(double value) {
			input = value;
		}

		public double GetWeight(int hiddenIndex) {
			return weights[hiddenIndex];
		}

		public void SetWeight(int weightIndex, double value) {
			weights[weightIndex] = value;
		}
	}
}
