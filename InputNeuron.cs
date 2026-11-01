using System;

namespace Backpropagation {
	/// <summary>
	/// A neuron in the input layer. It stores one input value and the weights
	/// that connect it to each hidden neuron.
	/// </summary>
	public class InputNeuron {
		private double[] weights;
		private int id;
		private double input;
		private static Random random;

		public InputNeuron() {
			id = 0;
			input = 0;
			weights = new double[64];
			this.RandomizeWeights();
		}

		public InputNeuron(int id, int weightCount) {
			this.id = id;
			input = 0;
			weights = new double[weightCount];
			this.RandomizeWeights();
		}

		private double RandomWeight() {
			if (random == null) {
				random = new Random();
			}

			int maxLimit = 1000;
			int minLimit = -1000;

			return (double)random.Next(minLimit, maxLimit) / 2000;
		}

		public void RandomizeWeights() {
			for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++) {
				weights[weightIndex] = this.RandomWeight();
			}
		}

		// Update the weight between this input neuron and one hidden neuron.
		public void UpdateWeight(int hiddenIndex, double error, double learningRate) {
			double scaledError = error * learningRate;
			weights[hiddenIndex] += scaledError * input;
		}

		public int GetId() {
			return id;
		}

		public void SetId(int id) {
			this.id = id;
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
