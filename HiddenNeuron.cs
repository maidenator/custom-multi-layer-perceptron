using System;

namespace Backpropagation {
	/// <summary>
	/// A neuron in the hidden layer. It stores an activation value, a bias,
	/// an error, and the weights that connect it to each output neuron.
	/// </summary>
	public class HiddenNeuron {
		private double[] weights;
		private int id;
		private double activation;
		private double bias;
		private double error;
		private static Random random;

		public HiddenNeuron() {
			id = 0;
			activation = 0.0;
			bias = this.RandomWeight();
			error = 0.0;
			weights = new double[10];
			this.RandomizeWeights();
		}

		public HiddenNeuron(int id, int weightCount) {
			this.id = id;
			activation = 0.0;
			bias = 0.01;
			error = 0.0;
			weights = new double[weightCount];
			this.RandomizeWeights();
		}

		// Calculate this neuron's error contribution from the output layer.
		public void CalculateError(double[] outputErrorDerivatives) {
			double weightedSum = 0.0;
			for (int outputIndex = 0; outputIndex < outputErrorDerivatives.Length; outputIndex++) {
				weightedSum += outputErrorDerivatives[outputIndex] * weights[outputIndex];
			}
			error = weightedSum * activation * (1 - activation);
		}

		// Update the weights between this hidden neuron and the output layer.
		public void UpdateWeights(double learningRate, double[] outputErrorDerivatives) {
			double scaledActivation = activation * learningRate;
			for (int outputIndex = 0; outputIndex < outputErrorDerivatives.Length; outputIndex++) {
				weights[outputIndex] += scaledActivation * outputErrorDerivatives[outputIndex];
			}
		}

		public void SetWeight(int outputIndex, double value) {
			weights[outputIndex] = value;
		}

		public double GetWeight(int outputIndex) {
			return weights[outputIndex];
		}

		public void SetError(double value) {
			error = value;
		}

		public double GetError() {
			return error;
		}

		public void SetActivation(double value) {
			activation = value;
		}

		public double GetActivation() {
			return activation;
		}

		public void SetId(int id) {
			this.id = id;
		}

		public int GetId() {
			return id;
		}

		public void SetBias(double value) {
			bias = value;
		}

		public double GetBias() {
			return bias;
		}

		// Update this neuron's bias using its error.
		public void UpdateBias(double learningRate) {
			bias += error * learningRate;
		}

		public void RandomizeWeights() {
			for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++) {
				weights[weightIndex] = this.RandomWeight();
			}
		}

		private double RandomWeight() {
			if (random == null) {
				random = new Random();
			}

			int maxLimit = 1000;
			int minLimit = -1000;

			return (double)random.Next(minLimit, maxLimit) / 2000;
		}
	}
}
