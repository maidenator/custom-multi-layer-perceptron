using System;

namespace Backpropagation {
	/// <summary>
	/// A neuron in the output layer. It stores an activation value, a bias,
	/// and an identifier used to look up its error derivative.
	/// </summary>
	public class OutputNeuron {
		private double bias;
		private int id;
		private double activation;
		private static Random random;

		public OutputNeuron() {
			bias = this.RandomBias();
			id = 0;
			activation = 0.0;
		}

		public OutputNeuron(int id) {
			bias = 0.01;
			this.id = id;
			activation = 0.0;
		}

		public void SetId(int id) {
			this.id = id;
		}

		public int GetId() {
			return id;
		}

		public void SetActivation(double value) {
			activation = value;
		}

		public double GetActivation() {
			return activation;
		}

		public double GetBias() {
			return bias;
		}

		public void SetBias(double value) {
			bias = value;
		}

		// Update this neuron's bias using its error derivative.
		public void UpdateBias(double learningRate, double[] outputErrorDerivatives) {
			bias += learningRate * outputErrorDerivatives[id];
		}

		private double RandomBias() {
			if (random == null) {
				random = new Random();
			}

			int maxLimit = 1000;
			int minLimit = -1000;

			return (double)random.Next(minLimit, maxLimit) / 2000;
		}
	}
}
