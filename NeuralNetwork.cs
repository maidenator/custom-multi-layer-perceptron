using System;
using System.IO;

namespace Backpropagation {
	/// <summary>
	/// A feed-forward neural network with one input, one hidden, and one output layer.
	/// The network is trained with backpropagation using mean squared error.
	/// </summary>
	public class NeuralNetwork {
		private readonly InputNeuron[] inputNeurons;
		private readonly HiddenNeuron[] hiddenNeurons;
		private readonly OutputNeuron[] outputNeurons;

		private readonly double outputLearningRate;
		private readonly double inputLearningRate;

		private readonly double[] outputErrors;
		private readonly double[] outputErrorDerivatives;
		private readonly double[] desiredOutputs;

		/// <summary>Creates the default 1024-64-10 network used by the drawing app.</summary>
		public NeuralNetwork()
			: this(32 * 32, 64, 10, 0.2, 0.15, 1234) {
		}

		public NeuralNetwork(int inputCount, int hiddenCount, int outputCount, double outputLearningRate, double inputLearningRate, int seed) {
			if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
			if (hiddenCount <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenCount));
			if (outputCount <= 0) throw new ArgumentOutOfRangeException(nameof(outputCount));

			this.outputLearningRate = outputLearningRate;
			this.inputLearningRate = inputLearningRate;

			inputNeurons = new InputNeuron[inputCount];
			hiddenNeurons = new HiddenNeuron[hiddenCount];
			outputNeurons = new OutputNeuron[outputCount];
			desiredOutputs = new double[outputCount];
			outputErrors = new double[outputCount];
			outputErrorDerivatives = new double[outputCount];
			CreateNeurons(new Random(seed));
		}

		private void CreateNeurons(Random random) {
			for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
				inputNeurons[inputIndex] = new InputNeuron(hiddenNeurons.Length, random);
			}
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				hiddenNeurons[hiddenIndex] = new HiddenNeuron(outputNeurons.Length, random);
			}
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				outputNeurons[outputIndex] = new OutputNeuron(outputIndex);
			}
		}

		public double GetOutput(int outputIndex) {
			return outputNeurons[outputIndex].GetActivation();
		}

		public void SetInput(int inputIndex, double value) {
			inputNeurons[inputIndex].SetInput(value);
		}

		public void SetDesiredOutput(int outputIndex, double value) {
			desiredOutputs[outputIndex] = value;
		}

		/// <summary>
		/// Returns 0.5 * mean((desired - actual)^2) for the current forward pass.
		/// </summary>
		public double CalculateLoss() {
			double sum = 0.0;
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				double difference = desiredOutputs[outputIndex] - outputNeurons[outputIndex].GetActivation();
				sum += difference * difference;
			}
			return 0.5 * sum / outputNeurons.Length;
		}

		// Apply the sigmoid activation function.
		private double Sigmoid(double value) {
			if (value >= 20.00) {
				value = 32;
			}
			return 1.0 / (1.0 + Math.Exp(-value));
		}

		// Calculate the activation of each hidden neuron.
		public void CalculateHiddenActivations() {
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				double weightedSum = 0.0;
				for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
					weightedSum += inputNeurons[inputIndex].GetInput() * inputNeurons[inputIndex].GetWeight(hiddenIndex);
				}
				hiddenNeurons[hiddenIndex].SetActivation(Sigmoid(weightedSum + hiddenNeurons[hiddenIndex].GetBias()));
			}
		}

		// Calculate the activation of each output neuron.
		public void CalculateOutputActivations() {
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				double weightedSum = 0.0;
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					weightedSum += hiddenNeurons[hiddenIndex].GetActivation() * hiddenNeurons[hiddenIndex].GetWeight(outputIndex);
				}
				outputNeurons[outputIndex].SetActivation(Sigmoid(weightedSum + outputNeurons[outputIndex].GetBias()));
			}
		}

		// Calculate the error for each output neuron.
		public void CalculateOutputErrors() {
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				outputErrors[outputIndex] = desiredOutputs[outputIndex] - outputNeurons[outputIndex].GetActivation();
			}
		}

		// Calculate the derivative of the error for each output neuron.
		public void CalculateOutputErrorDerivatives() {
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				double activation = outputNeurons[outputIndex].GetActivation();
				outputErrorDerivatives[outputIndex] = activation * (1 - activation) * outputErrors[outputIndex];
			}
		}

		/// <summary>
		/// Runs one training step (forward pass plus backpropagation) and returns
		/// the mean squared error for this sample before the weights were updated.
		/// </summary>
		public double TrainStep() {
			ForwardPass();
			double loss = CalculateLoss();
			CalculateOutputErrors();
			CalculateOutputErrorDerivatives();

			// Calculate the error for each hidden neuron.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++)
				hiddenNeurons[hiddenIndex].CalculateError(outputErrorDerivatives);

			// Update the weights between the hidden and output layers.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++)
				hiddenNeurons[hiddenIndex].UpdateWeights(outputLearningRate, outputErrorDerivatives);

			// Update the weights between the input and hidden layers.
			for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					inputNeurons[inputIndex].UpdateWeight(hiddenIndex, hiddenNeurons[hiddenIndex].GetError(), inputLearningRate);
				}
			}

			// Update the output neuron biases.
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++)
				outputNeurons[outputIndex].UpdateBias(outputLearningRate, outputErrorDerivatives);

			// Update the hidden neuron biases.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++)
				hiddenNeurons[hiddenIndex].UpdateBias(inputLearningRate);

			return loss;
		}

		// Run the forward pass through the network.
		public void ForwardPass() {
			CalculateHiddenActivations();
			CalculateOutputActivations();
		}

		// Save all weights and biases to a file.
		public void SaveWeights(string filePath) {
			using (StreamWriter writer = new StreamWriter(filePath)) {
				for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
					for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
						writer.WriteLine(inputNeurons[inputIndex].GetWeight(hiddenIndex));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
						writer.WriteLine(hiddenNeurons[hiddenIndex].GetWeight(outputIndex));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					writer.WriteLine(hiddenNeurons[hiddenIndex].GetBias());
				}
				for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
					writer.WriteLine(outputNeurons[outputIndex].GetBias());
				}
			}
		}

		// Load all weights and biases from a file.
		public void LoadWeights(string filePath) {
			using (StreamReader reader = new StreamReader(filePath)) {
				for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
					for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
						inputNeurons[inputIndex].SetWeight(hiddenIndex, Convert.ToDouble(reader.ReadLine()));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
						hiddenNeurons[hiddenIndex].SetWeight(outputIndex, Convert.ToDouble(reader.ReadLine()));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					hiddenNeurons[hiddenIndex].SetBias(Convert.ToDouble(reader.ReadLine()));
				}
				for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
					outputNeurons[outputIndex].SetBias(Convert.ToDouble(reader.ReadLine()));
				}
			}
		}
	}
}
