using System;
using System.IO;

namespace Backpropagation {
	/// <summary>
	/// A feed-forward neural network with one input, one hidden, and one output layer.
	/// The network is trained with backpropagation.
	/// </summary>
	public class NeuralNetwork {
		private InputNeuron[] inputNeurons;
		private HiddenNeuron[] hiddenNeurons;
		private OutputNeuron[] outputNeurons;

		private const double OutputLearningRate = 0.2;
		private const double InputLearningRate = 0.15;

		private double[] outputErrors;
		private double[] outputErrorDerivatives;
		private double[] desiredOutputs;

		public NeuralNetwork() {
			inputNeurons = new InputNeuron[3072];
			hiddenNeurons = new HiddenNeuron[64];
			outputNeurons = new OutputNeuron[10];
			desiredOutputs = new double[10];
			outputErrors = new double[10];
			outputErrorDerivatives = new double[10];
			CreateNeurons(inputNeurons.Length, hiddenNeurons.Length, outputNeurons.Length);
		}

		public NeuralNetwork(int inputCount, int hiddenCount, int outputCount) {
			inputNeurons = new InputNeuron[inputCount];
			hiddenNeurons = new HiddenNeuron[hiddenCount];
			outputNeurons = new OutputNeuron[outputCount];

			outputErrors = new double[outputCount];
			outputErrorDerivatives = new double[outputCount];
			desiredOutputs = new double[outputCount];
			CreateNeurons(inputNeurons.Length, hiddenNeurons.Length, outputNeurons.Length);
		}

		public void CreateNeurons(int inputCount, int hiddenCount, int outputCount) {
			for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
				inputNeurons[inputIndex] = new InputNeuron(inputIndex, hiddenCount);
			}
			for (int hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex++) {
				hiddenNeurons[hiddenIndex] = new HiddenNeuron(hiddenIndex, outputCount);
			}
			for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
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

		// Apply the sigmoid activation function.
		public double Sigmoid(double value) {
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

		// Run one training step: a forward pass followed by backpropagation.
		public void TrainStep() {
			this.ForwardPass();
			this.CalculateOutputErrors();
			this.CalculateOutputErrorDerivatives();

			// Calculate the error for each hidden neuron.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++)
				hiddenNeurons[hiddenIndex].CalculateError(outputErrorDerivatives);

			// Update the weights between the hidden and output layers.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++)
				hiddenNeurons[hiddenIndex].UpdateWeights(OutputLearningRate, outputErrorDerivatives);

			// Update the weights between the input and hidden layers.
			for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					inputNeurons[inputIndex].UpdateWeight(hiddenIndex, hiddenNeurons[hiddenIndex].GetError(), InputLearningRate);
				}
			}

			// Update the output neuron biases.
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++)
				outputNeurons[outputIndex].UpdateBias(OutputLearningRate, outputErrorDerivatives);

			// Update the hidden neuron biases.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++)
				hiddenNeurons[hiddenIndex].UpdateBias(InputLearningRate);
		}

		// Run the forward pass through the network.
		public void ForwardPass() {
			this.CalculateHiddenActivations();
			this.CalculateOutputActivations();
		}

		// Return true when the network has converged.
		public bool HasConverged() {
			bool isConverged = true;
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				if ((outputErrors[outputIndex] - outputErrorDerivatives[outputIndex]) != 0)
					isConverged = false;
			}
			return isConverged;
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
