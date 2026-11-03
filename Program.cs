using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Backpropagation {
	internal static class Program {
		private const int InputSize = 32 * 32;
		private const int OutputCount = 10;
		private const double CorrectTarget = 0.9;
		private const double WrongTarget = 0.1;

		private const string TrainImagesPath = "data/train-images-idx3-ubyte";
		private const string TrainLabelsPath = "data/train-labels-idx1-ubyte";
		private const string TestImagesPath = "data/t10k-images-idx3-ubyte";
		private const string TestLabelsPath = "data/t10k-labels-idx1-ubyte";

		private sealed class EvaluationResult {
			public double Loss { get; }
			public ClassificationMetrics Metrics { get; }

			public EvaluationResult(double loss, ClassificationMetrics metrics) {
				Loss = loss;
				Metrics = metrics;
			}
		}

		private static void Main(string[] args) {
			TrainingConfig config;
			try {
				config = TrainingConfig.Parse(args);
			}
			catch (ArgumentException exception) {
				Console.Error.WriteLine(exception.Message);
				Console.Error.WriteLine();
				Console.Error.WriteLine(TrainingConfig.Usage);
				Environment.ExitCode = 1;
				return;
			}

			if (config.ShowHelp) {
				Console.WriteLine(TrainingConfig.Usage);
				return;
			}

			Run(config);
		}

		private static void Run(TrainingConfig config) {
			Console.WriteLine("Loading MNIST...");
			(double[][] trainInputs, int[] trainLabels) = DatasetLoader.Load(TrainImagesPath, TrainLabelsPath);
			(double[][] testInputs, int[] testLabels) = DatasetLoader.Load(TestImagesPath, TestLabelsPath);

			// Keep MNIST's official test set untouched and carve a validation set
			// out of the 60k training images: 50k train / 10k validation / 10k test.
			int validationSplitSize = Math.Min(TrainingConfig.ValidationSize, trainInputs.Length / 2);
			int trainSplitSize = trainInputs.Length - validationSplitSize;

			int[] allTrainIndices = CreateIndices(trainInputs.Length);
			Shuffle(allTrainIndices, new Random(config.Seed));

			int trainCount = Math.Min(config.TrainLimit ?? trainSplitSize, trainSplitSize);
			int validationCount = Math.Min(config.ValidationLimit ?? validationSplitSize, validationSplitSize);
			int[] trainIndices = Slice(allTrainIndices, 0, trainCount);
			int[] validationIndices = Slice(allTrainIndices, trainSplitSize, validationCount);
			int[] testIndices = CreateIndices(testInputs.Length);

			Directory.CreateDirectory(config.OutputDirectory);
			Directory.CreateDirectory(config.WeightDirectory);

			Console.WriteLine(FormattableString.Invariant(
				$"config: hidden={config.HiddenCount} lr={config.OutputLearningRate} hidden-lr={config.HiddenLearningRate} epochs={config.Epochs} seed={config.Seed}"));
			Console.WriteLine(FormattableString.Invariant(
				$"split: train={trainIndices.Length} validation={validationIndices.Length} test={testIndices.Length}"));

			NeuralNetwork network = new NeuralNetwork(
				InputSize,
				config.HiddenCount,
				OutputCount,
				config.OutputLearningRate,
				config.HiddenLearningRate,
				config.Seed);
			Random epochRandom = new Random(config.Seed + 1);

			List<string> logRows = new List<string>();
			logRows.Add("epoch,train_loss,train_accuracy,val_loss,val_accuracy,seconds");

			double[] trainLosses = new double[config.Epochs];
			double[] validationLosses = new double[config.Epochs];
			double[] trainAccuracies = new double[config.Epochs];
			double[] validationAccuracies = new double[config.Epochs];

			double bestValidationLoss = double.MaxValue;
			double bestValidationAccuracy = 0.0;
			int bestEpoch = 0;
			DateTime trainingStart = DateTime.Now;

			for (int epoch = 1; epoch <= config.Epochs; epoch++) {
				DateTime epochStart = DateTime.Now;
				Shuffle(trainIndices, epochRandom);

				double totalTrainLoss = 0.0;
				int correct = 0;

				for (int position = 0; position < trainIndices.Length; position++) {
					int index = trainIndices[position];
					totalTrainLoss += TrainOnSample(network, trainInputs[index], trainLabels[index]);

					if (Predict(network) == trainLabels[index]) {
						correct++;
					}
				}

				EvaluationResult validation = Evaluate(network, trainInputs, validationIndices, trainLabels);
				double trainLoss = totalTrainLoss / trainIndices.Length;
				double trainAccuracy = 100.0 * correct / trainIndices.Length;

				trainLosses[epoch - 1] = trainLoss;
				validationLosses[epoch - 1] = validation.Loss;
				trainAccuracies[epoch - 1] = trainAccuracy;
				validationAccuracies[epoch - 1] = validation.Metrics.Accuracy * 100.0;

				network.SaveWeights(Path.Combine(config.WeightDirectory, $"epoch-{epoch}.txt"));

				if (validation.Loss < bestValidationLoss) {
					bestValidationLoss = validation.Loss;
					bestValidationAccuracy = validation.Metrics.Accuracy * 100.0;
					bestEpoch = epoch;
					network.SaveWeights(Path.Combine(config.WeightDirectory, "best.txt"));
				}

				double seconds = (DateTime.Now - epochStart).TotalSeconds;
				logRows.Add(FormattableString.Invariant(
					$"{epoch},{trainLoss:F6},{trainAccuracy:F3},{validation.Loss:F6},{validation.Metrics.Accuracy * 100.0:F3},{seconds:F2}"));

				Console.WriteLine(FormattableString.Invariant(
					$"epoch {epoch,2}  loss {trainLoss:F4}  train {trainAccuracy,5:F2}%  val_loss {validation.Loss:F4}  val {validation.Metrics.Accuracy * 100.0,5:F2}%  {seconds,6:F1}s"));
			}

			double totalSeconds = (DateTime.Now - trainingStart).TotalSeconds;
			string trainLogPath = Path.Combine(config.OutputDirectory, $"train-{config.RunName}.csv");
			File.WriteAllLines(trainLogPath, logRows);

			ChartWriter.WriteLineChart(
				Path.Combine(config.OutputDirectory, $"loss-{config.RunName}.svg"),
				"Training and validation loss",
				"MSE loss",
				new[] { "train", "validation" },
				new[] { trainLosses, validationLosses },
				0.0,
				false);
			ChartWriter.WriteLineChart(
				Path.Combine(config.OutputDirectory, $"accuracy-{config.RunName}.svg"),
				"Training and validation accuracy",
				"accuracy",
				new[] { "train", "validation" },
				new[] { trainAccuracies, validationAccuracies },
				100.0,
				true);

			Console.WriteLine(FormattableString.Invariant(
				$"best epoch {bestEpoch}  val_loss {bestValidationLoss:F4}  val {bestValidationAccuracy:F2}%  total {totalSeconds:F1}s"));
			Console.WriteLine(FormattableString.Invariant($"wrote {trainLogPath}"));

			ClassificationMetrics testMetrics = null;
			if (!config.SkipTest) {
				NeuralNetwork bestNetwork = new NeuralNetwork(
					InputSize,
					config.HiddenCount,
					OutputCount,
					config.OutputLearningRate,
					config.HiddenLearningRate,
					config.Seed);
				bestNetwork.LoadWeights(Path.Combine(config.WeightDirectory, "best.txt"));

				EvaluationResult test = Evaluate(bestNetwork, testInputs, testIndices, testLabels);
				testMetrics = test.Metrics;

				Console.WriteLine();
				Console.WriteLine(FormattableString.Invariant($"test loss {test.Loss:F4}"));
				Console.Write(test.Metrics.FormatTable());

				string testPrefix = Path.Combine(config.OutputDirectory, $"test-{config.RunName}");
				File.WriteAllText(testPrefix + "-metrics.csv", test.Metrics.ToMetricsCsv());
				File.WriteAllText(testPrefix + "-confusion.csv", test.Metrics.ToConfusionCsv());
				ChartWriter.WriteConfusionMatrix(testPrefix + "-confusion.svg", test.Metrics.GetConfusionMatrix(), test.Metrics.ClassCount);
				Console.WriteLine(FormattableString.Invariant($"wrote {testPrefix}-metrics.csv and {testPrefix}-confusion.svg"));
			}

			if (!string.IsNullOrEmpty(config.SummaryCsvPath)) {
				AppendSummary(config, bestEpoch, bestValidationLoss, bestValidationAccuracy, totalSeconds, testMetrics);
			}
		}

		private static double TrainOnSample(NeuralNetwork network, double[] image, int label) {
			SetInputs(network, image);
			SetTargets(network, label);
			return network.TrainStep();
		}

		private static EvaluationResult Evaluate(NeuralNetwork network, double[][] inputs, int[] indices, int[] labels) {
			double totalLoss = 0.0;
			int[] actual = new int[indices.Length];
			int[] predicted = new int[indices.Length];

			for (int position = 0; position < indices.Length; position++) {
				int index = indices[position];
				SetInputs(network, inputs[index]);
				SetTargets(network, labels[index]);
				network.ForwardPass();

				totalLoss += network.CalculateLoss();
				actual[position] = labels[index];
				predicted[position] = Predict(network);
			}

			ClassificationMetrics metrics = ClassificationMetrics.Create(actual, predicted, OutputCount);
			return new EvaluationResult(totalLoss / indices.Length, metrics);
		}

		private static void SetInputs(NeuralNetwork network, double[] image) {
			for (int inputIndex = 0; inputIndex < image.Length; inputIndex++) {
				network.SetInput(inputIndex, image[inputIndex]);
			}
		}

		private static void SetTargets(NeuralNetwork network, int label) {
			for (int outputIndex = 0; outputIndex < OutputCount; outputIndex++) {
				network.SetDesiredOutput(outputIndex, outputIndex == label ? CorrectTarget : WrongTarget);
			}
		}

		private static int Predict(NeuralNetwork network) {
			int best = 0;
			for (int outputIndex = 1; outputIndex < OutputCount; outputIndex++) {
				if (network.GetOutput(outputIndex) > network.GetOutput(best)) {
					best = outputIndex;
				}
			}
			return best;
		}

		private static int[] CreateIndices(int count) {
			int[] indices = new int[count];
			for (int index = 0; index < count; index++) {
				indices[index] = index;
			}
			return indices;
		}

		private static int[] Slice(int[] source, int start, int count) {
			int[] result = new int[count];
			Array.Copy(source, start, result, 0, count);
			return result;
		}

		private static void Shuffle(int[] indices, Random random) {
			for (int i = indices.Length - 1; i > 0; i--) {
				int j = random.Next(i + 1);
				(indices[i], indices[j]) = (indices[j], indices[i]);
			}
		}

		private static void AppendSummary(
			TrainingConfig config,
			int bestEpoch,
			double bestValidationLoss,
			double bestValidationAccuracy,
			double totalSeconds,
			ClassificationMetrics testMetrics) {
			string directory = Path.GetDirectoryName(config.SummaryCsvPath);
			if (!string.IsNullOrEmpty(directory)) {
				Directory.CreateDirectory(directory);
			}

			bool writeHeader = !File.Exists(config.SummaryCsvPath);
			using (StreamWriter writer = new StreamWriter(config.SummaryCsvPath, append: true)) {
				if (writeHeader) {
					writer.WriteLine("hidden,output_lr,hidden_lr,epochs,seed,train_limit,val_limit,best_epoch,best_val_loss,best_val_accuracy,test_accuracy,test_macro_f1,total_seconds");
				}

				string trainLimit = config.TrainLimit.HasValue
					? config.TrainLimit.Value.ToString(CultureInfo.InvariantCulture)
					: string.Empty;
				string validationLimit = config.ValidationLimit.HasValue
					? config.ValidationLimit.Value.ToString(CultureInfo.InvariantCulture)
					: string.Empty;
				string testAccuracy = testMetrics == null
					? string.Empty
					: (testMetrics.Accuracy * 100.0).ToString("F3", CultureInfo.InvariantCulture);
				string testMacroF1 = testMetrics == null
					? string.Empty
					: testMetrics.MacroF1.ToString("F6", CultureInfo.InvariantCulture);

				writer.WriteLine(FormattableString.Invariant(
					$"{config.HiddenCount},{config.OutputLearningRate:F3},{config.HiddenLearningRate:F3},{config.Epochs},{config.Seed},{trainLimit},{validationLimit},{bestEpoch},{bestValidationLoss:F6},{bestValidationAccuracy:F3},{testAccuracy},{testMacroF1},{totalSeconds:F2}"));
			}
		}
	}
}
