using System;
using System.Globalization;
using System.Text;

namespace Backpropagation {
	/// <summary>
	/// Multi-class evaluation results: accuracy, per-class precision/recall/F1,
	/// and a confusion matrix indexed as [actual, predicted].
	/// </summary>
	public sealed class ClassificationMetrics {
		private readonly int[,] confusion;
		private readonly double[] precision;
		private readonly double[] recall;
		private readonly double[] f1;
		private readonly int[] support;

		public int ClassCount { get; }
		public double Accuracy { get; }
		public double MacroPrecision { get; }
		public double MacroRecall { get; }
		public double MacroF1 { get; }

		private ClassificationMetrics(
			int[,] confusion,
			double[] precision,
			double[] recall,
			double[] f1,
			int[] support,
			double accuracy,
			double macroPrecision,
			double macroRecall,
			double macroF1) {
			this.confusion = confusion;
			this.precision = precision;
			this.recall = recall;
			this.f1 = f1;
			this.support = support;
			ClassCount = support.Length;
			Accuracy = accuracy;
			MacroPrecision = macroPrecision;
			MacroRecall = macroRecall;
			MacroF1 = macroF1;
		}

		public int[,] GetConfusionMatrix() {
			return confusion;
		}

		public static ClassificationMetrics Create(int[] actual, int[] predicted, int classCount) {
			if (actual.Length != predicted.Length) {
				throw new ArgumentException("Actual and predicted label counts must match.");
			}

			int[,] confusion = new int[classCount, classCount];
			for (int index = 0; index < actual.Length; index++) {
				confusion[actual[index], predicted[index]]++;
			}

			double[] precision = new double[classCount];
			double[] recall = new double[classCount];
			double[] f1 = new double[classCount];
			int[] support = new int[classCount];
			int correct = 0;

			for (int classIndex = 0; classIndex < classCount; classIndex++) {
				int truePositives = confusion[classIndex, classIndex];
				int actualCount = 0;
				int predictedCount = 0;

				for (int other = 0; other < classCount; other++) {
					actualCount += confusion[classIndex, other];
					predictedCount += confusion[other, classIndex];
				}

				support[classIndex] = actualCount;
				correct += truePositives;
				precision[classIndex] = predictedCount == 0 ? 0.0 : (double)truePositives / predictedCount;
				recall[classIndex] = actualCount == 0 ? 0.0 : (double)truePositives / actualCount;

				double denominator = precision[classIndex] + recall[classIndex];
				f1[classIndex] = denominator == 0.0
					? 0.0
					: 2.0 * precision[classIndex] * recall[classIndex] / denominator;
			}

			double macroPrecision = 0.0;
			double macroRecall = 0.0;
			double macroF1 = 0.0;
			for (int classIndex = 0; classIndex < classCount; classIndex++) {
				macroPrecision += precision[classIndex];
				macroRecall += recall[classIndex];
				macroF1 += f1[classIndex];
			}
			macroPrecision /= classCount;
			macroRecall /= classCount;
			macroF1 /= classCount;

			double accuracy = actual.Length == 0 ? 0.0 : (double)correct / actual.Length;
			return new ClassificationMetrics(confusion, precision, recall, f1, support, accuracy, macroPrecision, macroRecall, macroF1);
		}

		/// <summary>Formats the per-class metrics as a fixed-width console table.</summary>
		public string FormatTable() {
			StringBuilder builder = new StringBuilder();
			builder.AppendLine("class  precision  recall      f1  support");

			for (int classIndex = 0; classIndex < ClassCount; classIndex++) {
				builder.AppendLine(FormattableString.Invariant(
					$"{classIndex,5}  {precision[classIndex],9:F3}  {recall[classIndex],6:F3}  {f1[classIndex],6:F3}  {support[classIndex],7}"));
			}

			builder.AppendLine(FormattableString.Invariant(
				$"macro  {MacroPrecision,9:F3}  {MacroRecall,6:F3}  {MacroF1,6:F3}  {TotalSupport,7}"));
			builder.AppendLine(FormattableString.Invariant($"accuracy {Accuracy * 100.0:F2}%"));
			return builder.ToString();
		}

		public string ToMetricsCsv() {
			StringBuilder builder = new StringBuilder();
			builder.AppendLine("class,precision,recall,f1,support");

			for (int classIndex = 0; classIndex < ClassCount; classIndex++) {
				builder.AppendLine(FormattableString.Invariant(
					$"{classIndex},{precision[classIndex]:F6},{recall[classIndex]:F6},{f1[classIndex]:F6},{support[classIndex]}"));
			}

			builder.AppendLine(FormattableString.Invariant(
				$"macro,{MacroPrecision:F6},{MacroRecall:F6},{MacroF1:F6},{TotalSupport}"));
			return builder.ToString();
		}

		public string ToConfusionCsv() {
			StringBuilder builder = new StringBuilder();
			builder.Append("actual\\predicted");
			for (int classIndex = 0; classIndex < ClassCount; classIndex++) {
				builder.Append(',');
				builder.Append(classIndex.ToString(CultureInfo.InvariantCulture));
			}
			builder.AppendLine();

			for (int actual = 0; actual < ClassCount; actual++) {
				builder.Append(actual.ToString(CultureInfo.InvariantCulture));
				for (int predicted = 0; predicted < ClassCount; predicted++) {
					builder.Append(',');
					builder.Append(confusion[actual, predicted].ToString(CultureInfo.InvariantCulture));
				}
				builder.AppendLine();
			}
			return builder.ToString();
		}

		private int TotalSupport {
			get {
				int total = 0;
				for (int classIndex = 0; classIndex < support.Length; classIndex++) {
					total += support[classIndex];
				}
				return total;
			}
		}
	}
}
