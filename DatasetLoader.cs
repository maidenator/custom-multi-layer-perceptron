using System.Buffers.Binary;
using System.IO;

namespace Backpropagation {
	/// <summary>
	/// Reads the MNIST IDX files and turns them into network-ready input vectors.
	/// Each image stays at its native 28x28 and is normalized to the range 0..1.
	/// </summary>
	public static class DatasetLoader {
		private const int ImageSize = 28;
		private const int TargetSize = 32;
		private const int Padding = (TargetSize - ImageSize) / 2;

		public static (double[][] inputs, int[] labels) Load(string imagesPath, string labelsPath) {
			double[][] inputs;
			using (FileStream imageFile = File.OpenRead(imagesPath))
			using (BinaryReader imageReader = new BinaryReader(imageFile)) {
				ReadBigEndianInt(imageReader); // magic number
				int imageCount = ReadBigEndianInt(imageReader);
				ReadBigEndianInt(imageReader); // rows
				ReadBigEndianInt(imageReader); // cols

				inputs = new double[imageCount][];
				for (int i = 0; i < imageCount; i++) {
					inputs[i] = PadAndNormalize(imageReader.ReadBytes(ImageSize * ImageSize));
				}
			}

			int[] labels;
			using (FileStream labelFile = File.OpenRead(labelsPath))
			using (BinaryReader labelReader = new BinaryReader(labelFile)) {
				ReadBigEndianInt(labelReader); // magic number
				int labelCount = ReadBigEndianInt(labelReader);

				byte[] rawLabels = labelReader.ReadBytes(labelCount);
				labels = new int[labelCount];
				for (int i = 0; i < labelCount; i++) {
					labels[i] = rawLabels[i];
				}
			}

			return (inputs, labels);
		}

		private static int ReadBigEndianInt(BinaryReader reader) {
			return BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));
		}

		private static double[] PadAndNormalize(byte[] pixels) {
			double[] result = new double[TargetSize * TargetSize];
			for (int y = 0; y < ImageSize; y++) {
				for (int x = 0; x < ImageSize; x++) {
					int targetIndex = (y + Padding) * TargetSize + (x + Padding);
					result[targetIndex] = pixels[y * ImageSize + x] / 255.0;
				}
			}
			return result;
		}
	}
}
