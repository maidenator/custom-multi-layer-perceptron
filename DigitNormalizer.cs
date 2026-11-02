using System;

namespace Backpropagation {
	/// <summary>
	/// Reproduces the preprocessing MNIST applied to its own images: crop to the
	/// ink, scale the longest side to 20 pixels, then centre by centre of mass.
	/// Without this, hand-drawn input is out of distribution for the network.
	/// </summary>
	public static class DigitNormalizer {
		private const int GridSize = 28;
		private const int DigitSize = 20;
		private const double InkThreshold = 0.05;

		public static double[] Normalize(double[,] pixels) {
			double[] result = new double[GridSize * GridSize];

			int minX = GridSize;
			int minY = GridSize;
			int maxX = -1;
			int maxY = -1;

			for (int y = 0; y < GridSize; y++) {
				for (int x = 0; x < GridSize; x++) {
					if (pixels[y, x] <= InkThreshold) {
						continue;
					}
					if (x < minX) minX = x;
					if (x > maxX) maxX = x;
					if (y < minY) minY = y;
					if (y > maxY) maxY = y;
				}
			}

			if (maxX < 0) {
				return result;
			}

			int width = maxX - minX + 1;
			int height = maxY - minY + 1;
			double scale = (double)DigitSize / Math.Max(width, height);

			int scaledWidth = Math.Max(1, (int)Math.Round(width * scale));
			int scaledHeight = Math.Max(1, (int)Math.Round(height * scale));

			double[] cropped = new double[scaledWidth * scaledHeight];
			double mass = 0.0;
			double massX = 0.0;
			double massY = 0.0;

			for (int y = 0; y < scaledHeight; y++) {
				for (int x = 0; x < scaledWidth; x++) {
					double sourceX = minX + (x + 0.5) / scale - 0.5;
					double sourceY = minY + (y + 0.5) / scale - 0.5;
					double value = SampleBilinear(pixels, sourceX, sourceY);

					cropped[y * scaledWidth + x] = value;
					mass += value;
					massX += value * x;
					massY += value * y;
				}
			}

			if (mass <= 0.0) {
				return result;
			}

			double centerX = massX / mass;
			double centerY = massY / mass;
			int offsetX = (int)Math.Round(GridSize / 2.0 - centerX);
			int offsetY = (int)Math.Round(GridSize / 2.0 - centerY);

			for (int y = 0; y < scaledHeight; y++) {
				for (int x = 0; x < scaledWidth; x++) {
					int targetX = x + offsetX;
					int targetY = y + offsetY;
					if (targetX < 0 || targetX >= GridSize || targetY < 0 || targetY >= GridSize) {
						continue;
					}
					result[targetY * GridSize + targetX] = cropped[y * scaledWidth + x];
				}
			}

			return result;
		}

		private static double SampleBilinear(double[,] pixels, double x, double y) {
			int x0 = (int)Math.Floor(x);
			int y0 = (int)Math.Floor(y);
			double fx = x - x0;
			double fy = y - y0;

			double topLeft = PixelAt(pixels, x0, y0);
			double topRight = PixelAt(pixels, x0 + 1, y0);
			double bottomLeft = PixelAt(pixels, x0, y0 + 1);
			double bottomRight = PixelAt(pixels, x0 + 1, y0 + 1);

			return topLeft * (1.0 - fx) * (1.0 - fy)
				+ topRight * fx * (1.0 - fy)
				+ bottomLeft * (1.0 - fx) * fy
				+ bottomRight * fx * fy;
		}

		private static double PixelAt(double[,] pixels, int x, int y) {
			if (x < 0 || x >= GridSize || y < 0 || y >= GridSize) {
				return 0.0;
			}
			return pixels[y, x];
		}
	}
}
