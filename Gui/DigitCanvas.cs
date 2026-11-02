using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;

namespace Backpropagation {
	public class DigitCanvas : Control {
		private const int GridSize = 28;
		private const int BorderPixels = 2;
		private const int InputSize = GridSize + BorderPixels * 2;
		private const double BrushRadius = 1.8;

		private readonly double[,] pixels = new double[GridSize, GridSize];
		private bool isDrawing;
		private Point lastPoint;

		public void Clear() {
			Array.Clear(pixels, 0, pixels.Length);
			InvalidateVisual();
		}

		public double[] BuildInput() {
			double[] normalized = DigitNormalizer.Normalize(pixels);
			double[] input = new double[InputSize * InputSize];
			for (int y = 0; y < GridSize; y++) {
				for (int x = 0; x < GridSize; x++) {
					input[(y + BorderPixels) * InputSize + (x + BorderPixels)] = normalized[y * GridSize + x];
				}
			}
			return input;
		}

		public override void Render(DrawingContext context) {
			context.FillRectangle(Brushes.Black, new Rect(Bounds.Size));

			double cellWidth = Bounds.Width / GridSize;
			double cellHeight = Bounds.Height / GridSize;

			for (int y = 0; y < GridSize; y++) {
				for (int x = 0; x < GridSize; x++) {
					double value = pixels[y, x];
					if (value <= 0.0) {
						continue;
					}
					byte alpha = (byte)(255.0 * value);
					SolidColorBrush brush = new SolidColorBrush(Color.FromArgb(alpha, 255, 255, 255));
					context.FillRectangle(brush, new Rect(x * cellWidth, y * cellHeight, cellWidth, cellHeight));
				}
			}
		}

		protected override void OnPointerPressed(PointerPressedEventArgs e) {
			if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed) {
				return;
			}

			isDrawing = true;
			lastPoint = ToGrid(e.GetPosition(this));
			PaintDisc(lastPoint);
			e.Pointer.Capture(this);
			e.Handled = true;
		}

		protected override void OnPointerMoved(PointerEventArgs e) {
			if (!isDrawing) {
				return;
			}

			Point current = ToGrid(e.GetPosition(this));
			PaintLine(lastPoint, current);
			lastPoint = current;
		}

		protected override void OnPointerReleased(PointerReleasedEventArgs e) {
			isDrawing = false;
			e.Pointer.Capture(null);
		}

		private Point ToGrid(Point position) {
			double x = position.X / Bounds.Width * GridSize;
			double y = position.Y / Bounds.Height * GridSize;
			return new Point(x, y);
		}

		private void PaintLine(Point from, Point to) {
			double dx = to.X - from.X;
			double dy = to.Y - from.Y;

			int steps = (int)Math.Ceiling(Math.Max(Math.Abs(dx), Math.Abs(dy)));
			if (steps < 1) {
				steps = 1;
			}

			for (int step = 0; step <= steps; step++) {
				PaintDisc(new Point(from.X + dx * step / steps, from.Y + dy * step / steps));
			}
		}

		private void PaintDisc(Point center) {
			for (int y = 0; y < GridSize; y++) {
				for (int x = 0; x < GridSize; x++) {
					double dx = x + 0.5 - center.X;
					double dy = y + 0.5 - center.Y;
					double distance = Math.Sqrt(dx * dx + dy * dy);
					if (distance > BrushRadius) {
						continue;
					}

					double value = 1.0 - distance / BrushRadius;
					if (value > pixels[y, x]) {
						pixels[y, x] = value;
					}
				}
			}
			InvalidateVisual();
		}
	}
}
