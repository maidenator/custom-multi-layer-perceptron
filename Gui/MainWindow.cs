using System;
using System.IO;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;

namespace Backpropagation {
	public class MainWindow : Window {
		private readonly DigitCanvas canvas = new DigitCanvas();
		private readonly TextBlock result = new TextBlock();
		private readonly NeuralNetwork network = new NeuralNetwork();

		public MainWindow() {
			Title = "Digit Recognizer";
			Width = 300;
			Height = 410;
			CanResize = false;

			canvas.Width = 280;
			canvas.Height = 280;

			Button readButton = new Button();
			readButton.Content = "Read";
			readButton.Width = 90;
			readButton.Click += (sender, args) => Read();

			Button clearButton = new Button();
			clearButton.Content = "Clear";
			clearButton.Width = 90;
			clearButton.Click += (sender, args) => canvas.Clear();

			StackPanel buttons = new StackPanel();
			buttons.Orientation = Orientation.Horizontal;
			buttons.Spacing = 10;
			buttons.Children.Add(readButton);
			buttons.Children.Add(clearButton);

			result.FontSize = 16;
			result.FontWeight = FontWeight.Bold;
			result.Text = "Draw a digit, then press Read.";

			StackPanel layout = new StackPanel();
			layout.Margin = new Thickness(10);
			layout.Spacing = 10;
			layout.Children.Add(canvas);
			layout.Children.Add(buttons);
			layout.Children.Add(result);

			Content = layout;

			network.LoadWeights(Path.Combine(AppContext.BaseDirectory, "weights", "epoch-5.txt"));
		}

		private void Read() {
			double[] input = canvas.BuildInput();
			for (int inputIndex = 0; inputIndex < input.Length; inputIndex++) {
				network.SetInput(inputIndex, input[inputIndex]);
			}
			network.ForwardPass();

			int best = 0;
			for (int digit = 1; digit < 10; digit++) {
				if (network.GetOutput(digit) > network.GetOutput(best)) {
					best = digit;
				}
			}

			result.Text = $"Guess: {best}     Confidence: {network.GetOutput(best):P0}";
		}
	}
}
