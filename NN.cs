[Serializable]
public class NeuralNetwork
{
	public static double Sigmoid(double x)
	{
		return 1 / (1 + Math.Exp(-x));
	}

	public static double SigmoidDerivative(double x)
	{
		return x * (1 - x);
	}

	public static double[,] DotProduct(double[,] A, double[,] B)
	{
		int rowsA = A.GetLength(0);
		int colsA = A.GetLength(1);
		int colsB = B.GetLength(1);
		double[,] result = new double[rowsA, colsB];

		for (int i = 0; i < rowsA; i++)
		{
			for (int j = 0; j < colsB; j++)
			{
				for (int k = 0; k < colsA; k++)
				{
					result[i, j] += A[i, k] * B[k, j];
				}
			}
		}
		return result;
	}

	public static double[,] Transpose(double[,] matrix)
	{
		int rows = matrix.GetLength(0);
		int cols = matrix.GetLength(1);
		double[,] transposed = new double[cols, rows];

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				transposed[j, i] = matrix[i, j];
			}
		}
		return transposed;
	}

	public static double[,] AddMatrices(double[,] A, double[,] B)
	{
		int rows = A.GetLength(0);
		int cols = A.GetLength(1);
		double[,] result = new double[rows, cols];

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				result[i, j] = A[i, j] + B[i, j];
			}
		}
		return result;
	}

	public static double[,] ScalarMultiply(double[,] matrix, double scalar)
	{
		int rows = matrix.GetLength(0);
		int cols = matrix.GetLength(1);
		double[,] result = new double[rows, cols];

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				result[i, j] = scalar * matrix[i, j];
			}
		}
		return result;
	}

	public static double[,] ElementwiseMultiply(double[,] A, double[,] B)
	{
		int rows = A.GetLength(0);
		int cols = A.GetLength(1);
		double[,] result = new double[rows, cols];

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				result[i, j] = A[i, j] * B[i, j];
			}
		}
		return result;
	}

	[JsonProperty]
	private int numLayers { get; set; }
	[JsonProperty]
	private int[] layerSizes { get; set; }
	[JsonProperty]
	private List<double[,]> weights { get; set; }
	[JsonProperty]
	private List<double[,]> activations { get; set; }
	[JsonProperty]
	private List<double[,]> zValues { get; set; }
	[JsonProperty]
	private List<double[,]> errors { get; set; }
	[JsonProperty]
	private List<double[,]> deltas { get; set; }

	public NeuralNetwork(params int[] layerSizes)
	{
		this.numLayers = layerSizes.Length;
		this.layerSizes = layerSizes;
		this.weights = new List<double[,]>();

		Random rand = new Random();
		for (int i = 0; i < numLayers - 1; i++)
		{
			double[,] layerWeights = new double[layerSizes[i], layerSizes[i + 1]];
			for (int j = 0; j < layerSizes[i]; j++)
			{
				for (int k = 0; k < layerSizes[i + 1]; k++)
				{
					layerWeights[j, k] = rand.NextDouble() * 2 - 1;
				}
			}
			this.weights.Add(layerWeights);
		}
	}

	public double[,] Forward(double[,] X)
	{
		activations = new List<double[,]>();
		zValues = new List<double[,]>();
		activations.Add(X);

		for (int i = 0; i < numLayers - 1; i++)
		{
			double[,] z = DotProduct(activations[i], weights[i]);
			zValues.Add(z);

			double[,] activation = new double[z.GetLength(0), z.GetLength(1)];
			for (int j = 0; j < z.GetLength(0); j++)
			{
				for (int k = 0; k < z.GetLength(1); k++)
				{
					activation[j, k] = Sigmoid(z[j, k]);
				}
			}
			activations.Add(activation);
		}

		return activations[activations.Count - 1];
	}

	public void Backward(double[,] correctOutputs, double[,] lastOutput)
	{
		errors = new List<double[,]>();
		deltas = new List<double[,]>();

		double[,] outputError = new double[correctOutputs.GetLength(0), correctOutputs.GetLength(1)];
		for (int sample = 0; sample < correctOutputs.GetLength(0); sample++)
		{
			for (int outputIndex = 0; outputIndex < correctOutputs.GetLength(1); outputIndex++)
			{
				outputError[sample, outputIndex] = correctOutputs[sample, outputIndex] - lastOutput[sample, outputIndex];
			}
		}
		errors.Add(outputError);

		int numberOfSamples = lastOutput.GetLength(0);
		int numberOfOutputs = lastOutput.GetLength(1);
		double[,] outputDelta = new double[numberOfSamples, numberOfOutputs];
		for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
		{
			for (int outputIndex = 0; outputIndex < numberOfOutputs; outputIndex++)
			{
				outputDelta[sampleIndex, outputIndex] = outputError[sampleIndex, outputIndex] * SigmoidDerivative(lastOutput[sampleIndex, outputIndex]);
			}
		}
		deltas.Add(outputDelta);

		for (int layerIndex = numLayers - 2; layerIndex > 0; layerIndex--)
		{
			double[,] zError = DotProduct(deltas[0], Transpose(weights[layerIndex]));
			errors.Insert(0, zError);

			double[,] zDelta = new double[activations[layerIndex].GetLength(0), activations[layerIndex].GetLength(1)];
			for (int j = 0; j < activations[layerIndex].GetLength(0); j++)
			{
				for (int k = 0; k < activations[layerIndex].GetLength(1); k++)
				{
					zDelta[j, k] = zError[j, k] * SigmoidDerivative(activations[layerIndex][j, k]);
				}
			}
			deltas.Insert(0, zDelta);
		}

		for (int i = 0; i < numLayers - 1; i++)
		{
			double[,] deltaWeights = DotProduct(Transpose(activations[i]), deltas[i]);
			weights[i] = AddMatrices(weights[i], deltaWeights);
		}
	}


	public void Train(double[,] X, double[,] y, int epochs = 10000)
	{
		for (int epoch = 0; epoch < epochs; epoch++)
		{
			double[,] output = Forward(X);
			Backward(y, output);
		}
	}

	public void Save(string name)
	{
		JsonSerializerSettings settings = new JsonSerializerSettings
		{
			Formatting = Formatting.Indented,
			NullValueHandling = NullValueHandling.Ignore,
			DefaultValueHandling = DefaultValueHandling.Ignore
		};
		string json = JsonConvert.SerializeObject(this, settings);
		System.IO.File.WriteAllText(name + ".nn", json);
	}

	public NeuralNetwork Load(string name)
	{
		string json = System.IO.File.ReadAllText(name + ".nn");
		return JsonConvert.DeserializeObject<NeuralNetwork>(json);
	}
}
