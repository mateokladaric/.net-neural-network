using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

[Serializable]
public class NeuralNetwork
{
    private int[] sizes;
    private List<double[,]> weights;
    private List<double[]> biases;
    private List<double[,]> activations;
    private List<double[,]> preActivations;
    private double learningRate;
    private double l2Lambda;
    private Random rng;

    [JsonConstructor]
    private NeuralNetwork() { }

    public NeuralNetwork(int[] layerSizes, double learningRate = 0.1, double l2Lambda = 0.0001)
    {
        sizes = layerSizes;
        this.learningRate = learningRate;
        this.l2Lambda = l2Lambda;
        rng = new Random();
        weights = new List<double[,]>();
        biases = new List<double[]>();

        for (int i = 0; i < sizes.Length - 1; i++)
        {
            double scale = Math.Sqrt(2.0 / sizes[i]);
            double[,] w = new double[sizes[i], sizes[i + 1]];
            for (int r = 0; r < sizes[i]; r++)
                for (int c = 0; c < sizes[i + 1]; c++)
                    w[r, c] = GaussianRandom() * scale;

            double[] b = new double[sizes[i + 1]];
            weights.Add(w);
            biases.Add(b);
        }
    }

    private double GaussianRandom()
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Clamp(x)));
    private static double SigmoidDerivative(double sig) => sig * (1.0 - sig);
    private static double Clamp(double x, double min = -500, double max = 500) =>
        x < min ? min : x > max ? max : x;

    private static double[,] Softmax(double[,] z)
    {
        int rows = z.GetLength(0), cols = z.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            double max = double.MinValue;
            for (int j = 0; j < cols; j++)
                if (z[i, j] > max) max = z[i, j];

            double sum = 0;
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = Math.Exp(z[i, j] - max);
                sum += result[i, j];
            }
            for (int j = 0; j < cols; j++)
                result[i, j] /= sum;
        }
        return result;
    }

    public double[,] Forward(double[,] input)
    {
        activations = new List<double[,]>();
        preActivations = new List<double[,]>();

        activations.Add(input);

        for (int l = 0; l < weights.Count; l++)
        {
            int batchSize = input.GetLength(0);
            int rows = sizes[l], cols = sizes[l + 1];
            double[,] z = new double[batchSize, cols];

            for (int s = 0; s < batchSize; s++)
                for (int c = 0; c < cols; c++)
                {
                    double sum = biases[l][c];
                    for (int r = 0; r < rows; r++)
                        sum += activations[l][s, r] * weights[l][r, c];
                    z[s, c] = sum;
                }

            preActivations.Add(z);

            if (l == weights.Count - 1)
                activations.Add(Softmax(z));
            else
            {
                double[,] a = new double[batchSize, cols];
                for (int s = 0; s < batchSize; s++)
                    for (int c = 0; c < cols; c++)
                        a[s, c] = Sigmoid(z[s, c]);
                activations.Add(a);
            }
        }

        return activations[activations.Count - 1];
    }

    public void Backward(double[,] targets)
    {
        int L = weights.Count;
        List<double[,]> deltas = new List<double[,]>(L);
        for (int i = 0; i < L; i++)
            deltas.Add(null);

        int batchSize = targets.GetLength(0);

        // output layer
        int last = L - 1;
        double[,] outputA = activations[last + 1];
        double[,] dOut = new double[batchSize, sizes[L]];

        for (int s = 0; s < batchSize; s++)
            for (int c = 0; c < sizes[L]; c++)
                dOut[s, c] = outputA[s, c] - targets[s, c];

        deltas[last] = dOut;

        // hidden layers
        for (int l = last - 1; l >= 0; l--)
        {
            int rows = sizes[l + 1], cols = sizes[l + 2];
            double[,] d = new double[batchSize, rows];

            for (int s = 0; s < batchSize; s++)
                for (int r = 0; r < rows; r++)
                {
                    double sum = 0;
                    for (int c = 0; c < cols; c++)
                        sum += deltas[l + 1][s, c] * weights[l + 1][r, c];
                    d[s, r] = sum * SigmoidDerivative(activations[l + 1][s, r]);
                }

            deltas[l] = d;
        }

        // update weights and biases
        for (int l = 0; l < L; l++)
        {
            int inSize = sizes[l], outSize = sizes[l + 1];

            for (int r = 0; r < inSize; r++)
                for (int c = 0; c < outSize; c++)
                {
                    double grad = 0;
                    for (int s = 0; s < batchSize; s++)
                        grad += activations[l][s, r] * deltas[l][s, c];
                    grad /= batchSize;
                    grad += l2Lambda * weights[l][r, c];
                    weights[l][r, c] -= learningRate * grad;
                }

            for (int c = 0; c < outSize; c++)
            {
                double grad = 0;
                for (int s = 0; s < batchSize; s++)
                    grad += deltas[l][s, c];
                grad /= batchSize;
                biases[l][c] -= learningRate * grad;
            }
        }
    }

    public double Train(double[,] X, double[,] y, int epochs = 10000, bool verbose = false)
    {
        double loss = 0;
        for (int e = 0; e < epochs; e++)
        {
            Forward(X);
            Backward(y);

            if (verbose && e % 1000 == 0)
            {
                loss = CrossEntropyLoss(y);
                Console.WriteLine($"Epoch {e}: loss = {loss:F6}");
            }
        }
        loss = CrossEntropyLoss(y);
        if (verbose) Console.WriteLine($"Final: loss = {loss:F6}");
        return loss;
    }

    public double[,] Predict(double[,] input)
    {
        return Forward(input);
    }

    private double CrossEntropyLoss(double[,] targets)
    {
        double[,] output = activations[activations.Count - 1];
        int rows = targets.GetLength(0), cols = targets.GetLength(1);
        double loss = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (targets[i, j] == 1.0)
                    loss -= Math.Log(output[i, j] + 1e-12);
        return loss / rows;
    }

    public void Save(string name)
    {
        var settings = new JsonSerializerSettings
        {
            Formatting = Formatting.Indented,
            NullValueHandling = NullValueHandling.Ignore
        };
        string json = JsonConvert.SerializeObject(this, settings);
        File.WriteAllText(name + ".nn", json);
    }

    public static NeuralNetwork Load(string name)
    {
        string json = File.ReadAllText(name + ".nn");
        return JsonConvert.DeserializeObject<NeuralNetwork>(json);
    }
}
