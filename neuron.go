package perceptron

import (
	"log"
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func (nn *NeuralNetwork) Initialize() {
	nn.NumLayers = uint(len(nn.NeuronsInLayer))

	if nn.NumLayers == 0 {
		log.Fatal("NumLayers and NeuronsInLayer must be specified")
	}

	if nn.MinAccuracy == 0 {
		nn.MinAccuracy = 0.7 // default value for MinAccuracy
	}

	if nn.MaxEpochs == 0 {
		nn.MaxEpochs = 10000 // default value for maxEpochs
	}

	if nn.LearningRate == 0 {
		nn.LearningRate = 0.001 // default value for learningRate
	}

	nn.Epochs = 0
	nn.Accuracy = 0

	nn.Neurons = make([][]Neuron, nn.NumLayers)

	// Creating a slice to store the dimensions of the layers (the number of neurons in each layer)
	layerSizes := make([]uint, nn.NumLayers)
	copy(layerSizes, nn.NeuronsInLayer)

	for i := range nn.Neurons {
		nn.Neurons[i] = make([]Neuron, layerSizes[i])

		for j := range nn.Neurons[i] {
			if i != 0 {
				nn.Neurons[i][j] = Neuron{
					Weights: make([]float64, layerSizes[i-1]),
					Bias:    0.0,
				}
			} else {
				nn.Neurons[i][j] = Neuron{
					Weights: make([]float64, 1),
					Bias:    0.0,
				}
			}

			for k := range nn.Neurons[i][j].Weights {
				// Initialize weights randomly between -1 and 1
				if i > 0 {
					nn.Neurons[i][j].Weights[k] = (rand.Float64() * 2) - 1
				}
			}
		}
	}
}

func (nn *NeuralNetwork) ForwardPropagation(inputs []float64) []float64 {
	outputs := make([]float64, len(nn.Neurons[nn.NumLayers-1]))

	// Set inputs to the first layer
	for i := range nn.Neurons[0] {
		nn.Neurons[0][i].Bias = inputs[i]
	}

	// Calculate outputs for each layer
	for i := 1; i < int(nn.NumLayers); i++ {
		for j := range nn.Neurons[i] {
			weightedSum := nn.Neurons[i][j].Bias

			for k := range nn.Neurons[i][j].Weights {
				weightedSum += nn.Neurons[i][j].Weights[k] * nn.Neurons[i-1][k].Bias
			}

			nn.Neurons[i][j].Bias = sigmoid(weightedSum)
			if i == int(nn.NumLayers)-1 {
				outputs[j] = nn.Neurons[i][j].Bias
			}
		}
	}

	return outputs
}

func (nn *NeuralNetwork) BackPropagation(inputs []float64, targets []float64) {
	outputs := nn.ForwardPropagation(inputs)

	// Calculate the output layer deltas
	outputLayerIdx := int(nn.NumLayers) - 1
	for i := range nn.Neurons[outputLayerIdx] {
		outputDelta := (targets[i] - outputs[i]) * sigmoidDerivative(outputs[i])

		for j := range nn.Neurons[outputLayerIdx][i].Weights {
			nn.Neurons[outputLayerIdx][i].Weights[j] += nn.LearningRate * outputDelta * nn.Neurons[outputLayerIdx-1][j].Bias
		}

		nn.Neurons[outputLayerIdx][i].Bias += nn.LearningRate * outputDelta
	}

	// Calculate deltas for the hidden layers
	for i := int(nn.NumLayers) - 2; i > 0; i-- {
		for j := range nn.Neurons[i] {
			hiddenDelta := 0.0

			// Calculate the weighted sum of gradients in the next layer
			for k := 0; k < len(nn.Neurons[i+1]); k++ {
				// Calculate the contribution of each neuron in the output layer
				hiddenDelta += nn.Neurons[i+1][k].Weights[j] * sigmoidDerivative(nn.Neurons[i+1][k].Bias) * (targets[0] - outputs[0])
			}

			hiddenDelta *= sigmoidDerivative(nn.Neurons[i][j].Bias)

			for k := range nn.Neurons[i][j].Weights {
				nn.Neurons[i][j].Weights[k] += nn.LearningRate * hiddenDelta * nn.Neurons[i-1][k].Bias
			}

			nn.Neurons[i][j].Bias += nn.LearningRate * hiddenDelta
		}
	}
}
