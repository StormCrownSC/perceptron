package perceptron

import (
	"encoding/gob"
	"errors"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
)

func copyNeuralNetwork(nn *NeuralNetwork) NeuralNetwork {
	// Create a new NeuralNetwork object
	newNN := NeuralNetwork{
		NumLayers:      nn.NumLayers,
		NeuronsInLayer: make([]uint, len(nn.NeuronsInLayer)),
		MinAccuracy:    nn.MinAccuracy,
		MaxEpochs:      nn.MaxEpochs,
		LearningRate:   nn.LearningRate,
		Neurons:        make([][]Neuron, len(nn.Neurons)),
		Epochs:         nn.Epochs,
		Accuracy:       nn.Accuracy,
	}

	// Copy NeuronsInLayer array values
	copy(newNN.NeuronsInLayer, nn.NeuronsInLayer)

	// Copy values of each neuron and their weights in layers
	for i := range nn.Neurons {
		newNN.Neurons[i] = make([]Neuron, len(nn.Neurons[i]))
		for j := range nn.Neurons[i] {
			newNN.Neurons[i][j] = Neuron{
				Weights: make([]float64, len(nn.Neurons[i][j].Weights)),
				Bias:    nn.Neurons[i][j].Bias,
			}
			copy(newNN.Neurons[i][j].Weights, nn.Neurons[i][j].Weights)
		}
	}

	return newNN
}

// save saves the state of the neural network to a file.
func save(filename string, nn *NeuralNetwork) error {
	// Check that the file does not exist.
	if _, err := os.Stat(filename); err == nil {
		return errors.New("file already exists")
	}

	dir := filepath.Dir(filename)
	err := checkDirectory(dir)
	if err != nil {
		log.Fatal(err)
		return err
	}

	// Creating a file to save the state of the neural network.
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Creating a Gob encoder to serialize the NeuralNetwork structure.
	encoder := gob.NewEncoder(file)

	// Serialize the Neural Network state to a file.
	if err := encoder.Encode(nn); err != nil {
		log.Println(err)
		return err
	}
	return nil
}

func checkDirectory(directory string) error {
	// Checking the presence of a folder
	if _, err := os.Stat(directory); os.IsNotExist(err) {
		// Creating a folder
		err := os.Mkdir(directory, 0755)
		if err != nil {
			return err
		}
	}
	return nil
}

func bestOfSaves(dirPath string) float64 {
	fileInfos, err := os.ReadDir(dirPath)
	bestAccuracy := 0.1

	if err != nil {
		return bestAccuracy
	}

	for _, fileInfo := range fileInfos {
		if !fileInfo.IsDir() {
			num := extractionAccuracy(fileInfo.Name())
			if num > bestAccuracy {
				bestAccuracy = num
			}

		}
	}

	return bestAccuracy
}

func extractionAccuracy(fileName string) float64 {
	// Define a regular expression to extract floating-point number from the file name
	re := regexp.MustCompile(`TrainAccuracy_([\d.]+)%`)
	matches := re.FindStringSubmatch(fileName)

	if len(matches) < 2 {
		return 0.1
	}

	// Extract the found number and convert it to float64
	accuracyStr := matches[1]
	accuracy, err := strconv.ParseFloat(accuracyStr, 64)
	if err != nil {
		return 0.1
	}

	return accuracy / 100.0

}

func predict(outputs []float64, dataOutputs []string) string {
	if len(outputs) > 1 {
		// Initialize variable to store maximum value
		_max := -math.MaxFloat64
		number := 0

		// Iterate through the array and compare elements
		for i, value := range outputs {
			if value > _max {
				_max = value
				number = i
			}
		}
		return dataOutputs[number]
	} else {
		log.Fatal("Outputs must be more one")
		return "Outputs must be more one"
	}
}
