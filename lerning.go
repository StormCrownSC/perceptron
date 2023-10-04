package perceptron

import (
	"fmt"
	"strconv"
	"time"
)

func networkLearning(nn *NeuralNetwork, generateConf NetworkConfig, isTraining bool) bool {
	bestNeuralNetwork := copyNeuralNetwork(nn)

	for epoch := uint(1); epoch <= nn.MaxEpochs; epoch++ {
		nn.Epochs = epoch
		correctPredictions := 0

		for _, data := range generateConf.TrainingData {
			nn.BackPropagation(data.Inputs, data.Targets)

			// Calculate predicted values
			outputs := nn.ForwardPropagation(data.Inputs)

			predictedLabel := predict(outputs, generateConf.OutputsArray)
			actualLabel := predict(data.Targets, generateConf.OutputsArray)

			// Compare predicted values with actual
			if predictedLabel == actualLabel {
				correctPredictions++
			}
		}

		accuracy := float64(correctPredictions) / float64(len(generateConf.TrainingData))
		nn.Accuracy = accuracy

		if accuracy > bestNeuralNetwork.Accuracy {
			bestNeuralNetwork = copyNeuralNetwork(nn)
		}

		if accuracy >= nn.MinAccuracy {
			break
		}

		if epoch-bestNeuralNetwork.Epochs > 100 && accuracy-bestNeuralNetwork.Accuracy <= 1 {
			break
		}
	}

	if isTraining == false || bestNeuralNetwork.Accuracy >= bestOfSaves(generateConf.SaveDir) {
		testAccuracy := testNeuralNetwork(&bestNeuralNetwork, generateConf)
		trainAccuracyStr := strconv.FormatFloat(bestNeuralNetwork.Accuracy*100, 'f', 2, 64)
		testAccuracyStr := strconv.FormatFloat(testAccuracy*100, 'f', 2, 64)

		datetime := time.Now().Format("02-01-2006_15-04-05")
		filename := fmt.Sprintf("%s_%f_%d_%d_TrainAccuracy_%s%%_TestAccuracy_%s%%_%d_epochs.gob", datetime, bestNeuralNetwork.LearningRate,
			bestNeuralNetwork.NumLayers, bestNeuralNetwork.NeuronsInLayer[1], trainAccuracyStr, testAccuracyStr, bestNeuralNetwork.Epochs)
		save(generateConf.SaveDir+filename, &bestNeuralNetwork)
		if isTraining == true && bestNeuralNetwork.Accuracy >= nn.MinAccuracy && testAccuracy >= nn.MinAccuracy {
			return true
		}
	}
	return false
}

func learningPrepare(conf NetworkConfig, isTraining bool) {
	NeuronsInLayer := make([]uint, conf.NumHiddenLayers+2)
	NeuronsInLayer[0] = uint(len(conf.TrainingData[0].Inputs))
	NeuronsInLayer[conf.NumHiddenLayers+1] = uint(len(conf.TrainingData[0].Targets))
	for i := 1; int64(i) <= conf.NumHiddenLayers; i++ {
		NeuronsInLayer[i] = uint(conf.NumNeurons)
	}

	var count int
	if isTraining {
		count = conf.CountTraining
	} else {
		count = conf.CountGenerate
	}

	for i := 0; i < count; i++ {
		nn := &NeuralNetwork{
			NeuronsInLayer: NeuronsInLayer,
			LearningRate:   conf.LearningRate,
		}
		nn.Initialize()

		networkLearning(nn, conf, isTraining)
	}
}

func testNeuralNetwork(nn *NeuralNetwork, trainConf NetworkConfig) float64 {
	correctPredictions := 0

	for _, data := range trainConf.TestData {
		outputs := nn.ForwardPropagation(data.Inputs)

		predictedLabel := predict(outputs, trainConf.OutputsArray)
		actualLabel := predict(data.Targets, trainConf.OutputsArray)

		// Compare predicted values with actual
		if predictedLabel == actualLabel {
			correctPredictions++
		}

	}
	totalAccuracy := float64(correctPredictions) / float64(len(trainConf.TestData))
	return totalAccuracy
}
