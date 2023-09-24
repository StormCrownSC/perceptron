package perceptron

func GenerateStructure(conf NetworkConfig, learningRates []float64, countHiddenLayers []int64, countNeurons []int64) {
	for numHiddenLayers := range countHiddenLayers {
		conf.NumHiddenLayers = numHiddenLayers
		for numNeurons := range countNeurons {
			conf.NumNeurons = numNeurons
			for _, lr := range learningRates {
				conf.LearningRate = lr
				learningPrepare(conf, false)
			}
		}
	}
}
