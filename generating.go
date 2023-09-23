package perceptron

func GenerateStructure(learningRates []float64, countHiddenLayers []int64, countNeurons []int64) {
	var conf NetworkConfig
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
