package perceptron

type NeuralNetwork struct {
	NumLayers      uint
	NeuronsInLayer []uint
	MinAccuracy    float64
	MaxEpochs      uint
	LearningRate   float64
	Neurons        [][]Neuron
	Epochs         uint
	Accuracy       float64
}

type Neuron struct {
	Weights []float64
	Bias    float64
}

// -----------------------------------------------------------------------------------------------------------------------------------

type NetworkConfig struct {
	NumHiddenLayers int     `bson:"numHiddenLayers" json:"numHiddenLayers"`
	NumNeurons      int     `bson:"numNeurons" json:"numNeurons"`
	LearningRate    float64 `bson:"learningRate" json:"learningRate"`

	SaveDir       string         `bson:"saveDir" json:"saveDir"`
	OutputsArray  []string       `bson:"OutputsArray" json:"OutputsArray"`
	TrainingData  []LearningData `bson:"TrainingData" json:"TrainingData"`
	TestData      []LearningData `bson:"TestData" json:"TestData"`
	CountTraining int            `bson:"countTraining" json:"countTraining"`
	CountGenerate int            `bson:"countGenerate" json:"countGenerate"`
}

type LearningData struct {
	Inputs  []float64 `bson:"inputs" json:"inputs"`
	Targets []float64 `bson:"targets" json:"targets"`
}
