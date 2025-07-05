class LinearClassifier:
    
    def __init__(self, weight, learningRate):
        
        self.__weight = weight
        self.__learningRate = learningRate

    def query(self, input):
        
        return self.__weight * input

    def train(self, input, target):
        
        output = self.__weight * input
        error = target - output 
        delta = self.__learningRate * (error / input)
        self.__weight += delta
