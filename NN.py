from dataclasses import dataclass
from xmlrpc.client import MAXINT
import time
import numpy as np
import math

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")
    
    distance = math.sqrt(sum((p - q) ** 2 for p, q in zip(point1, point2)))
    return distance

class Classifier:
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features
        self.trainingset = []
    
    def train(self, trainingset):
        self.trainingset = trainingset

    def test(self, id):
        guess = 0
        mindist = 999999999999.9
        thecords = self.dataset[id]
        realthecords = [thecords[j] for j in self.features]
        for i in self.trainingset:
            cordinates = self.dataset[i]
            realcord = [cordinates[j] for j in self.features]
            truedist = euclidean_distance(realthecords, realcord)
            if truedist < mindist:
                mindist = truedist
                guess = self.dataset[i][0]
        return guess

class Validator:
    def calculate(self, features, classifier, dataset):
        accuracy = [0, 0]
        trainingset = list(range(0, len(dataset)))
        Classi = Classifier(dataset, features)
        n = len(trainingset)
        
        for i in range(n):
            start_time = time.time()
            tobetest = trainingset.pop(0)
            train_start_time = time.time()
            Classi.train(trainingset)
            train_end_time = time.time()
            test_start_time = time.time()
            g = Classi.test(tobetest)
            test_end_time = time.time()
            
            #print(f"Training instance {tobetest}: Prediction: {g}, Actual: {dataset[tobetest][0]}, Training time: {train_end_time - train_start_time:.6f}s, Testing time: {test_end_time - test_start_time:.6f}s")
            
            if g == dataset[tobetest][0]:
                accuracy[0] += 1
            else:
                accuracy[1] += 1
            trainingset.append(tobetest)
        
        overall_accuracy = accuracy[0] / len(dataset)
       # print(f"Overall accuracy: {overall_accuracy:.6f}")
        return overall_accuracy

def normalize_dataset(dataset):
    features = dataset[:, 1:]
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    normalized_features = (features - mean) / std
    normalized_dataset = np.hstack((dataset[:, [0]], normalized_features))
    return normalized_dataset

def main():
    #Choosing dataset: Change filepath so it works for your computer
    # Use for Large Data Set
    # file_path = '/Users/austinyang/Desktop/CS170-Project-2/large-test-dataset.txt'
    # Use for Small Data Set
    file_path = '/Users/justincrafty/Documents/CS170/CS170-Project-2/small-test-dataset.txt'
    
    data = np.loadtxt(file_path)

    #Comment this out if you want to use Un-normalized datasetL
    normalized_data = normalize_dataset(data)
     
    validator = Validator()
    validation_start_time = time.time()
    #print(type(data))

    #Change the parameter for small/large dataset
    # Use for Large Data Set for Un-normalized dataset
    # accuracy = validator.calculate([1, 15, 27], "NN", data)

    # Use for Large Data Set for normalized dataset
    # accuracy = validator.calculate([1, 15, 27], "NN", normalized_data)

    # Use for Small Data Set Un-normalized dataset
    accuracy = validator.calculate([3, 5, 7], "NN", data)
    
    # Use for Small Data Set normalized dataset
    # accuracy = validator.calculate([3, 5, 7], "NN", normalized_data)
    

    validation_end_time = time.time()
    print(f"Validation completed in {validation_end_time - validation_start_time:.6f}s")
    print(f"Accuracy: {accuracy:.6f}")

if __name__ == "__main__":
    main()
