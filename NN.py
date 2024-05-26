from dataclasses import dataclass
from xmlrpc.client import MAXINT
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
        print(self.trainingset)
        for i in self.trainingset:
            cordinates = self.dataset[i]
            realcord = [cordinates[j] for j in self.features]
            truedist = euclidean_distance(realthecords, realcord)
            if(truedist < mindist):
                mindist = truedist
                guess = self.dataset[i][0]
        return guess

class Validator:
    def calculate(self, features, classifier, dataset):
        accuracy = [0, 0]
        trainingset = list(range(0, len(dataset)))
        print(features)
        Classi = Classifier(dataset, features)
        n = len(trainingset)
        for i in range(n):
            tobetest = trainingset.pop(0)
            Classi.train(trainingset)
            g = Classi.test(tobetest)
            print(f"{g} and {dataset[tobetest][0]} and {tobetest}")
            if(g == dataset[tobetest][0]):
                accuracy[0] = accuracy[0]+1
            else:
                accuracy[1] = accuracy[1]+1
            trainingset.append(tobetest)

        return accuracy[0]/len(dataset)





def main():

    file_path = '/Users/justincrafty/Downloads/large-test-dataset.txt'
    data = np.loadtxt(file_path)

## Print the resulting matrix
    epic = Validator()
    g = epic.calculate([1,15,27],"NN",data)
    print(g)



if __name__ == "__main__":
    main()






## Load the data from the file
#file_path = '/Users/justincrafty/Downloads/small-test-dataset.txt'
#data = np.loadtxt(file_path)

## Print the resulting matrix
#print(data)
