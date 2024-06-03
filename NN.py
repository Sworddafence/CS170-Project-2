from dataclasses import dataclass
from pyexpat import features
from xmlrpc.client import MAXINT
from collections import Counter
import time
import numpy as np
import math
import heapq as hq

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
        self.k = 0
    
    def train(self, trainingset):
        self.trainingset = trainingset

    def testNN(self, id):
        guess = 0
        mindist = float('inf')
        thecords = self.dataset[id]
        realthecords = [thecords[j] for j in self.features]
        for i in self.trainingset:
            cordinates = self.dataset[i]
            realcord = [cordinates[j] for j in self.features]
            #print(realthecords)
            #print(realcord)
            truedist = euclidean_distance(realthecords, realcord)
            if truedist < mindist:
                mindist = truedist
                guess = self.dataset[i][0]
        return guess

    def testKNN(self, id):        
        guess = 0
        distances = []

        thecords = self.dataset[id]
        realthecords = [thecords[j] for j in self.features]
        for i in self.trainingset:
            cordinates = self.dataset[i]
            realcord = [cordinates[j] for j in self.features]
            truedist = euclidean_distance(realthecords, realcord)
            #print(cordinates)
            hq.heappush(distances, (truedist, cordinates[0]))

        self.k = int(self.k)
        b = hq.nsmallest(self.k, distances)
        g = Counter(i[1] for i in b)
        mos, _ = g.most_common(1)[0]
        #print(mos)
        return mos 
    
    def testNB(self, id):
        guess = 0
        odd1out = self.dataset[id]
        trainingset = self.dataset[self.trainingset]
        trainingfeatures = trainingset[:, 1:]
        trainClasses = trainingset[:, 0]
        classes = np.unique(trainClasses)


        numClasses = len(classes)
        numFeatures = len(trainingset[0])
        mean = np.zeros((numClasses, numFeatures-1), dtype=np.float64)
        varience = np.zeros((numClasses, numFeatures-1), dtype=np.float64)
        priors = np.zeros(numClasses, dtype=np.float64)

        for i, j in enumerate(classes):
            trainFeatures = trainingfeatures[trainClasses == j]
            mean[i, :] = trainFeatures.mean(axis=0)
            varience[i, :] = trainFeatures.var(axis=0)
            priors[i] = trainFeatures.shape[0] / float(len(trainingset))

        if(self.features == set()):
            return classes[np.argmax(priors)]
        probability = [0,0]
        for x in self.features:
            for i, c in enumerate(classes):
                

                tempnumerator = np.exp(- (odd1out[x] - mean[i]) ** 2 / (2 * varience[i]))
                tempdenominator = np.sqrt(2 * np.pi * varience[i])

                class_conditional = np.sum(np.log(tempnumerator/tempdenominator))
                
                probability[int(i)] = probability[int(i)] + class_conditional
                #print(c)
            
            #print(probability)
        for i in range(len(classes)):
            probability[i] = probability[i] + priors[i]
        return classes[np.argmax(probability)]


class Validator:
    def calculate(self, features, dataset, classifier, k=None):
        accuracy = [0, 0]
        trainingset = list(range(0, len(dataset)))
        Classi = Classifier(dataset, features)
        n = len(trainingset)
        if k is not None:
            Classi.k = k

        for i in range(n):
            start_time = time.time()
            tobetest = trainingset.pop(0)
            train_start_time = time.time()
            Classi.train(trainingset)
            train_end_time = time.time()
            test_start_time = time.time()
            
            if(classifier == "NN"):
                g = Classi.testNN(tobetest)
            elif(classifier == "NB"):
                g = Classi.testNB(tobetest)
            elif(classifier == "KNN"):
                g = Classi.testKNN(tobetest)
            test_end_time = time.time()
            
            #print(f"Training instance {tobetest}: Prediction: {g}, Actual: {dataset[tobetest][0]}, Training time: {train_end_time - train_start_time:.6f}s, Testing time: {test_end_time - test_start_time:.6f}s")
            
            if g == dataset[tobetest][0]:
                accuracy[0] += 1
            else:
                accuracy[1] += 1
            trainingset.append(tobetest)
        
        overall_accuracy = accuracy[0] / len(dataset)
        #print(f"Overall accuracy: {overall_accuracy:.6f}")
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
    print("Select which dataset to use: ")
    data_num = int(input("Type '1' for small dataset. Type '2' for large dataset: "))

    if data_num == 1:
        file_path = 'CS170_Spring_2024_Small_data__69.txt'
        data = np.loadtxt(file_path)

    elif data_num == 2:
        file_path = 'CS170_Spring_2024_Large_data__69.txt'
        data = np.loadtxt(file_path)
    else:
        print("Wrong input")

    print("Do you want to use Un-normalized dataset or Normalized dataset")
    normalize_num  = int(input("Type '1' for Un-normalized dataset or Type '2' for Normalized dataset: "))

    #Comment this out if you want to use Un-normalized dataset
    if normalize_num == 2:
        normalized_data = normalize_dataset(data)
    

    validator = Validator()
    validation_start_time = time.time()

    #Change the parameter for small/large dataset
    # Use for Large Data Set for Un-normalized dataset
    if data_num == 2 and normalize_num == 1:
        accuracy = validator.calculate([1, 15, 27], data, "NN")
        validation_end_time = time.time()
        print(f"Validation completed in {validation_end_time - validation_start_time:.6f}s")
        print(f"Accuracy: {accuracy:.6f}")

    # Use for Large Data Set for normalized dataset
    elif data_num == 2 and normalize_num == 2:
        accuracy = validator.calculate([1, 15, 27], normalized_data, "NN")
        validation_end_time = time.time()
        print(f"Validation completed in {validation_end_time - validation_start_time:.6f}s")
        print(f"Accuracy: {accuracy:.6f}")

    # Use for Small Data Set Un-normalized dataset
    elif data_num == 1 and normalize_num == 1:
        accuracy = validator.calculate([3, 5, 7], data, "NN")
        validation_end_time = time.time()
        print(f"Validation completed in {validation_end_time - validation_start_time:.6f}s")
        print(f"Accuracy: {accuracy:.6f}")
    
    # Use for Small Data Set normalized dataset
    elif data_num == 1 and normalize_num == 2:
        accuracy = validator.calculate([3, 5, 7], normalized_data, "NN")
        validation_end_time = time.time()
        print(f"Validation completed in {validation_end_time - validation_start_time:.6f}s")
        print(f"Accuracy: {accuracy:.6f}")

if __name__ == "__main__":
    main()