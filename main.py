import random
from NN import Validator
import numpy as np
from select import select


class Node:
    def __init__(self, curr=None):
        if curr is None or not curr:
            self.state = []
        else:
            self.state = curr
        self.k = None
    #def evaluate(self):
    #    return 1  
    def randomeval(self, dataset):
        classes = dataset[:, 0]
        clasSet = set(classes)
        classList = list(clasSet)
        score = 0
        for i in dataset:
            random_number = random.choice(classList)
            if (i[0] == random_number):
                score = score + 1
        return (score/len(dataset))



    def evaluate(self, features, classifer, dataset):
        validator = Validator()
        if(self.k):
           accuracy = validator.calculate(features, classifer, dataset, self.k) 
           #print(self.k)
        else:
            accuracy = validator.calculate(features, classifer, dataset)
        return accuracy

    def dosearchfoward(self, fullset, dataset, classifer):
        maxacc = 0
        toeval =  list(fullset - set(self.state))
        nextstep = Node
        for i in toeval:
            copycurr = self.state.copy()
            copycurr.append(i)
            temp = Node(copycurr)
            if(self.k):
                temp.k = self.k
            eval = temp.evaluate(copycurr, dataset, classifer)
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}\n')
        return nextstep, maxacc

    def dosearchback(self, dataset, classifer):
        maxacc = 0
        nextstep = Node
        featremoved = 0
        for i in self.state:
            copycurr = self.state.copy()
            copycurr.remove(i)
            temp = Node(copycurr)
            if(self.k):
                temp.k = self.k
            eval = temp.evaluate(copycurr, dataset, classifer)
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}\n')
        return nextstep, maxacc

    def dosearcheverything(self, fullset, dataset, classifer):
        maxacc = 0
        toeval =  list(fullset - set(self.state))
        print(toeval)
        nextstep = Node
        #print(self.state)
        n = len(fullset)
        lfullset = list(fullset)
        all_subsets = []
        for i in range(2**n):  # Iterate over all numbers from 0 to 2^n - 1
            subset = []
            for j in range(n):
               if (i >> j) & 1:  # Check if the j-th bit of i is set
                  subset.append(lfullset[j])  # Include lst[j] in the subset
            copycurr = subset
            temp = Node(copycurr)
            if(self.k):
                temp.k = self.k
            eval = temp.evaluate(copycurr, dataset, classifer)
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}\n')
        return nextstep, maxacc

    def dosearchsignifcant(self, fullset, dataset, classifer):
        maxacc = 0
        toeval =  list(fullset - set(self.state))
        nextstep = Node
        print(toeval)
        for i in toeval:
            copycurr = self.state.copy()
            copycurr.append(i)
            temp = Node(copycurr)
            if(self.k):
                temp.k = self.k
            eval = temp.evaluate(copycurr, dataset, classifer)
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}\n')
        return nextstep, maxacc


def forward(numfeat, classifer):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    if classifer == 'KNN':
        k = input("What K value do you want:")
    file_path = '/Users/justincrafty/Documents/CS170/CS170-Project-2/CS170_Spring_2024_Small_data__69.txt'
    dataset = np.loadtxt(file_path)
    #print(type(dataset))
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    beginningPercent = head.randomeval(dataset)

    pper = beginningPercent
    print(f'Using no features and random evaluation, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    for i in range(numfeat):
        if classifer == 'KNN':
            head.k = k
        head, fper = head.dosearchfoward(totalpath, dataset, classifer)
        if(pper > fper):
            print(f'(Warning, Accuracy has decreased!)')
        elif(fper > mper): 
            max = head
            mper = fper
        pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0

def backward(numfeat, classifer):
    file_path = '/Users/justincrafty/Documents/CS170/CS170-Project-2/CS170_Spring_2024_Small_data__69.txt'
    dataset = np.loadtxt(file_path)
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    pper = 0
    mper = 0
    max = Node()
    head = Node(totalpath)
    if classifer == 'KNN':
        k = input("What K value do you want:")
        head.k = k

    

    beginningPercent = head.evaluate(list(totalpath), dataset, classifer)

    print(f'Using all features, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    for i in range(numfeat):
        if classifer == 'KNN':
            head.k = k
        head, fper = head.dosearchback(dataset, classifer)
        if(pper > fper):
            print(f'(Warning, Accuracy has decreased!)')
        elif(fper > mper): 
            max = head
            mper = fper
        pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0

def everything(numfeat, classifer):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    if classifer == 'KNN':
        k = input("What K value do you want:")
    file_path = '/Users/justincrafty/Documents/CS170/CS170-Project-2/CS170_Spring_2024_Small_data__69.txt'
    dataset = np.loadtxt(file_path)
    #print(type(dataset))
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    beginningPercent = head.randomeval(dataset)

    pper = beginningPercent
    if classifer == 'KNN':
        head.k = k
    head, fper = head.dosearcheverything(totalpath, dataset, classifer)
    
    print(f'Finished search!! The best feature subset is {head.state} , which has an accuracy of {fper}')
    return 0

def mostsignificant(numfeat, classifer):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    if classifer == 'KNN':
        k = input("What K value do you want:")
    file_path = '/Users/justincrafty/Documents/CS170/CS170-Project-2/CS170_Spring_2024_Small_data__69.txt'
    dataset = np.loadtxt(file_path)
    #print(type(dataset))
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    beginningPercent = head.randomeval(dataset)

    pper = beginningPercent
    print(f'Using no features and random evaluation, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    if classifer == 'KNN':
        head.k = k
    head, fper = head.dosearchsignifcant(totalpath, dataset, classifer)
    if(pper > fper):
        print(f'(Warning, Accuracy has decreased!)')
    elif(fper > mper): 
        max = head
        mper = fper
    pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0


def main():
    print("Welcome to Justin's Feature Selection Algorithm. \n")
    numfeat = int(input("Please enter total number of features: "))
    print("\n")
    selection = input("Type the number of the algorithm you want to run. \n Forward Selection \n Backward Selection \n Justin's Special Algorithm \n")
    classifierr = input("Type the classifer of the algorithm you want to run.")

    if selection == '1':
        mostsignificant(numfeat, classifierr)
    elif selection == '2':
        backward(numfeat, classifierr)
    elif selection == '3':
        everything(numfeat, classifierr)
    else:
        print("Invalid selection. Please choose either 'forward' or 'backward'.")

if __name__ == "__main__":
    main()


