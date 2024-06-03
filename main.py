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
            eval = temp.evaluate(copycurr, dataset, classifer)
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}%')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}%\n')
        return nextstep, maxacc

    def dosearchback(self, dataset, classifer):
        maxacc = 0
        nextstep = Node
        featremoved = 0
        for i in self.state:
            copycurr = self.state.copy()
            copycurr.remove(i)
            temp = Node(copycurr)
            eval = temp.evaluate(copycurr, dataset, classifer)
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}%')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}%\n')
        return nextstep, maxacc

def forward(numfeat, classifer):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)

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
    beginningPercent = head.evaluate(list(totalpath), dataset, classifer)

    print(f'Using all features, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    for i in range(numfeat):
        head, fper = head.dosearchback(dataset, classifer)
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

    if selection == '1':
        forward(numfeat, "NN")
    elif selection == '2':
        backward(numfeat, "NB")
    # elif selection == '3':
    #     custom?
    else:
        print("Invalid selection. Please choose either 'forward' or 'backward'.")

if __name__ == "__main__":
    main()


