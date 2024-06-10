import random
from NN import Validator
from NN import normalize_dataset
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

        paired_val = []

        for i in toeval:
            copycurr = self.state.copy()
            copycurr.append(i)
            temp = Node(copycurr)
            if(self.k):
                temp.k = self.k
            eval = temp.evaluate(copycurr, dataset, classifer)
            pair = (eval, i)
            paired_val.append(pair)
            print(f'Using feature(s) {copycurr} accuracy is {eval}')
    

        paired_val.sort(reverse=True)
        return paired_val


def forward(numfeat, classifer, choice, select):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    if classifer == 'KNN':
        k = input("What K value do you want:")
    if select == 1:
        file_path = 'CS170_Spring_2024_Small_data__69.txt'
    if select == 2:
        file_path = 'CS170_Spring_2024_Large_data__69.txt'
    dataset = np.loadtxt(file_path)
    #print(type(dataset))
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    if choice == 1:
        beginningPercent = head.randomeval(dataset)
    if choice == 2:
        normalized_data = normalize_dataset(dataset)
        beginningPercent = head.randomeval(normalized_data)

    pper = beginningPercent
    print(f'Using no features and random evaluation, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    if choice == 1:
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
    if choice == 2:
        for i in range(numfeat):
            if classifer == 'KNN':
                head.k = k
            head, fper = head.dosearchfoward(totalpath, normalized_data, classifer)
            if(pper > fper):
                print(f'(Warning, Accuracy has decreased!)')
            elif(fper > mper): 
                max = head
                mper = fper
            pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0

def backward(numfeat, classifer, choice, select):
    if select == 1:
        file_path = 'CS170_Spring_2024_Small_data__69.txt'
    if select == 2:
        file_path = 'CS170_Spring_2024_Large_data__69.txt'
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

    if choice == 1:
        beginningPercent = head.evaluate(list(totalpath), dataset, classifer)
    if choice == 2:
        normalized_data = normalize_dataset(dataset)
        beginningPercent = head.evaluate(list(totalpath), normalized_data, classifer)

    print(f'Using all features, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    if choice == 1:
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
    if choice == 2:
        for i in range(numfeat):
            if classifer == 'KNN':
                head.k = k
            head, fper = head.dosearchback(normalized_data, classifer)
            if(pper > fper):
                print(f'(Warning, Accuracy has decreased!)')
            elif(fper > mper): 
                max = head
                mper = fper
            pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0

def everything(numfeat, classifer, choice, select):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    if classifer == 'KNN':
        k = input("What K value do you want:")
    if select == 1:
        file_path = 'CS170_Spring_2024_Small_data__69.txt'
    if select == 2:
        file_path = 'CS170_Spring_2024_Large_data__69.txt'
    dataset = np.loadtxt(file_path)
    #print(type(dataset))
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    if choice == 1:
        beginningPercent = head.randomeval(dataset)
    if choice == 2:
        normalized_data = normalize_dataset(dataset)
        beginningPercent = head.randomeval(normalized_data)

    pper = beginningPercent
    if classifer == 'KNN':
        head.k = k
    if choice == 1:
        head, fper = head.dosearcheverything(totalpath, dataset, classifer)
    if choice == 2:
        head, fper = head.dosearcheverything(totalpath, normalized_data, classifer)
    
    print(f'Finished search!! The best feature subset is {head.state} , which has an accuracy of {fper}')
    return 0

def mostsignificant(numfeat, classifer, choice, select):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    if classifer == 'KNN':
        k = input("What K value do you want:")
    if select == 1:
        file_path = 'CS170_Spring_2024_Small_data__69.txt'
    if select == 2:
        file_path = 'CS170_Spring_2024_Large_data__69.txt'
    dataset = np.loadtxt(file_path)
    #print(type(dataset))
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    if choice == 1:
        beginningPercent = head.randomeval(dataset)
    if choice == 2:
        normalized_data = normalize_dataset(dataset)
        beginningPercent = head.randomeval(normalized_data)

    pper = beginningPercent
    print(f'Using no features and random evaluation, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    if classifer == 'KNN':
        head.k = k
    if choice == 1:
        accuracys = head.dosearchsignifcant(totalpath, dataset, classifer)
    if choice == 2:
        accuracys = head.dosearchsignifcant(totalpath, normalized_data, classifer)
    filtered_list = [x for x in accuracys if x[0] >= 0.70]

    first_values = [pair[1] for pair in filtered_list]

    print(f'Finished search!! The best feature subset is {first_values}')

    return 0


def main():
    print("Welcome to Justin's Feature Selection Algorithm. \n")
    numfeat = int(input("Please enter total number of features: "))
    print("\n")
    print("'1' for Forward Selection, '2' for Backward Selection, '3' for Justin's Special Algorithm, '4' for Most Significant Features Algorithm")
    selection = input("Type the number of the algorithm you want to run. \n Forward Selection \n Backward Selection \n Justin's Special Algorithm \n Most signifcant features\n")
    select = int(input("Select small/large dataset. '1' for small. '2' for large: "))
    choice = int(input("Enter your choice. '1' for non-normalize. '2' for normalize: "))
    choice2 = input("Type the classifer of the algorithm you want to run. Type '1' for KNN, '2' for NN, '3' for NB: ")
    if choice2 == '1':
        classifierr = 'KNN'
    elif choice2 == '2':
        classifierr = 'NN'
    elif choice2 == '3':
        classifierr = 'NB'

    if selection == '1':
        forward(numfeat, classifierr, choice, select)
    elif selection == '2':
        backward(numfeat, classifierr, choice, select)
    elif selection == '3':
        everything(numfeat, classifierr, choice, select)
    elif selection == '4':
        mostsignificant(numfeat, classifierr, choice, select)
    else:
        print("Invalid selection. Please choose either 'forward' or 'backward'.")

if __name__ == "__main__":
    main()