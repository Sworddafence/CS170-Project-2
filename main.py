import random
from select import select


class Node:
    def __init__(self, curr=None):
        if curr is None or not curr:
            self.state = []
        else:
            self.state = curr
    
    def evaluate(self):
        r = round(random.uniform(0, 100), 2)
        return r

    def dosearchfoward(self, fullset):
        maxacc = 0
        toeval =  list(fullset - set(self.state))
        nextstep = Node
        for i in toeval:
            copycurr = self.state.copy()
            copycurr.append(i)
            temp = Node(copycurr)
            eval = temp.evaluate()
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}%')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}%\n')
        return nextstep, maxacc

    def dosearchback(self):
        maxacc = 0
        nextstep = Node
        featremoved = 0
        for i in self.state:
            copycurr = self.state.copy()
            copycurr.remove(i)
            temp = Node(copycurr)
            eval = temp.evaluate()
            if(eval > maxacc):
                nextstep = temp
                maxacc = eval
            print(f'Using feature(s) {copycurr} accuracy is {eval}%')
        print(f'\nFeature set {nextstep.state} was best, accuracy is {maxacc}%\n')
        return nextstep, maxacc

def forward(numfeat):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    pper = 0
    mper = 0
    max = Node()
    head = Node()
    beginningPercent = head.evaluate()

    pper = beginningPercent
    print(f'Using no features and random evaluation, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    for i in range(numfeat):
        head, fper = head.dosearchfoward(totalpath)
        if(pper > fper):
            print(f'(Warning, Accuracy has decreased!)')
        elif(fper > mper): 
            print("blahblah")
            max = head
            mper = fper
        pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0

def backward(numfeat):
    totalpath = set()
    for i in range(1, numfeat + 1):
        totalpath.add(i)
    pper = 0
    mper = 0
    max = Node()
    head = Node(totalpath)
    beginningPercent = head.evaluate();

    print(f'Using no features and random evaluation, I get an accuracy of {beginningPercent} \n')
    print("Beginning Search. \n")

    for i in range(numfeat):
        head, fper = head.dosearchback()
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
        forward(numfeat)
    elif selection == '2':
        backward(numfeat)
    # elif selection == '3':
    #     custom?
    else:
        print("Invalid selection. Please choose either 'forward' or 'backward'.")

if __name__ == "__main__":
    main()


