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
    for i in range(numfeat):
        head, fper = head.dosearchfoward(totalpath)
        if(pper > fper):
            print(f'(Warning, Accuracy has decreased!)')
            break
        else:
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
    for i in range(numfeat):
        head, fper = head.dosearchback()
        if(pper > fper):
            print(f'(Warning, Accuracy has decreased!)')
            break
        else:
            max = head
            mper = fper
        pper = fper

    print(f'Finished search!! The best feature subset is {max.state} , which has an accuracy of {mper}')
    return 0
    
def main():
    numfeat = int(input("Please enter the total number of features: "))
    selection = input("Which algorithm do you want to run? (Enter 'forward' or 'backward'): ").lower()

    if selection == 'forward':
        forward(numfeat)
    elif selection == 'backward':
        backward(numfeat)
    else:
        print("Invalid selection. Please choose either 'forward' or 'backward'.")

if __name__ == "__main__":
    main()


