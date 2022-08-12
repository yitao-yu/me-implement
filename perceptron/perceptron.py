#!/usr/bin/python3

# AUTHOR:  Yitao Yu
# NetID:   yyu56

import numpy as np
# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points

# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        return xs, ys

# Do learning.
def perceptron(train_xs, train_ys, iterations):
    #return weights#
    w = []
    for i in train_xs[0]:
        w.append(np.float64(1))
    #training part#
    accuracy = []
    for i in range(0,iterations):
        for j in range(0,len(train_xs)):
            if perceptron_class(w,train_xs[j])*train_ys[j] < 0:
                w += train_xs[j]*train_ys[j]
        acc = test_accuracy(w,train_xs,train_ys)
        accuracy.append(acc)
        if(acc >= 1.0000000):#converged
            return w,accuracy
    return w,accuracy

def perceptron_class(w,d):
    if np.dot(w,d) > 0:
        return 1
    else:
        return -1

# Return the accuracy over the data using current weights.
def test_accuracy(weights, test_xs, test_ys):
    w = weights
    count = [0,0]
    for i in range(0, len(test_xs)):
        if perceptron_class(w,test_xs[i])*test_ys[i] > 0:
            count[0] += 1
        count[1] += 1
    return np.float64(count[0])/np.float64(count[1])
        
def some_plot(folderpath, filearr):#plot for non-separable dataset
    #this method may only not be run on the CSUG machine#
    import matplotlib.pyplot as plt
    for i in filearr:
        print(i)
        fpath = str(folderpath) + i
        train_xs, train_ys = parse_data(fpath)
        iteration = 1000000
        weights,accuracy = perceptron(train_xs, train_ys, iteration)
        print("done training")
        x = []
        maxacc = [0,0]
        ittr = 0
        score = test_accuracy(weights, train_xs, train_ys)
        for y in accuracy:
            ittr = len(x)+1
            x.append(ittr)
            if y > maxacc[1]:
                maxacc[1] = y
                maxacc[0] = ittr
        fig = plt.figure()
        plt.plot(x,accuracy)
        plt.margins(x = 0.2, y = 0.1)
        if(score < 1.0):
            plt.text(maxacc[0], maxacc[1],str('max_accuracy['+str(maxacc[0])+','+str(round(maxacc[1],4))+']'))
            plt.text(ittr, score,str('                      max_itteration['+str(ittr)+','+str(round(score,4))+']'))
        else:
            plt.text(ittr, score,str('converged'+str(ittr)))
        plt.xlabel('iterations')
        plt.ylabel('accuracy score')
        plt.title(i.split('.')[0])
        plt.show()
        name = str(i.split('.')[0])+'_fig' +str('.png')
        fig.savefig(name)
        print('end')
        
def suggest_iter(train_xs):
    max_r = 0
    for i in train_xs:
        if(np.linalg.norm(i)>max_r):
            max_r = np.linalg.norm(i)
    print(max_r)
    if(max_r > 4.0):
        return 30000
    return (int(max_r)**2) * 100#Supposing sigma is 0.1#
    
def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')
    
    args = parser.parse_args()
    
    #filearr = ['challenge4.dat']
    #some_plot('C:/Users/yyu56/Desktop/perceptron-master/data/', filearr)
    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(args.train_file)

    weights,accuracy = perceptron(train_xs, train_ys, args.iterations)
    accuracy = test_accuracy(weights, train_xs, train_ys)
    print('Train accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))
    
if __name__ == '__main__':
    main()
