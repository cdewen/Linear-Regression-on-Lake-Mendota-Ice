import sys
import csv
import numpy as np
import matplotlib.pyplot as plt 

def plot(file):
    x = [] 
    y = [] 
    
    with open(file,'r') as csvfile: 
        lines = csv.reader(csvfile, delimiter=',') 
        for row in lines: 
            x.append(row[0]) 
            y.append(row[1])
    x = np.delete(x, 0)
    y = np.delete(y, 0) 
    y = y.astype(int)
    plt.plot(x, y)
    plt.xlabel('Year') 
    plt.ylabel('Number of Frozen Days') 
    plt.savefig("plot.jpg")


def getX(file):
    results = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            results.append(row[0])
    X = np.empty(((len(results) - 1),2), dtype=np.int64)
    for i in range(1, len(results)):
        X[i-1][0] = 1
        X[i-1][1] = results[i]

    print("Q3a:")
    print(X)
    return X

def getY(file):
    results = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            results.append(row[1])
    Y = np.empty(((len(results) - 1)), dtype=np.int64)
    for i in range(1, len(results)):
        Y[i-1] = results[i]

    print("Q3b:")
    print(Y)
    return Y

def getZ(X):
    Z = X.transpose().dot(X)
    print("Q3c:")
    print(Z)
    return Z

def getI(Z):
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

def getPI(X):
    Pi = np.linalg.matrix_power((X.T.dot(X)), -1).dot(X.T)
    print("Q3e:")
    print(Pi)
    return Pi

def getBeta(file):
    X = getX(file)
    Y = getY(file)
    Z = getZ(X)
    I = getI(Z)
    PI = getPI(X)
    beta = PI.dot(Y)
    print("Q3f:")
    print(beta)
    return beta

if __name__ == "__main__":
    file = sys.argv[1]
    plot(file)
    beta =  getBeta(file)
    ytest = beta.T.dot(np.array([1, 2022]))
    print("Q4: " + str(ytest))
    symbol = ""
    if beta[1] == 0:
        symbol = "="
    elif beta[1] > 0:
        symbol = ">"
    else:
        symbol = "<"
    print("Q5a: " + str(symbol))
    print("Q5b: " + "a sign of = means there is no statistically singnificant change caused by an increase in year. A sign of > means that there is an statisically significant increase based on year and vice versa for <")
    xStar = -beta[0] / beta[1]
    print("Q6a: " + str(xStar))
    print("Q6b: " + "It is not a compelling prediction because there most likely isn't a steady linear decrease to a time when there is on frozen days, and rather is probably logarithmic and will flatten out if the decrease continues.")

