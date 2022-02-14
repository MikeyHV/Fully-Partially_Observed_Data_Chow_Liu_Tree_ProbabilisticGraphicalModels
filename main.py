import copy
import os
import time
import ast
import multiprocessing

from pgmpy.readwrite import UAIReader
import math
from decimal import *
import random
import sys
import portalocker
from operator import itemgetter
from functools import partial
from itertools import product
from multiprocessing.pool import Pool


def pointWiseDiffTask3(var, testData, factors, thetaStar, runType, printType, data_file, resultsFile):
    final = 0
    '''print(var[0])
    testMe = testData[0].split()
    print(testMe)
    #  x = str(testMe[0][70]) + str(testMe[0][19]) + str(testMe[0][5]) + str(testMe[0][54]) + str(testMe[0][64])
    x = str(testMe[70]) + str(testMe[19]) + str(testMe[5]) + str(testMe[54]) + str(testMe[64])
    print(x)
    print(int(x, 2))
    print(thetaStar[2])
    print(factors[2])'''
    '''print(thetaStar)
    print(factors)

    for i in range(int(len(thetaStar))):
        print(i, ": ", thetaStar[i])

    print(testData[0])'''
    '''for i in range(int(len(thetaStar))):
        print(i, ": ", thetaStar[i])
    quit()'''
    # print(thetaStar)
    # print(factors)
    arrOfResults = []
    altVarList = thetaStar[2]
    numVarLists = len(altVarList)
    for i in range(len(testData)):
        arr = testData[i].split()
        # trying to count the number of times each tuple appears in this one line of data.
        # per variable, save the parIndex, and then use that to grab the theta of uai and learned
        # if the variable was 0, use the value directly. if the variable was 1, take the inverse.
        parIndices = []
        learnedParentIndices = [[]] * numVarLists
        for j in range(len(var)):
            r = int(arr[j])
            parBinaryFac = getBinaryOfParents(var, j, arr)
            parIndexFac = int(parBinaryFac, 2)
            parIndices.append((parIndexFac, r))
            for k in range(numVarLists):
                parBinaryAlt = getBinaryOfParents(altVarList[k], j, arr)
                parIndexAlt = int(parBinaryAlt, 2)
                temp = copy.deepcopy(learnedParentIndices[k])
                temp.append((parIndexAlt, r))
                learnedParentIndices[k] = temp
            # sum up the number of times i saw the tuple parArray
        result = LLTask3(parIndices, learnedParentIndices, var, thetaStar, factors, numVarLists)
        arrOfResults.append(result)
        final = final + result
        if i == 7500:
            mean = final / (i + 1)
            print(runType + " " + data_file + " " + str(numVarLists) + " " + str(i))
            Avg = "Avg: " + str(mean)
            # fin = "fin: " + str(final)
            finLog = "finLog: " + str(Decimal(200000) * mean)
            std = "Std: " + str(standardDeviation(mean, arrOfResults))
            print(Avg)
            # print(fin)
            print(finLog)
            print(std)
            # print(finLogged)
            if printType == 'True':
                g = open(resultsFile, "a")
                portalocker.lock(g, portalocker.LOCK_EX)
                g.write("\n")
                g.write(runType + " " + data_file + " " + str(i))
                g.write("\n")
                g.write(Avg)
                '''g.write("\n")
                g.write(fin)'''
                g.write("\n")
                g.write(finLog)
                g.write("\n")
                g.write(std)
                g.write("\n")
                g.close()
            break


def LLTask3(parIndices, learnedParentIndices, var, thetaStar, factors, numVarLists):
    modelLikelihood = 0
    theta = thetaStar[0]
    p = thetaStar[1]
    finalLearnedLikelihood = 0
    for i in range(len(var)):
        parentIndex = parIndices[i][0]
        varValue = Decimal(parIndices[i][1])
        if float(factors[i][parentIndex]) < 0.00001:
            factors[i][parentIndex] = '0.00001'
        modelValue = Decimal(math.log10(Decimal(factors[i][parentIndex])))
        if varValue <= 0.5:
            modelLikelihood = modelLikelihood + modelValue
        else:
            modelLikelihood = Decimal(modelLikelihood + (1 - modelValue))

        learnedLikelihood = [0] * 3
        # todo this might be wrong. Something about hwere to actually multiply times p[k]
        for k in range(numVarLists):  # for every model
            learnedParentIndex = learnedParentIndices[k][i][0]
            learnedVarValue = Decimal(learnedParentIndices[k][i][0])
            learnedValue = Decimal(math.log10(Decimal(theta[k][i][learnedParentIndex])))
            if learnedVarValue <= 0.5:
                learnedValue = learnedValue
            else:
                learnedValue = (1 - learnedValue)
            learnedLikelihood[k] = learnedValue + learnedLikelihood[k]
        tempLearnedLikelihood = 0
        for k in range(numVarLists):
            tempLearnedLikelihood = Decimal(tempLearnedLikelihood) + Decimal(learnedLikelihood[k] * p[k])
        finalLearnedLikelihood = Decimal(finalLearnedLikelihood) + Decimal(tempLearnedLikelihood)
    return Decimal(abs(finalLearnedLikelihood - modelLikelihood))


def LL(parIndices, var, theta, factors):
    modelLikelihood = 0
    learnedLikelihood = 0
    for i in range(len(var)):
        parentIndex = parIndices[i][0]
        varValue = Decimal(parIndices[i][1])
        if float(factors[i][parentIndex]) < 0.00001:
            factors[i][parentIndex] = '0.00001'
        modelValue = Decimal(math.log10(Decimal(factors[i][parentIndex])))
        thetaValue = Decimal(math.log10(Decimal(theta[i][parentIndex])))
        if varValue <= 0.5:
            learnedLikelihood = learnedLikelihood + thetaValue
            modelLikelihood = modelLikelihood + modelValue
        else:
            learnedLikelihood = Decimal(learnedLikelihood + (1 - thetaValue))
            modelLikelihood = Decimal(modelLikelihood + (1 - modelValue))
    return Decimal(abs(learnedLikelihood - modelLikelihood))


def FOD_learn(data, var):
    '''
    :param data: the training data
    :param var: the list of the parents for each var
    :return: thetaStar
    '''
    # initializing the table which will hold the num occurrences of each tuple per var
    numXiandParentXiOccurences = [[]] * len(var)
    numOccurencesParentPermutation = [[]] * len(var)
    thetastars = [[]] * len(var)
    for i in range(len(var)):
        temp = [0] * (2 ** len(var[i]))
        numXiandParentXiOccurences[i] = copy.deepcopy(temp)
        numOccurencesParentPermutation[i] = copy.deepcopy(temp)
        thetastars[i] = copy.deepcopy(temp)

    for i in range(len(data)):  # for each line of data
        arr = data[i].split()
        for j in range(len(var)):
            parBinary = getBinaryOfParents(var, j, arr)
            parIndex = int(parBinary, 2)
            # sum up the number of times i saw the tuple parArray
            numOccurencesParentPermutation[j][parIndex] = int(numOccurencesParentPermutation[j][parIndex]) + 1
            if arr[j] == '0':
                numXiandParentXiOccurences[j][parIndex] = int(numXiandParentXiOccurences[j][parIndex]) + 1

    for z in range(len(thetastars)):
        x = thetastars[z]
        varAndParent = numXiandParentXiOccurences[z]
        parent = numOccurencesParentPermutation[z]
        for j in range(len(x)):  # calculating the theta star with laPlace smoothing
            # TODO: Make sure this is correct
            x[j] = (varAndParent[j] + 1) / (parent[j] + 2)
            # x[j] = varAndParent[j] / parent[j]
            if x[j] == 1:
                x[j] = 0.99999
            if x[j] == 0:
                x[j] = 0.00001
            # if varAndParent[j] == 0 and parent[j] == 0:
            '''else:
                #x[j] = (varAndParent[j] + 1) / (parent[j] + 2)
                # x[j] = varAndParent[j] / parent[j]'''
            '''if parent[j] == 0:
                x[j] = 0.00000000000001
            else:
                x[j] = varAndParent[j] / parent[j]
            if x[j] == 0:
                x[j] = 0.00000000000001'''
        thetastars[z] = x

    return thetastars


def task3(numVars, k, data):
    """
    :param k:
    :param numVars:
    :param data: full dataset
    :return: normalized parameters
    """
    '''
    arrOfNormWeights = []
    for i in range(len(data)):
        arr = data[i].split()
        #  for each variable, go in to calculate the parameters associated with it
        arrOfWeightPerVar = []  # this array holds all weights
        if '?' in arr:  # if any of its parents were missing
            quesCount = arr.count('?')
            arrOfWeights = []
            normalizeMe = []
            for j in range(2 ** quesCount):
                temp = questionMarkProcedure(arr, varRandomParameters, j, varList, quesCount)
                #  question mark procedure output is correct
                normalizeMe.append(temp[1])
                arrOfWeights.append(temp)
            finalWeightsForJ = normalizeProbabilities(normalizeMe)
            x = 0
            finalReturn = []
            for j in range(len(arrOfWeights)):
                finalReturn.append((arrOfWeights[j][0], finalWeightsForJ[j]))
            arrOfNormWeights.extend(finalReturn)
    '''
    '''
    valofCurrData = 1
    for o in range(len(newBin)):
        # newBin is now a binary number, but i still need to know where the question marks were
        # so use recordOfQuestionMarks to indicate where the ? were so I can pull the values from varRan
        # arrTo... may be unnecessary, i was going to product over that array but that's not necessary
        parents = getBinaryOfParents(varList, o, newBin)
        valOfCurrData = Decimal(valOfCurrData) * Decimal(varRandomParameters[o][int(parents, 2)])
    return newBin, valOfCurrData
    '''
    '''
    numXiandParentXiOccurences = [[]] * len(var)
    numOccurencesParentPermutation = [[]] * len(var)
    thetastars = [[]] * len(var)
    for i in range(len(var)):
        temp = [0] * (2 ** len(var[i]))
        numXiandParentXiOccurences[i] = copy.deepcopy(temp)
        numOccurencesParentPermutation[i] = copy.deepcopy(temp)
        thetastars[i] = copy.deepcopy(temp)
    '''
    varLists = []  # k lists, each list is the parents of each variable, indexed
    parameters = []  # k lists, each list is the parameters of each variable, indexed
    p = []
    for i in range(k):
        list = copy.deepcopy(randomDAGs(numVars))
        varLists.append(list[0])
        parameters.append(list[1])
        p.append(random.random())
    p = normalizeProbabilities(p)

    k = len(varLists)
    for q in range(18):
        weightOfModelPerLine = [[]] * len(data)  # each index should be a different model, that index holds the weights per line
        weightOfLine = [0] * len(data)  # total weight for each line of data
        weightParPerModelAndVar = [[[]]] * k  # weight of occurrences of each parameter, per model, per variable
        weightParParentPerModelAndVar = [[[]]] * k  # weight of occurrences of each parameter's parents, per model, per variable
        for i in range(k):  # for each model
            temp1 = varLists[i]
            numVariables = len(temp1)
            temp1_1 = [[]] * numVariables
            weightParParentPerModelAndVar[i] = copy.deepcopy(temp1_1)
            weightParPerModelAndVar[i] = copy.deepcopy(temp1_1)
            for j in range(numVariables):  # for each variable
                numParameters = 2 ** len(temp1[j])
                temp2 = [0] * numParameters
                weightParParentPerModelAndVar[i][j] = copy.deepcopy(temp2)
                weightParPerModelAndVar[i][j] = copy.deepcopy(temp2)
        returnValues = task3EStep(weightParPerModelAndVar, weightParParentPerModelAndVar, weightOfLine,
                                  weightOfModelPerLine, varLists, data, parameters, p)
        weightOfModelPerLine = returnValues[0]
        weightParParentPerModelAndVar = returnValues[1]
        weightOfLine = returnValues[2]
        weightOfModelPerLine = returnValues[3]
        returnValues2 = task3MStep(p, weightOfModelPerLine, parameters, varLists, weightOfLine,
                                   weightParParentPerModelAndVar,
                                   weightParPerModelAndVar)
        parameters = returnValues2[0]
        p = returnValues2[1]
    return parameters, p, varLists


def task3EStep(weightParPerModelAndVar, weightParParentPerModelAndVar, weightOfLine, weightOfModelPerLine, varLists,
               data, parameters, p):
    k = len(varLists)
    for i in range(len(data)):  # each line of data
        arr = data[i].split()
        for j in range(k):  # each model
            modelVars = varLists[j]
            modelParas = parameters[j]
            valOfCurrModel = 1
            for o in range(len(arr)):  # for each of the variables of the line of data
                # projecting the value of teh data on model i
                parents = getBinaryOfParents(modelVars, o, arr)
                parIndex = int(parents, 2)
                if int(arr[o]) == 0:
                    valOfParameter = Decimal(modelParas[o][parIndex])
                    valOfCurrModel = Decimal(valOfCurrModel) * valOfParameter
                    weightParPerModelAndVar[j][o][parIndex] = weightParPerModelAndVar[j][o][
                                                                  parIndex] + valOfParameter * Decimal(p[j])
                    weightParParentPerModelAndVar[j][o][parIndex] = Decimal(
                        weightParParentPerModelAndVar[j][o][parIndex]) + valOfParameter * Decimal(p[j])
                elif int(arr[o]) == 1:
                    valOfParameter = 1 - Decimal(modelParas[o][parIndex])
                    valOfCurrModel = Decimal(valOfCurrModel) * (1 - Decimal(modelParas[o][parIndex]))
                    weightParParentPerModelAndVar[j][o][parIndex] = Decimal(
                        weightParParentPerModelAndVar[j][o][parIndex]) + valOfParameter * Decimal(p[j])
            valOfCurrModel = valOfCurrModel * Decimal(p[j])  # valOfCurrModel is PiVi
            weightOfLine[i] = weightOfLine[i] + valOfCurrModel
            temp = copy.deepcopy(weightOfModelPerLine[j])
            temp.append(valOfCurrModel)
            weightOfModelPerLine[j] = temp
    return weightParParentPerModelAndVar, weightParParentPerModelAndVar, weightOfLine, weightOfModelPerLine


def task3MStep(p, weightOfModelPerLine, parameters, varLists, weightOfLine, weightParParentPerModelAndVar,
               weightParPerModelAndVar):
    """
    :param p:
    :param weightOfModelPerLine:
    :param parameters:
    :param varLists:
    :param weightOfLine:
    :param weightParParentPerModelAndVar:
    :param weightParPerModelAndVar:
    :return: no return, i dont make any deep copies so all arrays are changed automagically
    """
    k = len(varLists)
    for i in range(k):  # for each of the models
        val = 0
        for j in range(len(weightOfModelPerLine)):  # for each line of data
            val = val + weightOfModelPerLine[i][j] / weightOfLine[j]
        parametersForModel = parameters[i]
        varListPerModel = varLists[i]
        for o in range(len(varListPerModel)):  # over the varList, over each variable
            for j in range(len(varListPerModel[o])):  # over the parameters of the variable
                weightPar = weightParParentPerModelAndVar[i][o][j]
                weightVar = weightParPerModelAndVar[i][o][j]
                parametersForModel[o][j] = Decimal(weightVar + 1) / Decimal(weightPar + 2)
        parameters[i] = parametersForModel
        p[i] = val / len(weightOfLine)
    return parameters, p


def randomDAGs(numVars):
    varList = [] * numVars
    parList = []
    for i in range(numVars):
        varList.append(i)
    random.shuffle(varList)
    for i in range(numVars):
        if i == numVars - 3:
            numParents = random.randint(0, 2)
        elif i == numVars - 2:
            numParents = random.randint(0, 1)
        elif i == numVars - 1:
            numParents = 0
        else:
            numParents = random.randint(0, 3)
        x = random.sample(range(i + 1, numVars), numParents)
        temp = []
        for j in range(numParents):
            temp.append(varList[x[j]])
        parList.append((varList[i], temp))
    parList = sorted(parList, key=itemgetter(0))
    paras = []
    for i in range(numVars):
        temp = [Decimal(random.random()) for _ in range(2 ** len(parList[i][1]))]
        parList[i] = copy.deepcopy(parList[i][1])
        paras.append(temp)
    return parList, paras


def POD_EM_Learn(data, varList):
    # initializing the table which will hold the num occurrences of each tuple per var
    varRandomParameters = [[]] * len(varList)  # random values for all parameters [0, 1)
    for i in range(len(varList)):
        # temp = [0] * (2 ** len(varList[i]))
        temp = [Decimal(random.random()) for _ in range(2 ** len(varList[i]))]
        varRandomParameters[i] = copy.deepcopy(temp)

    maxQuestMarks = 0
    for i in range(len(data)):
        count = data[i].count('?')
        if count > maxQuestMarks:
            maxQuestMarks = count
    if maxQuestMarks > 8:
        maxQuestMarks = 8
    maxRuns = 2 ** maxQuestMarks
    if maxRuns > 20:
        maxRuns = 18
    #  data is the array that holds all the thingies, pulled from the actual data text file
    #  i believe this entire for loop will be the E-step, and the next for loop will be the M-step
    for i in range(maxRuns):
        # list of tuples, (permutation, normalized weight of permutation)
        arrOfNormWeights = EStep(data, varList, varRandomParameters)
        # weights are normalize. so output is decent at least
        # E-step is done. WOO, except its not loser
        # On to the M-step
        varRandomParameters = MStep(arrOfNormWeights, varList, varRandomParameters)
        '''if i == 2:
            for z in range(len(varRandomParameters)):
                print(z, ":", varRandomParameters[z])
            quit()'''
        # todo, verify m-step is correct and also make sure that varRandomPara is updated for e-step

    '''for i in range(len(varRandomParameters)):
        print(i, ":", varRandomParameters[i])'''
    # quit()
    return varRandomParameters


def MStep(arrOfNormWeights, varList, varRandomParameters):
    '''
    need to sum all the things where A = a, and divide by the total weight
    after that, update the parameter of A = a to equal the result of the line above
    ------
    # for loop over all lines of data, so like 200 here or something.
    
    create an array of arrays, each index represents a variable.
        or every index holds the running weight of the things this var has appeared in
    
    for each line:
        get a running total of the total weight
        update the array of arrays thing from above
    return the final varRandomParameter list once done with every line
    '''
    finalWeight = 0
    var = [[]] * len(varRandomParameters)  # weight of the occurences of each variable, per parameter
    var2 = [[]] * len(varRandomParameters)  # weight of occurences of each parents of each varaible, per parent set
    for i in range(len(varRandomParameters)):
        lenJ = len(varRandomParameters[i])
        temp = [0] * lenJ
        var[i] = copy.deepcopy(temp)
        var2[i] = copy.deepcopy(temp)
    for i in range(len(arrOfNormWeights)):  # for each of the normalized weights
        arr = arrOfNormWeights[i][0]
        currWeight = Decimal(arrOfNormWeights[i][1])
        for j in range(len(varList)):  # for each of the variables
            parBinary = getBinaryOfParents(varList, j, arr)
            parIndex = int(parBinary, 2)
            var2[j][parIndex] = var2[j][parIndex] + currWeight
            if int(arr[j]) == 0:
                var[j][parIndex] = Decimal(var[j][parIndex]) + currWeight
        finalWeight = finalWeight + currWeight

    for i in range(len(varList)):
        for j in range(len(var[i])):
            # weight = Decimal(var[i][j] + 1)/Decimal(finalWeight + 2)
            weight = Decimal(var[i][j] + 1) / Decimal(var2[i][j] + 2)
            if weight < 0.00001:
                weight = Decimal(0.00001)
            elif weight == 1.0:
                weight = Decimal(0.99999)
            var[i][j] = Decimal(weight)
    '''for i in range(len(varList)):
        print(var[i])
    quit()'''
    return var


def EStep(data, varList, varRandomParameters):
    arrOfNormWeights = []
    for i in range(len(data)):
        arr = data[i].split()
        #  for each variable, go in to calculate the parameters associated with it
        arrOfWeightPerVar = []  # this array holds all weights
        if '?' in arr:  # if any of its parents were missing
            quesCount = arr.count('?')
            arrOfWeights = []
            normalizeMe = []
            for j in range(2 ** quesCount):
                #  for each value of ?, need to make 2 differnt data things, one for if the ? was 0 or 1
                #  here i need to get every permutation of the ? for the array parBinary
                temp = questionMarkProcedure(arr, varRandomParameters, j, varList, quesCount)
                #  question mark procedure output is correct
                normalizeMe.append(temp[1])
                arrOfWeights.append(temp)
            finalWeightsForJ = normalizeProbabilities(normalizeMe)

            finalReturn = []
            for j in range(len(arrOfWeights)):
                finalReturn.append((arrOfWeights[j][0], finalWeightsForJ[j]))
            arrOfNormWeights.extend(finalReturn)
        else:  # the case where all the parents are not missing
            #  todo not even sure if this is going to be used but fix it anyways incase it is
            parIndex = int(arr, 2)
            # weightForJ = factors[j][parIndex]

            '''if not arr[i] == "0":
                weightForJ = 1 - weightForJ
            arrOfWeightPerVar.append((arr, weightForJ))'''
    return arrOfNormWeights


def getBinaryOfParents(varList, indx, arr):
    '''
    Retrieves the parents of indx from varList
    :param varList: the list of parents for each variable
    :param indx: the variable i want to retrieve the parents of, 1...n
    :param arr: a binary number in array form
    :return: a binary number to so that you can retrieve the value of a var from the parameterValues
    '''
    parBinary = ""  # the binary value of the parents of the variable
    parents = varList[indx]  # the parents of the variable
    for t in range(len(parents)):
        parBinary = parBinary + arr[int(parents[t])]
    if parBinary == "":  # the case where the var has no parents
        parBinary = "0"
    return parBinary


def questionMarkProcedure(arr, varRandomParameters, j, varList, quesCount):
    """
    calculate the weight of the current permutation
    :param arr: current data line, used for indices of ?
    :param varRandomParameters: random values for parameterrs, used to get weights for all ?
    :param varList: list of all varible parents
    :param j: current iteration of outer loop, an int, used to get permutation of binary numbers
    :param quesCount the number of quesiton marks in arr
    :return: a tuple, (current permutation of data line w/o ?, weight of current permutation)
    """
    #  turn the 't' value into a binary number
    #  for every iteration of the loop, set the ?'s in the array to the binary number
    #    do this by getting the index of every ?, and looping over to insert 1 by 1
    #  save the result in the newDataPoints array
    arrToHoldProbToBeMultiplied = []
    newBin = arr.copy()  # the data array that contains the 0's, 1's, and ?'s
    recordOfQuestionMarks = newBin.copy()
    tempString = "\'{0:0" + str(len(str(bin((2 ** quesCount) - 1))[2:])) + "b}\'"
    binT = tempString.format(j)[1:-1]  # the binary value to indicate what the question marks should be
    quesMarkCounter = 0
    for o in range(len(newBin)):
        if newBin[o] == '?':
            newBin[o] = binT[quesMarkCounter]
            quesMarkCounter = quesMarkCounter + 1

    #  now every ? is a number
    valOfCurrData = 1  # supposed to be the un-normalized weight of the current permutation of data for j
    for o in range(len(newBin)):
        parents = getBinaryOfParents(varList, o, newBin)
        if int(newBin[o]) == 0:
            valOfCurrData = Decimal(valOfCurrData) * Decimal(varRandomParameters[o][int(parents, 2)])
        else:
            valOfCurrData = Decimal(valOfCurrData) * Decimal(1 - Decimal(varRandomParameters[o][int(parents, 2)]))
    return newBin, valOfCurrData


def normalizeProbabilities(nums):
    """
    :param nums: take in a series of numbers
    :return: return that series of nubmers, normalized
    """
    returnMe = 1 / sum(nums)
    return [returnMe * x for x in nums]


def complete_incomplete_Data():
    '''
    complete incomplete data using current parameters
    :return:
    '''
    print("hello")


def define_edges(UAI):
    # edges = list(reader.edges)
    # variables = reader.variables
    '''tempMe = []
    for i in edges:
        print(i, end='')
        tempMe.append(i)
    print("")
    edges = tempMe
    tempedge = dict.fromkeys(edges)
    edges = list(edges)
    vars = [-1] * len(variables)
    vars2 = [-1] * len(variables)
    print(tempedge)
    print(edges)

    for i in range(len(edges) - 1):
        temp = vars[int(name(edges[i][0]))]
        if temp == -1:
            temp = [int(name(edges[i][1]))]
        else:
            temp.append(int(name(edges[i][1])))
        vars[int(name(edges[i][0]))] = temp'''

    f = open(UAI, "r")
    x = f.readline()
    x = f.readline()
    num_vars = int(x)
    variables = []
    x = f.readline()
    x = f.readline()
    for i in range(num_vars):
        x = f.readline().split()
        x.pop(0)
        x.pop(-1)
        variables.append(x)
    return variables


def name(string):
    varName = string
    if not string[0].isdigit():
        varName = string[4:]
    return varName


def data_reader(file_name):
    f = open(file_name, "r")
    x = f.readline()
    num_vars = x.split()[0]
    num_examples = x.split()[1]
    data = []

    for i in range(int(num_examples)):
        x = f.readline().strip()
        data.append(x)
    return num_examples, num_vars, data


def checkFilesScientific(location, num):
    '''
    goes through the file, and for any scientific number replaces it with a float
    opens a new file to replace the scientific numbers
    returns the location of the new file
    :param location: first file
    :param num: number of the temp file
    :return: location of new file
    '''
    name = "temp" + num + ".uai"
    temp = open(name, "w")
    f = open(location, "r")
    counter = 0
    line = f.readline()
    temp.write(line)
    while True:
        line = f.readline()
        if not line:
            break
        tempLine = ""
        if 'E' in line:
            yur = line.split()
            for i in yur:
                if "-" in i:
                    tempme = "{:.11f}".format(float(i))
                elif "+":
                    tempme = float(i)
                tempLine = tempLine + " " + str(tempme)
            line = tempLine.strip() + "\n"
        temp.write(line)
        counter = counter + 1
    return name


def standardDeviation(mean, resultArr):
    runningMean = 0
    for i in range(len(resultArr)):
        x = Decimal(resultArr[i] - mean)
        x = x ** 2
        runningMean = x + runningMean
    runningMean = runningMean / len(resultArr)
    return math.sqrt(runningMean)


def pointWiseDiff(var, factors, thetaStar, runType, printType, data_file, resultsFile, testData):
    final = 0
    '''print(var[0])
    testMe = testData[0].split()
    print(testMe)
    #  x = str(testMe[0][70]) + str(testMe[0][19]) + str(testMe[0][5]) + str(testMe[0][54]) + str(testMe[0][64])
    x = str(testMe[70]) + str(testMe[19]) + str(testMe[5]) + str(testMe[54]) + str(testMe[64])
    print(x)
    print(int(x, 2))
    print(thetaStar[2])
    print(factors[2])'''
    '''print(thetaStar)
    print(factors)

    for i in range(int(len(thetaStar))):
        print(i, ": ", thetaStar[i])

    print(testData[0])'''
    '''for i in range(int(len(thetaStar))):
        print(i, ": ", thetaStar[i])'''
    # quit()
    # print(thetaStar)
    # print(factors)
    arrOfResults = []
    counter = 0
    for i in range(len(testData)):
        arr = testData[i].split()
        # trying to count the number of times each tuple appears in this one line of data.
        # per variable, save the parIndex, and then use that to grab the theta of uai and learned
        # if the variable was 0, use the value directly. if the variable was 1, take the inverse.
        parIndices = []
        for j in range(len(var)):
            parBinary = getBinaryOfParents(var, j, arr)
            parIndex = int(parBinary, 2)
            r = int(arr[j])
            parIndices.append((parIndex, r))
            # sum up the number of times i saw the tuple parArray
        result = LL(parIndices, var, thetaStar, factors)
        arrOfResults.append(result)
        final = final + result
        counter = counter + 1
        '''if i == 3000:
            mean = final / (i + 1)
            print(runType + " " + data_file + " " + str(i))
            Avg = "Avg: " + str(mean)
            # fin = "fin: " + str(final)
            finLog = "finLog: " + str(Decimal(200000) * mean)
            std = "Std: " + str(standardDeviation(mean, arrOfResults))
            # Avg = "Avg: " + str(final / (i + 1))
            # fin = "fin: " + str(final)
            # finLogged = "finLogged: " + str(math.log10(Decimal(200000) * (final / (i + 1))))
            print(Avg)
            # print(fin)
            print(finLog)
            print(std)
            # print(finLogged)
            if printType == 'True':
                g = open(resultsFile, "a")
                portalocker.lock(g, portalocker.LOCK_EX)
                g.write("\n")
                g.write(runType + " " + data_file + " " + str(i))
                g.write("\n")
                g.write(Avg)
                g.write("\n")
                g.write(finLog)
                g.write("\n")
                g.write(std)
                g.write("\n")
                g.close()
            break'''
    return final / counter


if __name__ == '__main__':
    if len(sys.argv) != 9:
        raise ValueError("Please provide configuration", len(sys.argv))
    runType = sys.argv[1]
    printType = sys.argv[2]
    UAI = sys.argv[3]
    data_file = sys.argv[4]
    test_data_file = sys.argv[5]
    resultsFile = sys.argv[6]
    k = int(sys.argv[7])
    happyRunTimes = sys.argv[8]
    readTime = time.time()
    print("Reading...")
    # UAI = "1.uai"
    # data_file = "train-p-4.txt"
    # test_data_file = "test.txt"
    UAI = checkFilesScientific(UAI, happyRunTimes)
    data_read_return = data_reader(data_file)
    test_data_read_return = data_reader(test_data_file)
    data = data_read_return[2]
    testData = test_data_read_return[2]
    # reader = UAIReader(UAI)
    # factors = reader.tables

    if test_data_file == 'set2/test.txt':
        f = open("factorsUAI2.txt", "r")
        factors = f.readline()
        f.close()
        factors = ast.literal_eval(factors)
    elif test_data_file == 'set1/test.txt':
        f = open("factorsUAI1.txt", "r")
        factors = f.readline()
        f.close()
        factors = ast.literal_eval(factors)
    elif test_data_file == 'set3/test.txt':
        f = open("factorsUAI3.txt", "r")
        factors = f.readline()
        f.close()
        factors = ast.literal_eval(factors)
    else:
        factors = 0
        print("wrong data file. do sumthin nerd")
        quit()

    print("Done reading: ", time.time() - readTime)
    name = "temp" + happyRunTimes + ".uai"
    varList = define_edges(UAI)
    os.remove(name)

    if runType == '1':
        # p = Pool(processes=3)
        testData = [testData[0:2500], testData[2501:5000], testData[5001:7500]]
        # testData = [testData[0:1000], testData[1001:2000], testData[2001:3000]]
        learnTime = time.time()
        print("learning....")
        thetaStar = FOD_learn(data, varList)
        print("Learn time: ", time.time() - learnTime)
        testTime = time.time()
        print("testing....")
        pool = multiprocessing.Pool(processes=3)
        help1 = partial(pointWiseDiff, varList, factors, thetaStar, runType, printType, data_file, resultsFile)
        final = pool.map(help1, testData)
        print(final)
        # final = pointWiseDiff(varList, testData, factors, thetaStar, runType, printType, data_file, resultsFile)
        print("Testing time: ", time.time() - testTime)

    elif runType == '2':
        for i in range(2):
            learnTime = time.time()
            print("learning....")
            thetaStar = POD_EM_Learn(data, varList)
            print("Learn time: ", time.time() - learnTime)
            print("testing....")
            pointWiseDiff(varList, testData, factors, thetaStar, runType, printType, data_file, resultsFile)
    elif runType == '3':
        for i in range(2):
            learnTime = time.time()
            print("learning....")
            thetaStar = task3(len(varList), k, data)
            print("Learn time: ", time.time() - learnTime)
            testTime = time.time()
            print("testing....")
            pointWiseDiffTask3(varList, testData, factors, thetaStar, runType, printType, data_file, resultsFile)
            print("Testing time: ", time.time() - testTime)