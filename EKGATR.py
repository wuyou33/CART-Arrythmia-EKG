import numpy as np
import random as rnd
import os as os




def Weight_Making(N_CLASS,data_training):
    dimension = np.shape(data_training)  # j,k
    #print(dimension)
    weight = np.zeros((N_CLASS, dimension[1]))
    for j in range(len(weight)):
        for k in range(len(weight[j])):
            weight[j][k] = rnd.uniform(-0.5,0.5)
    return weight

def training_matrice(weight,data,classify):
    for k in range(len(classify)):
        classify[k]=classify[k]+1
    H_init = np.matmul(data,np.transpose(weight))
    H = 1/(1+np.exp(np.negative(H_init)))
    H.dropna(inplace=True)
    #H=np.sin(H_init)
    H_pos = np.matmul(np.linalg.pinv(np.matmul(np.transpose(H),H)),np.transpose(H))
    output = np.matmul(H_pos,classify)
    #Y_predict = np.matmul(H,output)
    return output

def testing_matrice(weight,data,classify,bHat):
    for k in range(len(classify)):
        classify[k]=classify[k]+1
    H_init = np.matmul(data,np.transpose(weight))
    H = 1/(1+np.exp(np.negative(H_init)))
    H.dropna(inplace=True)
    #H = np.sin(H_init)
    H_pos = np.matmul(np.linalg.pinv(np.matmul(np.transpose(H),H)),np.transpose(H))
    output = np.matmul(H_pos,classify)
    Y_predict = np.matmul(H,output)
    return Y_predict

def MAPE(yHat,classify):
    mape=0
    for k in range(len(classify)):
        classify[k]=classify[k]+1
    for l in range(len(yHat)):
        mape += (abs(((yHat[l]-classify[l])/classify[l])*100))
    mape=(1/len(classify))*mape
    return mape

def training_seq(data,classify,N_CLASS):
    weight = Weight_Making(N_CLASS, data)
    train_result = training_matrice(weight, data, classify)
    bHat = train_result
    write_bHat(bHat)
    write_weight(weight)



def testing_seq(weight,data,classify,bHat):
    yHat = testing_matrice(weight,data,classify,bHat)
    return yHat

def accuracy(yHat,classify):
    correct =0
    for k in range(len(classify)):
        classify[k]=classify[k]+1
    for j in range(len(yHat)):
        yHat[j]=int((yHat[j])+1)
        for l in range(len(yHat)):
            if(yHat[l]==classify[l]):
                correct=correct+1
                l++1
        accur = float(correct/len(yHat))*100
        return accur

def write_weight(w):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_weight = dir_path + '/mitbih_weight.csv'
    np.savetxt(dir_path_weight, w, delimiter=";")

def write_bHat(b):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_weight = dir_path + '/mitbih_bHat.csv'
    np.savetxt(dir_path_weight, b, delimiter=";")

def main_training(min_mape,N_CLASS,ambang):
    mape_std=ambang
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_training = dir_path + '/testing.csv'
    dir_path_class = dir_path + '/tcl.csv'
    dir_path_w = dir_path + '/weight.csv'
    dir_path_b = dir_path + '/bhat.csv'
    trainer = np.genfromtxt(dir_path_training, delimiter=';')  # trainer
    classify = np.genfromtxt(dir_path_class, delimiter=';')  # class
    weight = np.genfromtxt(dir_path_w, delimiter=';')  # weight
    bHat = np.genfromtxt(dir_path_b, delimiter=';')  # b
    while(mape_std>min_mape):
        training_seq(trainer, classify, N_CLASS)
        yHat = testing_seq(weight,trainer,classify,bHat)
        mape = MAPE(yHat,classify)
        if(mape<mape_std):
            if(mape<=min_mape):
                print("DONE!\nMAPE = " + str(mape))
                break
            else:
                mape_std=mape
                print("MAPE = " + str(mape))

def write_weight(w):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_w = dir_path + '/weight.csv'
    np.savetxt(dir_path_w, w, delimiter=";")


def main_testing():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_testing = dir_path + '/testing.csv'
    dir_path_class = dir_path + '/tcl.csv'
    dir_path_w = dir_path + '/weight.csv'
    dir_path_b = dir_path + '/bhat.csv'
    tester = np.genfromtxt(dir_path_testing,encoding="utf8", delimiter=';')  # tester
    classify = np.genfromtxt(dir_path_class,encoding="utf8", delimiter=';')  # class
    weight = np.genfromtxt(dir_path_w, delimiter=';')  # weight
    for g in range(len(weight)) :
        for c in range (len(weight[g])):
            weight[g] = np.absolute(weight[g])
    bHat = np.genfromtxt(dir_path_b, delimiter=';')  # b
    y_hat=testing_seq(weight, tester, classify, bHat)
    accurate =accuracy(y_hat,classify)
    print("ACCURACY = "+str(accurate)+"%")



def main():
    answer = "YES"
    while(answer != "NO"):
        print("SELECT MODE : \n1.Training \n2.Testing \n<SELECT USING NUMBER>")
        choose= input()
        if(choose=="1"):
            N_CLASS = input("NUMBER OF HIDDEN LAYER(S)")
            min_mape = input("MINIMUM MAPE TARGET")
            threshold= input("STARTING MAPE THRESHOLD")
            N_CLASS=int(N_CLASS)
            min_mape=float(min_mape)
            threshold = float(threshold)
            print("BEGIN TRAINING...")
            main_training(min_mape,N_CLASS,threshold)
            print("BACK TO MAIN MENU?\nYES/NO")
            answer=input()
        elif(choose=="2"):
            main_testing()
            print("BACK TO MAIN MENU?\nYES/NO")
            answer = input()
        else:
            print("NO MODE CHOOSE\nPROGRAM WILL EXIT")
            exit()
main()






