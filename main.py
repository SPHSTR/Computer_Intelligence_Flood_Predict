import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self,filename,hidden_layer_size,function,epoch,learning_rate,momentum_rate):
        self.filename = filename
        self.hidden_layer_size = hidden_layer_size
        self.function = function
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def Activate_Function(self,x):
        if self.function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.function == "relu":
            return np.where(x > 0, x, 0.0)
        if self.function == "tanh":
            return np.tanh(x)

    def Diff_Activate_Function(self,x):
        if self.function == "sigmoid":
            return x * (1-x)
        if self.function == "relu":
            return np.where(x > 0, 1.0, 0.0)
        if self.function == "tanh":
            return 1 - x**2

    def Load_data(self):
        file = np.loadtxt(self.filename , dtype=float )
        self.input = file[:, :-1]
        self.desireoutput = file[:, -1].reshape(-1, 1)
        
        self.filemax = file.max()
        self.filemin = file.min()
    
    def Random_WeightandBias(self):
        self.actual_layer = [self.input.shape[1]] + self.hidden_layer_size + [self.desireoutput.shape[1]]
        self.weight = []
        self.bias = []
        for i in range(len(self.actual_layer)-1):
            self.weight.append( np.random.randn(self.actual_layer[i], self.actual_layer[i+1]) - 1)
            self.bias.append(np.zeros((1, self.actual_layer[i+1])))

        self.veloc_weight = []
        self.veloc_bias = []
        for i in range(len(self.actual_layer)-1):
            self.veloc_weight.append(np.zeros_like(self.weight[i]))
            self.veloc_bias.append(np.zeros_like(self.bias[i]))

    def Normallization(self):
        self.input = (self.input - self.filemin) / (self.filemax-self.filemin)
        self.desireoutput = (self.desireoutput - self.filemin) / (self.filemax-self.filemin)

    def DeNormallization(self):
        self.input = (self.input * (self.filemax-self.filemin) ) + self.filemin
        self.desireoutput = (self.desireoutput * (self.filemax-self.filemin) ) + self.filemin

    def Value_DeNormallization(self,x):
        return (x * (self.filemax-self.filemin) ) + self.filemin

    def Feed_Forward(self):
        self.layer_output = [self.input]
        for i in range(len(self.weight)):
            net_input = np.dot(self.layer_output[-1], self.weight[i]) + self.bias[i]
            activation_output = self.Activate_Function(net_input)
            self.layer_output.append(activation_output)
        
    def Back_Propagation(self):
        self.errors = [self.desireoutput - self.layer_output[-1]]
        deltas = [self.errors[-1] * self.Diff_Activate_Function(self.layer_output[-1])]

        for i in range(len(self.weight) - 1, 0, -1):
            error = np.dot(deltas[-1], self.weight[i].T)
            delta = error * self.Diff_Activate_Function(self.layer_output[i])
            self.errors.append(error)
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weight)):
            self.veloc_weight[i] = self.momentum_rate * self.veloc_weight[i] + np.dot(self.layer_output[i].T, deltas[i])
            self.veloc_bias[i] = self.momentum_rate * self.veloc_bias[i] + np.sum(deltas[i], axis=0, keepdims=True)
            self.weight[i] += self.learning_rate * self.veloc_weight[i]
            self.bias[i] += self.learning_rate * self.veloc_bias[i]
    
    def Train(self):
        self.training_loss = []
        for i in range(self.epoch):
            self.Feed_Forward()
            self.Back_Propagation()
            loss = np.mean(np.square(self.desireoutput - self.layer_output[-1]))
            self.training_loss.append(loss)
            print("Epoch:",i+1," Loss:",loss)
        self.DeNormallization()
    
    def Test(self, t):
        x = (t - self.filemin) / (self.filemax - self.filemin)
        for i in range(len(self.weight)):
            x = self.Activate_Function(np.dot(x, self.weight[i]) + self.bias[i])
        print("Test input =", t ,"Output =",self.Value_DeNormallization(x))

    def K_Fold_Cross_Validation(self, k):
        self.Load_data()
        self.Normallization()
        data = np.hstack((self.input, self.desireoutput))
        np.random.shuffle(data)
        folds = np.array_split(data, k)
        display = []

        plt.figure(figsize=(12, 8))
        
        for fold in range(k):
            print(f"Fold {fold + 1}/{k}")
            
            train_data = np.vstack([folds[i] for i in range(k) if i != fold])
            test_data = folds[fold]
            
            self.input = train_data[:, :-1]
            self.desireoutput = train_data[:, -1].reshape(-1, 1)
            self.Random_WeightandBias()
            
            self.Train()

            plt.plot(self.training_loss, label=f'Fold {fold + 1}')
            
            test_input = test_data[:, :-1]
            test_output = test_data[:, -1].reshape(-1, 1)
            self.input = test_input
            self.desireoutput = test_output
            self.Feed_Forward()
            test_loss = np.mean(np.square(self.desireoutput - self.layer_output[-1]))
            print("Test Loss:", test_loss)
            display.append(test_loss)
            print()

        plt.title('Training Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, k + 1), display)
        plt.title('Test Loss for Each Fold')
        plt.xlabel('Fold')
        plt.ylabel('Test Loss')
        plt.show()

        for i in range(k):
            print("Fold",i+1," Loss :",display[i])

        self.DeNormallization()




if __name__ == '__main__':
    # x = NN("Flood_dataset.txt",[2,2],"sigmoid",10000,0.05,0.9)
    # x = NN("Flood_dataset.txt",[5,10,50],"sigmoid",10000,0.034,0.91)
    # x = NN("Flood_dataset.txt",[5,10,50],"sigmoid",1000,0.035,0.91)
    x = NN("Flood_dataset.txt",[5,10,50],"sigmoid",10000,0.035,0.91)
    
    x.K_Fold_Cross_Validation(10)

    x.Test([100,100,101,101,153,153,153,153])
    x.Test([385,445,465,514,290,297,303,314])