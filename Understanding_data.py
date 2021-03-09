import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)

df_cancer = pd.read_csv("breastcancerwisconsin.csv")
df_cancer =df_cancer.drop("Code_number",axis=1)
df_cancer = df_cancer.replace("?",np.nan)
df_cancer = df_cancer.dropna()

a = df_cancer.drop("Class",axis=1)



plt.matshow(a.corr(), fignum=1,cmap="Blues_r")
plt.xticks(range(len(a.corr().columns)), a.corr().columns, fontsize=10, rotation=30)
plt.yticks(range(len(a.corr().columns)), a.corr().columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

for i in range(len(a.corr().columns)):
        for j in range(len(a.corr().columns)):
            plt.text(j, i, np.around(a.corr().iloc[i, j], decimals=1),
                     ha="center", va="center_baseline", color="black")

plt.show()

train_set = df_cancer.sample(frac=0.8)
train_input = np.array(train_set.drop("Class",axis=1))
train_label = np.array(train_set["Class"])


test_set = df_cancer.drop(train_set.index)
test_input = np.array(test_set.drop("Class",axis=1))
test_label = np.array(test_set["Class"])


train_label = np.transpose([train_label])
test_label = np.transpose([test_label])

def plot_loss_accuracy(accuracy_array, loss_array):
    plt.subplot(2, 1, 1)
    plt.plot(loss_array)
    plt.ylabel("Loss")
    plt.subplot(2,1,2)
    plt.plot(accuracy_array)
    plt.xlabel("# Ephocs")
    plt.ylabel("Accuracy")
    plt.show()

def run_on_test_set(test_input, weights,test_label):
    tp = 0
    test_output = sigmoid(np.dot(test_input.astype(int), weights))
    test_prediction = np.round(test_output)
    for predicted_val, label in zip(test_prediction,test_label):
        if predicted_val == label:
            tp += 1
    accuracy = tp / len(list(test_label))
    return accuracy

def main():
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    for iteration in range(iteration_count):
        outputs = np.dot(train_input.astype(int), weights)
        outputs = sigmoid(outputs)
        loss = train_label - outputs
        tunning = loss * sigmoid_derivative(outputs)
        weights += np.dot(np.transpose(train_input).astype(int), tunning)
        accuracy_array.append(run_on_test_set(test_input,weights,test_label))
        loss_array.append(np.mean(loss))
    plot_loss_accuracy(accuracy_array,loss_array)


if __name__ == '__main__':
    main()

