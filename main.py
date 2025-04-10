from model import UnidimensionalNeuron
from optimizer import gradient_descent
from utils.viz import plot_training

if __name__ == "__main__":
    EPOCHS = 72
    DATASET = "data/data.csv"
    labels = []
    results = []

    sgd = UnidimensionalNeuron()
    result = gradient_descent(neuron=sgd, dataset=DATASET, epochs=EPOCHS, batch_size=1)
    results.append(result)
    labels.append("sgd")

    bgd = UnidimensionalNeuron()
    result_two = gradient_descent(bgd, DATASET, EPOCHS, batch_size=10)
    results.append(result_two)
    labels.append("bgd")

    mbgd = UnidimensionalNeuron()
    results_three = gradient_descent(mbgd, DATASET, EPOCHS, batch_size=3)
    results.append(results_three)
    labels.append("mbgd")

    sgd_acc = UnidimensionalNeuron()
    results_four = gradient_descent(sgd_acc, DATASET, EPOCHS, batch_size=1, accumulate=10)
    results.append(results_four)
    labels.append("sgd_acc")

    bgd_acc = UnidimensionalNeuron()
    results_five = gradient_descent(bgd_acc, DATASET, EPOCHS, batch_size=10, accumulate=1)
    results.append(results_five)
    labels.append("bgd_acc")

    mbgd_acc = UnidimensionalNeuron()
    results_six = gradient_descent(mbgd_acc, DATASET, EPOCHS, batch_size=3, accumulate=2)
    results.append(results_six)
    labels.append("mbgd_acc")

    plot_training(results, labels)