from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer


def main():
    network = buildNetwork(256, 200, 100, 10, bias=True, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)

    # parse training data
    train_data, test_data = split_dataset("data/semeion.data", 0.5)

    train_network(network, train_data)

    return None

# splits training data in to training data and testing data 
def split_dataset(data_path, ratio):
    dataset = ClassificationDataSet(256, 10)
    with open("../data/semeion.data") as data:
        for record in data:
            line = record[:1812]
            line = line.replace(' ', ', ')

            data = line[:2046]
            dataList = data.split(',')
            dataList = map(float, dataList)

            ans = line[2048:-2]
            ansList = ans.split(',')
            ansList = map(int, ansList)

            dataset.appendLinked(dataList, ansList)

    train_data, test_data = dataset.splitWithProportion(ratio)
    return train_data, test_data

# trains network using a back propagation trainer 
def train_network(network, train_data):
    trainer = BackpropTrainer(network, dataset=train_data, learningrate=0.02, momentum=0.4, verbose=True, weightdecay=0.01)
    trainer.trainUntilConvergence(maxEpochs=7)

def save_network():
    # TODO
    return None

def load_network():
    # TODO
    return None

if __name__ == '__main__':
    main()