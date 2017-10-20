import network
import mnist_loader
import pylab

if __name__ == '__main__':
    net = network.Network([784,30,10])
    training_set, validation_set, testing_set = mnist_loader.load_data_wrapper()
    print len(training_set)
    print len(validation_set)
    print len(testing_set)
    epoches = 1000
    minbatchsize = 1
    eta = 0.5
    trainingcost, testaccuracy, elapsetime, realepochs = net.SGD(training_set, epoches, minbatchsize, eta, 10, validation_set)
    pylab.plot(range(1, realepochs+1), testaccuracy)
    # pylab.plot(elapsetime, testaccuracy)
    pylab.show()