import network2
import mnist_loader
import pylab

if __name__ == '__main__':
    net = network2.Network([784,30,10], cost=network2.CrossEntropyCost)
    training_set, validation_set, testing_set = mnist_loader.load_data_wrapper()
    epoches = 38
    minbatchsize = 5
    eta = 0.1
    lambd = 0.8
    testcost, testaccuracy, trainingcost, trainingaccuracy = net.SGD(training_set, epoches, minbatchsize, eta, lambd, validation_set, False, True, True, True)
    pylab.plot(range(1, epoches+1), testaccuracy, '-r', label = 'testaccuracy')
    pylab.plot(range(1, epoches+1), trainingaccuracy, '-g', label = 'trainingaccuracy')
    # pylab.plot(range(1, epoches+1), trainingcost, '-b', label = 'trainingcost')
    # pylab.plot(elapsetime, testaccuracy)
    pylab.ylim(0.8,1.0)
    pylab.show()