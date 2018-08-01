"""
Linear classifier.

Usage:
    net.py [--num=NUM] [--move=MOV] [--epoch=EPOCH] [--lr=LR] [--init_w W] [--val=VAL]
    net.py -h | --help

Options:
    -h, --help      : Show this screen.
    --num=NUM        : The number of data per class. [default: 500]
    --move=MOV       : The distance of the data distribution. [default: 2]
    --epoch=EPOCH      : The number of epoch to train on. [default: 10]
    --lr=LR       : Learning rate. [default: 0.05]
    --init_w=[W0 W1 W2]    : Initialize weight value.
    --val=VAL        : The value of separate validation rate from data. [default: 4]
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

class linear_model():
    def __init__(self, init_w, eps=1e-7):
        self.w = init_w
        self.line = lambda x: -(self.w[0]+self.w[1]*x)/(self.w[2]+eps)
        self.y = lambda x: 1*(np.dot(self.w.T, x)>0)

    def run(self, data):
        self.label = data[-1]
        self.x = np.r_[[1.], data[:2]]
        self.res = self.y(self.x)

    def update(self, lr):
        t = self.res - self.label
        self.w += -lr*t*self.x

class logistic_model(linear_model):
    def __init__(self, init_w):
        self.y = lambda x: self.sigmoid(np.dot(self.w.T, x))
    def sigmoid(x): return 1 / (1 + np.exp(-x))

class trainer():
    def __init__(self, model, epoch, lr):
        self.model = model
        self.epoch = epoch
        self.lr = lr

    def set_data(self, data_n, move, val):
        self.val = val
        v_n = int(data_n/val)
        rand = lambda x: np.random.randn(data_n) + x
        self.cx_0, self.cy_0 = rand(move), rand(move)
        self.cx_1, self.cy_1 = rand(-move), rand(-move)
        pos, neg = np.ones([data_n]), np.zeros([data_n])
        c0_data = np.c_[self.cx_0, self.cy_0, pos]
        c1_data = np.c_[self.cx_1, self.cy_1, neg]
        self.all_data = np.r_[c0_data, c1_data]
        shuffled = np.random.permutation(self.all_data)
        self.cross_data = lambda i: (shuffled[i*v_n:(i+1)*v_n], np.r_[shuffled[:i*v_n], shuffled[(i+1)*v_n:]])
        self.val_data, self.train_data = shuffled[:v_n], shuffled[v_n:]

    def display_plot(self):
        plt.clf()
        plt.scatter(self.cx_0,self.cy_0, facecolors='none', edgecolors='orange')
        plt.scatter(self.cx_1,self.cy_1, facecolors='none', edgecolors='skyblue')
        all_x = self.all_data[:, 0]
        x_ = np.arange(np.min(all_x),np.max(all_x))
        plt.plot(x_, self.model.line(x_), color='purple')
        plt.show()
    
    class summary_writer():
        def __init__(self):
            self.acc_sum = np.zeros((0), dtype=bool)
            self.acc = np.zeros((0))
        def clear(self):
            self.acc_sum = np.zeros((0), dtype=bool)
        def update(self, summary):
            self.acc_sum = np.append(self.acc_sum, summary)
        def write(self):
            accuracy = np.mean( 1 * self.acc_sum )
            self.acc = np.append(self.acc, accuracy)

    def train(self):
        try: self.all_data
        except AttributeError:
            print('train data not found')
        train_writer = self.summary_writer()
        valid_writer = self.summary_writer()
        for i in range(self.epoch):
            train_writer.clear()
            valid_writer.clear()
            for ind in range(self.val):
                self.train_data, self.val_data = self.cross_data(ind)
                for t_d in self.train_data:
                    self.model.run(t_d)
                    train_writer.update(self.model.res == self.model.label)
                    self.model.update(self.lr)
                for v_d in self.val_data:
                    self.model.run(v_d)
                    valid_writer.update(self.model.res == self.model.label)
            train_writer.write()
            valid_writer.write()

        plt.plot(np.arange(0,self.epoch),train_writer.acc)
        plt.plot(np.arange(0,self.epoch),valid_writer.acc)
        plt.show()

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    init_w = np.array(list(map(float,args['--init_w'].split())))
    model = linear_model(init_w)
    tr = trainer(model, int(args['--epoch']), float(args['--lr']))
    tr.set_data(int(args['--num']), float(args['--move']), int(args['--val']))
    tr.train()
    tr.display_plot()

