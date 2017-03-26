import numpy as np
import cPickle as pickle

class InferenceNet:
  ip0_w = None
  ip0_b = None
  ip0_next_w = None
  ip0_next_b = None
  ip1_w = None
  ip1_b = None
  outs = None

  def __init__(self, params):
    self.ip0_w = params[0][0]
    self.ip0_b = params[0][1]
    self.ip0_next_w = params[1][0]
    self.ip0_next_b = params[1][1]
    self.ip1_w = params[2][0]
    self.ip1_b = params[2][1]

  def relu(self, x):
    x[x<0] = 0
    return x

  def predict(self, x):
    fwd = self.relu(np.dot(x, self.ip0_w.T) + self.ip0_b)
    fwd = self.relu(np.dot(fwd, self.ip0_next_w.T) + self.ip0_next_b)
    fwd = np.dot(fwd, self.ip1_w.T) + self.ip1_b
    num = np.exp(fwd)
    den = np.sum(num, axis=1)
    softmax_out = np.array([num[i] / den[i] for i in xrange(den.shape[0])])
    outs = softmax_out.argmax(axis=1)
    return outs


if __name__ == '__main__':
  net_params = pickle.load(open('inference_net.pickle'))
  nn = InferenceNet(net_params)
  preds = nn.predict(x)