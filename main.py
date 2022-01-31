import numpy as np
import matplotlib.pyplot as plt


def random_sampling(n_sample, logweight_tensor_, n_landmark):
  t_weight = sum(np.exp(logweight_tensor_))
  running_t_weight = 0

  landmark_indices = []
  selected = np.full((n_sample), False, dtype=bool)

  n_count = 0

  while(n_count < n_landmark):
    tw = 0
    r01 = np.random.rand()
    r = (t_weight - running_t_weight) * r01

    for j in range(n_sample):
      if(selected[j] == False):
        tw += np.exp(logweight_tensor_[j])
        if(r < tw):
          selected[j] = True
          landmark_indices.append(j)
          running_t_weight += np.exp(logweight_tensor_[j])

          break
    n_count += 1

  return landmark_indices

def main():
    X = np.loadtxt("colvar-1D.data")
    N = len(X)
    loc = [row[1] for row in X]
    lw = [row[2] for row in X]

    ##data2 = [X[i] for i in range(1, 10)]
    ##loc = [row[1] for row in data2]
    ##lw = [row[2] for row in data2]

    n_sample = len(X)
    logweight_tensor_ = lw

    plt.hist(X[:, 1], bins=100, alpha=0.5, density=True, label=N);
    plt.xlim([0, 10])
    plt.legend(loc='upper right')
    plt.show();

    for k in range(1, 4):
        S = 1000 + 20000*k;
        ndx = random_sampling(n_sample, logweight_tensor_, S)
        print(ndx)
        ndx.sort();
        print(X.shape);
        A = np.empty(shape=[S, 3]);
        for i in range(S):
            A[i] = (X[ndx[i]]);
        plt.hist(A[:, 1], bins=100, alpha=0.5, density=True, label=S);
        plt.xlim([0, 10])
        plt.legend(loc='upper right')
        plt.show();


if __name__ == "__main__":
    main()
