import numpy as np
import matplotlib.pyplot as plt


def random_sampling(n_sample, logweight_tensor_, n_landmark):
  t_weight = sum(np.exp(logweight_tensor_))
  print(t_weight)
  t_weight = np.power(t_weight, 0.0001);

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
        tw += np.exp(logweight_tensor_[j]);
        if(r < tw):
          selected[j] = True
          landmark_indices.append(j)
          running_t_weight += np.exp(logweight_tensor_[j]);

          break
    n_count += 1

  return landmark_indices

def main():
    X = np.loadtxt("colvar-1D.data")
    N = len(X)
    loc = [row[1] for row in X]
    lw = [row[2] for row in X]

    # XX = np.loadtxt("colvar-2D.data")
    # X = XX[:, [2, 3, 51]]  # phi, psi, mtd.rbias
    # N = len(X)
    # lw = [row[2] for row in X]


    ##data2 = [X[i] for i in range(1, 10)]
    ##loc = [row[1] for row in data2]
    ##lw = [row[2] for row in data2]

    n_sample = len(X)
    print("num: ", n_sample, "\n")
    logweight_tensor_ = lw
    print(logweight_tensor_)
    plt.hist(X[:, 1], bins=100, alpha=0.5, density=True, label=N);
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('colvar-1D.data')
    plt.xlim([0, 10])
    plt.legend(loc='upper right')
    plt.show();

    for k in range(1, 3):
        S = 100 * k
        ndx = random_sampling(n_sample, logweight_tensor_, S)
        print("--",ndx)
        ndx.sort();
        print(X.shape);
        A = np.empty(shape=[S, 3]);
        print(len(A))
        print(S)
        print(len(ndx))
        for i in range(len(ndx)):
            print(i);
            A[i] = (X[ndx[i]]);
        plt.hist(A[:, 1], bins=100, alpha=0.5, density=True, label=S);
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('colvar-1D.data')
        plt.xlim([0, 10])
        plt.legend(loc='upper right')
        plt.show();


if __name__ == "__main__":
    main()
