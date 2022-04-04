import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans

# as a parameter 1 == 1D data, 2 = 2D data etc.
# 1D data: index, time p.x, mtd.rbias
# 2D data: index, phi, psi, mtd.rbias
def read_data(dimention):
    if dimention == 1:
        X = np.loadtxt("colvar-1D.data")
    elif dimention == 2:
        X_all = np.loadtxt("colvar-2D.data")
        ds = X_all[:, [2, 3, 51]]
        index = [i for i in range(len(ds))]
        X = np.c_[index, ds]
    else:
        print("No {} dimention data".format(dimention))
        X = []
    return X

# modified roulette selection //++ t_weight = np.power(t_weight, 0.0001);
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

def normalize_wages(X):
    log_w = X[:, -1]
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w

def scatter_2d_data(X):
    w = normalize_wages(X)
    fig, ax = plt.subplots()
    ax.set_title('scatter of normalized wages in 2D dataset')
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')

    s = ax.scatter(X[:, 1], X[:, 2], c=w, s=4)
    ax.set_xlim([np.amin(X[1]), np.amax(X[1])])
    ax.set_ylim([np.amin(X[2]), np.amax(X[2])])
    plt.colorbar(s)
    plt.show()

def plot_voronoi_diagram(X):
    vor = Voronoi(X[:, [1, 2]])
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.set_title('voronoi diagram for 2D dataset')
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    ax.set_xlim([np.amin(X[1]), np.amax(X[1])])
    ax.set_ylim([np.amin(X[2]), np.amax(X[2])])
    voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    plt.show()

def get_sin_cos(X):
    # X[index, phi, psi, mtd.rbias]
    sinphi, cosphi, sinpsi, cospsi = ([] for i in range(4))

    for c in range(len(X)):
        sinphi.append(math.sin(X[c, 1]))
        cosphi.append(math.cos(X[c, 1]))

        sinpsi.append(math.sin(X[c, 2]))
        cospsi.append(math.cos(X[c, 2]))

    # per_X[index,  sin phi, cos phi, sin psi, cos psi, mtd.rbias]
    per_X = np.c_[X[:, 0], sinphi, cosphi, sinpsi, cospsi, X[:, 3]]
    print(per_X[1])

def k_means(X, N_clusters):
    #N_clusters = 100

    # class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++',
    # n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
    km = KMeans(n_clusters=N_clusters)

    ##nowa wersja - zamiast phi psi to sinphi, cosphi, sinpsi, cospsi

    ## stara wersja
    y_pred = km.fit_predict(X[:, 1:3])
    new_X = np.c_[X, y_pred]
    w = normalize_wages(X)
    new_X[:, 3] = w  # normalized weights

    fig, ax = plt.subplots(figsize=(100, 100))
    ax.set_title('K-means clusterization ({} clusters) - 2D dataset'.format(N_clusters))
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    plt.scatter(new_X[:, 1], new_X[:, 2], c=new_X[:, 4])
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='green', marker='*')
    plt.show()

    ## dodatkowy diagram
    plot_voronoi_for_clusters(new_X, km)

    return new_X, km

def plot_voronoi_for_clusters(new_X, km):
    v_cluters = Voronoi(km.cluster_centers_[:, 0:2])

    fig, ax = plt.subplots(figsize=(100, 100))
    ax.set_title('Voronoi diagram for ({} clusters) - 2D dataset'.format(km.n_clusters))
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    ax.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 'o', color='k')
    ax.set_xlim([np.amin(new_X[1]), np.amax(new_X[1])])
    ax.set_ylim([np.amin(new_X[2]), np.amax(new_X[2])])
    voronoi_plot_2d(v_cluters, ax)
    plt.show()

# modified roulette selection //++ t_weight = np.power(t_weight, 0.0001);
def sample_clusters(n_sample, weights, n_landmark):
  t_weight = sum(weights)
  print(t_weight)
  #t_weight = np.power(t_weight, 0.0001);

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
        tw += np.exp(weights[j]);
        if(r < tw):
          selected[j] = True
          landmark_indices.append(j)
          running_t_weight += np.exp(weights[j]);

          break
    n_count += 1

  return landmark_indices

def show_which_clasters_are_picked(km, N_clusters):
    modified_clust_centres = km.cluster_centers_
    labels = range(1, N_clusters + 1)
    plt.rcParams['font.size'] = '14'

    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(modified_clust_centres[:, 0], modified_clust_centres[:, 1], label='True Position')

    for label, x, y in zip(labels, modified_clust_centres[0:N_clusters, 0], modified_clust_centres[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
    plt.show()

def random_sampling2(n_sample, logweight_tensor_, n_landmark, clust_points):
  t_weight = sum(np.exp(logweight_tensor_))
  print(t_weight)
  # t_weight = np.power(t_weight, 0.0001);

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
          landmark_indices.append(clust_points[j,0])
          running_t_weight += np.exp(logweight_tensor_[j]);

          break
    n_count += 1

  return landmark_indices

def sample_inside_cluster(new_X, N_clusters, sum_of_clust, S, SS):
    selected_points = np.full((len(new_X)), False, dtype=bool)
    indexes = []
    n_sample = N_clusters
    ##S = 20
    clust_ndx = sample_clusters(n_sample, sum_of_clust, S)
    print("\npicked clusters: (indexes) ", clust_ndx, "\n")
    for i in range(S):
        print("cluster index: ", clust_ndx[i])
        clust_points = new_X[np.where(new_X[:, 4] == clust_ndx[i])]
        print("number of points in this cluster: ", len(clust_points))

        n_sample = len(clust_points)
        print("n_sample: ", n_sample)
        logweight_tensor_ = clust_points[:, 2].tolist()
        print(type(logweight_tensor_))
        print("logweight_tensor_: ", logweight_tensor_)
        ##SS = 2000
        ndx = random_sampling2(n_sample, logweight_tensor_, SS, clust_points)
        indexes.extend(ndx)
        print("ndx (picked points from this cluster): ", ndx)
        print(len(ndx), "\n")

    print("\nall sampled points: ", indexes)
    print("number of points: ", len(indexes))
    indexes.sort()
    print("\nall sampled points sorted", indexes)
    return indexes

def plot_hist_2D(new_X):

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.set_title('Histogram of sampled data ({} clusters) - 2D dataset'.format(len(new_X)))
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')

    plt.hist2d(new_X[:, 1], new_X[:, 2], bins=(50, 50), cmap=plt.cm.jet, label=len(new_X))
    plt.colorbar()
    plt.show()


def main():
    X = read_data(2)
    #scatter_2d_data(X)
    #plot_voronoi_diagram(X)

    N_clusters = 100
    new_X, km = k_means(X, N_clusters) ##new_ X [index, phi, psi,  mtd.rbias, klaster]

    sum_of_clust = np.zeros(N_clusters)
    for i in range(len(new_X)):
        temp = int(new_X[i, 4])
        sum_of_clust[temp] += new_X[i, 3]

    ## picking clasters
    n_sample = N_clusters
    S = 20
    #clust_ndx = sample_clusters(n_sample, sum_of_clust, S)
    #print(clust_ndx)

    ##TODO test method - doesnt work correctly
    show_which_clasters_are_picked(km, S)

    ## sampling inside a cluster
    indexes = sample_inside_cluster(new_X, N_clusters, sum_of_clust, 20, 2000) # 20 clusters, 2000 samples

    plot_hist_2D(new_X)
    print(len(indexes))
    print(indexes)

    sampled_X = np.empty((0, 5), float)
    for i in indexes:
        sampled_X = np.vstack((sampled_X, new_X[int(i)]))
    plot_hist_2D(sampled_X)


if __name__ == "__main__":
    main()
