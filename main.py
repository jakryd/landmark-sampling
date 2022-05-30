import sys
import math
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import pdist, squareform
import pylab as pl

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

def get_rand_sample(X, size):
    return X[np.random.randint(X.shape[0], size=size), :]

def sort_arr_by_index(X):
    return X[X[:, 0].argsort()]

# modified roulette selection //++ t_weight = np.power(t_weight, 0.0001);
def random_sampling(n_sample, logweight_tensor_, n_landmark):
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
    ax.set_title('Scatterplot for 2D dataset - normalized wages ')
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')

    s = ax.scatter(X[:, 1], X[:, 2], c=w, s=100)
    ax.set_xlim([np.amin(X[1]), np.amax(X[1])])
    ax.set_ylim([np.amin(X[2]), np.amax(X[2])])
    plt.colorbar(s)
    # plt.savefig('outputs/3.png')
    plt.show()

def plot_voronoi_diagram(X):
    vor = Voronoi(X[:, [1, 2]])
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.set_title('Voronoi diagram for 2D dataset')
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    ax.set_xlim([np.amin(X[1]), np.amax(X[1])])
    ax.set_ylim([np.amin(X[2]), np.amax(X[2])])
    voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2, s=100)
    # plt.savefig('outputs/5.png')
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
    print(X[1])
    print(per_X[1])
    return per_X

def k_means_stara_wersja(X, N_clusters):
    # plt.scatter(X[:, 1], X[:, 2])

    # class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++',
    # n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
    km = KMeans(n_clusters=N_clusters)

    ##nowa wersja - zamiast phi psi to sinphi, cosphi, sinpsi, cospsi #TODO

    ## stara wersja
    y_pred = km.fit_predict(X[:, 1:3])
    new_X = np.c_[X, y_pred]
    # w = normalize_wages(X)
    # new_X[:, 3] = w  # normalized weights


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('K-means clusterization ({} clusters) - 2D dataset'.format(N_clusters))
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    # plt.scatter(new_X[:, 1], new_X[:, 2], c=new_X[:, 4])
    # plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='red', marker='*', s=100)
    # # plt.savefig('outputs/6.png')
    # plt.show()

    u_labels = np.unique(new_X[:,4])
    for i in u_labels:
        plt.scatter(new_X[new_X[:,4] == i, 1], new_X[new_X[:,4] == i, 2], label=int(i))
        # plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker='*', s=100)
        plt.annotate(int(i), km.cluster_centers_[int(i)], horizontalalignment='center', verticalalignment='center', size=16, weight='bold')
    # plt.legend()
    plt.show()

    ## dodatkowy diagram
    # plot_voronoi_for_clusters(new_X, km)
    # print(km.get_params())
    return new_X, km

def k_means_elbow(X,r=30):
    distortions = []
    K = range(1, r)
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(X[:, 1:3])
        distortions.append(km.inertia_)

    plt.plot(K, distortions, 'bx-')
    plt.title('Optymalna liczba klastrów - Metoda Elbow')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.show()

def get_centroids(X, labels, K):
    L = math.pi * 2
    centroids = []
    temp = []
    for k in range(K):
        for i in range(len(X)):
            if labels[i] == k:
                if X[i,0] >= L * 0.5:
                    X[i,0] -= L
                if X[i, 1] >= L * 0.5:
                    X[i, 1] -= L
                temp.append(X[i])
        centroids.append(np.mean(temp, axis=0))
        temp = []
    return np.array(centroids)

def k_means_per_and_nonper(X, K=5):
    L = math.pi * 2
    km = KMeans(n_clusters=K).fit(X[:, 1:3])
    plt.scatter(X[:, 1], X[:, 2], c=km.labels_, s=100)
    # plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, color='red', marker='*')
    plt.figure(figsize=(10, 10))
    plt.title('KMeans bez uwzględnienia periodyczności danych')
    plt.xlabel('phi')
    plt.ylabel('psi')

    #find the correct distance matrix
    for d in range(1,3):
        # all 1-d distances
        pd = pdist(X[:, d].reshape(len(X), 1))
        pd[pd > L * 0.5] -= L
        try:
            total += pd ** 2
        except:
            total = pd ** 2

    # transform the condensed distance matrix...
    total = pl.sqrt(total)
    # ...into a square distance matrix
    square = squareform(total)
    km2 = KMeans(n_clusters=K).fit(square)
    y_pred = km2.fit_predict(square)
    new_X = np.c_[X, y_pred]
    plt.scatter(X[:, 1], X[:, 2], c=km2.labels_, s=100)
    # centroids = get_centroids(X, km2.labels_, K)
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=100, color='red', marker='*')
    plt.title('KMeans z uwzględnieniem periodyczności danych')
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.show()
    return new_X, km

def dbscan_per_and_nonper(X, threshold = 0.3):
    L = math.pi * 2
    db = DBSCAN(eps=threshold).fit(X[:, 1:3])
    plt.scatter(X[:, 1], X[:, 2], c=db.labels_, s=100)
    plt.figure(figsize=(10, 10))
    plt.title('DBSCAN bez uwzględnienia periodyczności danych')
    plt.xlabel('phi')
    plt.ylabel('psi')

    for d in range(1,3):
        # all 1-d distances
        pd = pdist(X[:, d].reshape(len(X), 1))
        pd[pd > L * 0.5] -= L
        try:
            total += pd ** 2
        except:
            total = pd ** 2

    # transform the condensed distance matrix...
    total = pl.sqrt(total)
    # ...into a square distance matrix
    square = squareform(total)
    db2 = DBSCAN(eps=threshold, metric='precomputed').fit(square)
    plt.scatter(X[:, 1], X[:, 2], c=db2.labels_, s=100)
    plt.title('DBSCAN z uwzględnieniem periodyczności danych')
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.show()

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
    # plt.savefig('outputs/7.png')
    plt.show()

# modified roulette selection //++ t_weight = np.power(t_weight, 0.0001);
def sample_clusters(n_sample, weights, n_landmark):
  t_weight = sum(np.exp(weights))
  # print(t_weight)
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
        tw += np.exp(weights[j]);
        if(r < tw):
          selected[j] = True
          landmark_indices.append(j)
          running_t_weight += np.exp(weights[j]);
          break
    n_count += 1
  return landmark_indices

def sample_points(n_sample, logweight_tensor_, n_landmark, clust_points):
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

def sample_inside_clusters(new_X, N_clusters, sum_of_clust, pick_n_clusters, pick_n_samples):
    selected_points = np.full((len(new_X)), False, dtype=bool)
    print(len(selected_points))
    indexes = []
    n_sample = N_clusters
    sampled_clust_ndx = sample_clusters(n_sample, sum_of_clust, pick_n_clusters)
    print("\npicked clusters: (indexes) ", sampled_clust_ndx, "\n")
    show_picked_clust(new_X, sampled_clust_ndx)

    for i in range(pick_n_clusters):
        print("cluster index: ", sampled_clust_ndx[i])
        clust_points = new_X[np.where(new_X[:, 4] == sampled_clust_ndx[i])]
        print("number of points in this cluster: ", len(clust_points))

        n_sample = len(clust_points)
        print("n_sample: ", n_sample)
        logweight_tensor_ = clust_points[:, 2].tolist()
        print(type(logweight_tensor_))
        print("logweight_tensor_: ", logweight_tensor_)
        ndx = sample_points(n_sample, logweight_tensor_, pick_n_samples, clust_points)
        indexes.extend(ndx)
        print("ndx (picked points from this cluster): ", ndx)
        print(len(ndx), "\n")

    print("\nall sampled points: ", indexes)
    print("number of points: ", len(indexes))
    indexes.sort()
    print("\nall sampled points sorted", indexes)

    sampled_X = np.empty((0, 5), float)
    for i in indexes:
        sampled_X = np.vstack((sampled_X, new_X[new_X[:, 0] == i]))

    return sampled_X, indexes

def plot_hist_2D(new_X):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Histogram of sampled data ({} clusters) - 2D dataset'.format(len(new_X)))
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')

    plt.hist2d(new_X[:, 1], new_X[:, 2], bins=(50, 50), cmap=plt.cm.jet, label=len(new_X))
    plt.colorbar()
    plt.show()

# hist X - whole set, hist A - sampled set
def kl_divergence(HX, HA):
    epsilon = 0.0001
    X = HX + epsilon
    A = HA + epsilon

    s = 0
    d = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            temp = X[i][j] * log(X[i][j]/A[i][j])
            d.append(temp)
            s += temp
    print(d)
    print(sum(d))

    import matplotlib.pyplot as plt
    plt.plot(d)
    # plt.show()
    return s

def calc_KL_for_many_sets(X):
    sizeA = []
    KL = []
    for i in range(1, 17):
        print(i, i ** 4)
        A = get_rand_sample(X, i ** 4)

        H, xedges, yedges = np.histogram2d(X[:, 1], X[:, 2], bins=(100, 100))
        H = H.T
        # fig = plt.figure(figsize=(10, 20))
        # ax = fig.add_subplot(131, title='')
        # plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # plt.show()

        H2, xedges, yedges = np.histogram2d(A[:, 1], A[:, 2], bins=(100, 100))
        H2 = H2.T
        # fig = plt.figure(figsize=(10, 20))
        # ax = fig.add_subplot(131, title='')
        # plt.imshow(H2, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # plt.show()

        # (X || A)
        kl = kl_divergence(H, H2)
        print('KL(X || A=', len(A), '):', kl)
        sizeA.append(len(A))
        KL.append(kl)
    sizeA.append(len(X))
    KL.append(0.0)

    # print()
    # print(sizeA)
    # for i in range(17):
    #     #     max_val = sizeA[-1]
    #     #     sizeA[i] /= max_val
    #     # print(sizeA)
    # print(KL)
    plt.plot(sizeA, KL)
    plt.show()

def sum_wages_in_custers(X, N_clusters):
    sum_of_clust = np.zeros(N_clusters)
    for i in range(len(X)):
        temp = int(X[i, 4])
        sum_of_clust[temp] += X[i, 3]
    return sum_of_clust

def show_picked_clust(new_X,sampled_clust_ndx):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Picked clusters - 2D dataset')
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    for i in sampled_clust_ndx:
        j = i * 1.0
        plt.scatter(new_X[new_X[:, 4] == j, 1], new_X[new_X[:, 4] == j, 2], label=int(j))
        # plt.annotate(int(j), km.cluster_centers_[int(j)], horizontalalignment='center', verticalalignment='center', size=16, weight='bold')
    plt.show()

def overlay_plots(X, sampled_X):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Whole set (blue) vs sampled set (red)')
    ax.set_xlabel('phi')
    ax.set_ylabel('psi')
    plt.scatter(X[:, 1], X[:, 2], c='b', alpha=0.1)
    plt.scatter(sampled_X[:, 1], sampled_X[:, 2], c='r', alpha=1)
    plt.show()

def main(argv):
    if(sys.argv[1] == '1'):
        X = read_data(1)
    #argv: dane 2D, n_klastrow, ile klastrow, ile punktów; wykresy:
    elif(sys.argv[1] == '2'):
        X = read_data(2)
        N_clusters = int(sys.argv[2]) #10
        pick_n_clusters = int(sys.argv[3]) #6
        pick_n_samples = int(sys.argv[4]) #30

        A = get_rand_sample(X, 1000)
        sortedA = sort_arr_by_index(A)
        # scatter_2d_data(X)
        # scatter_2d_data(sortedA)
        # plot_voronoi_diagram(A)

        # -- KMEANS --
        # k_means_elbow(sortedA, 30)
        # X_per = get_sin_cos(A)
        new_X, km = k_means_per_and_nonper(A, pick_n_clusters)
        # dbscan_per_and_nonper(A, 0.2)

        # stara wersja kmeans
        # new_X, km = k_means_stara_wersja(A, N_clusters)  ##new_X [index, phi, psi, mtd.rbias, klaster]
        sum_of_clust = sum_wages_in_custers(new_X, N_clusters)

        # -- SAMPLING --
        # sampled_clust_ndx = sample_clusters(N_clusters, sum_of_clust, pick_n_clusters)
        # show_picked_clust(new_X,km,sampled_clust_ndx)
        ## sampling inside clusters
        sampled_X, indexes = sample_inside_clusters(new_X, N_clusters, sum_of_clust, pick_n_clusters, pick_n_samples)  # 7 clusters, 1500 samples

        # -- HISTOGRAMS --
        plot_hist_2D(new_X)
        plot_hist_2D(sampled_X)
        overlay_plots(new_X, sampled_X)

        #####
        # calc_KL_for_many_sets(X)

if __name__ == "__main__":
    main(sys.argv[1:])
