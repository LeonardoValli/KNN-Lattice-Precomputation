from statistics import mode, mean as avg
from random import shuffle
from time import perf_counter
import numpy as np
from scipy.stats import norm
from annoy import AnnoyIndex
import pickle
from itertools import product

# CL, FILE = 29, "full-normalized.csv"
# CL, FILE = 20, "all-normalized.csv"
CL, FILE = 12, "all-cfs_subset.csv"


with open(FILE, "r") as f:
    f.readline()  # Discard header line
    DATA = [line.strip().split(",") for line in f]
    DATA = [tuple(list(map(float, d[:CL])) + [d[CL]]) for d in DATA]
    CLASS_VALUES = set(d[CL] for d in DATA)

def generate_train_test(data=DATA):
    global TRAIN, TEST
    shuffle(data)
    TRAIN_PERCENT = 0.7
    TRAIN = DATA[:int(len(data) * TRAIN_PERCENT)]
    TEST = DATA[int(len(data) * TRAIN_PERCENT):]

generate_train_test()

############################################
K = 3
NUM_TESTS = 100
LATTICE_PTS_PER_DIM = 3

ATTRIBUTE_LISTS = [list(d[i] for d in DATA) for i in range(CL)]
ATTRIBUTE_SETS = [set(ATTRIBUTE_LISTS[i]) for i in range(CL)]
CATEGORICAL_ATTRIBUTES = [i for i in range(CL) if len(ATTRIBUTE_SETS[i]) < 10]
NUM_POINTS = 1_000_000  # 10_000_000
NUM_TREES = 10
SEARCH_K = 15

def generate_lattice():
    points = product(LATTICE_Z_VALUES, repeat=CL)
    print(f"Generating {LATTICE_PTS_PER_DIM ** CL} lattice points")
    lattice = {tuple(LATTICE_Z_TO_CDF[p] for p in point): classify(point) for point in points}
    with open(f"{FILE.split('.')[0]}-lattice-{LATTICE_PTS_PER_DIM}.pkl", "wb") as f:
        pickle.dump(lattice, f)
    return lattice

zscore_cache = {}
def zscore_to_lattice_cdf(z):
    z = round(z, 2)
    if z not in zscore_cache:
        zscore_cache[z] = max(min(round(norm.cdf(z) / (10 / (LATTICE_PTS_PER_DIM + 1)), 1), LATTICE_CDF_MAX), LATTICE_CDF_MIN)
    return zscore_cache[z]

def lattice_cdf_to_zscore(cdf):
    return norm.ppf(cdf * (10 / (LATTICE_PTS_PER_DIM + 1)))

LATTICE_Z_VALUES = [lattice_cdf_to_zscore(x / 10) for x in range(1, LATTICE_PTS_PER_DIM + 1)]
LATTICE_Z_TO_CDF = {lattice_cdf_to_zscore(x / 10): x / 10 for x in range(1, LATTICE_PTS_PER_DIM + 1)}
LATTICE_CDF_MIN = 1 / 10
LATTICE_CDF_MAX = LATTICE_PTS_PER_DIM / 10

# for i in range(-3, 4):
#     print(i, zscore_to_lattice_cdf(i), lattice_cdf_to_zscore(zscore_to_lattice_cdf(i)))


# def generate_lattice():
#     mean = tuple(avg(ATTRIBUTE_LISTS[i]) for i in range(CL))
#     cov = [[covariance(ATTRIBUTE_LISTS[i], ATTRIBUTE_LISTS[j]) for j in range(CL)] for i in range(CL)]
#     points = np.random.multivariate_normal(mean, cov, NUM_POINTS)
#     lattice = tuple((point, classify_tree(point)) for point in points)
#     with open(f"{FILE.split('.')[0]}-lattice.pkl", "wb") as f:
#         pickle.dump(lattice, f)
#     return lattice

def load_lattice():
    with open(f"{FILE.split('.')[0]}-lattice-{LATTICE_PTS_PER_DIM}.pkl", "rb") as f:
        return pickle.load(f)

def build_knn_tree():
    t = AnnoyIndex(CL, 'euclidean')
    for i, d in enumerate(TRAIN):
        t.add_item(i, d[:CL])
    t.build(NUM_TREES, n_jobs=-1)
    t.save(f"{FILE.split('.')[0]}.ann")
    return t

def build_lattice_tree():
    t = AnnoyIndex(CL, 'euclidean')
    for i, (point, _) in enumerate(LATTICE):
        t.add_item(i, point)
    t.build(NUM_TREES, n_jobs=-1)
    t.save(f"{FILE.split('.')[0]}-lattice.ann")
    return t

def load_knn_tree():
    t = AnnoyIndex(CL, 'euclidean')
    t.load(f"{FILE.split('.')[0]}.ann")
    return t

def distance(d1, d2):  # squared distance
    return sum((p1 - p2) ** 2 for p1, p2 in zip(d1[:CL], d2[:CL]))

def classify(inst):
    return mode(d[CL] for d in sorted(TRAIN, key=lambda d: distance(inst, d))[:K])

def classify_tree(inst):
    return mode(TRAIN[i][CL] for i in TREE.get_nns_by_vector(inst[:CL], K, search_k=SEARCH_K))

def classify_lattice(inst):
    return LATTICE[tuple(zscore_to_lattice_cdf(inst[i]) for i in range(CL))]

def classify_weighted_distance(inst):
    distances = ((distance(inst, d), d[CL]) for d in TRAIN)
    distances = sorted(distances)[:K]
    classifications = ((sum(1 / ((d[0] + 2)**3) for d in distances if d[1] == c), c) for c in set(d[1] for d in distances))
    return max(classifications)[1]

def test():
    return sum(1 for d in TEST if classify(d) == d[CL]) / len(TEST)

def confusion_matrix(classifier, model=None):
    matrix = [[0] * len(ATTR_TO_VALS_CL[CL]) for _ in range(len(ATTR_TO_VALS_CL[CL]))]
    if model:
        for inst in TEST:
            matrix[int(inst[CL])][int(classifier(inst[:CL], model))] += 1
    else:
        y_pred = [classifier(inst[:CL]) for inst in X_test]
        for i in range(len(y_test)):
            matrix[int(y_test[i] == "True")][int(y_pred[i] == "True")] += 1

    print("\t", end="")
    for i in range(len(ATTR_TO_VALS_CL[CL])):
        print(f"{i}\t", end="")
    print()
    for i in range(len(ATTR_TO_VALS_CL[CL])):
        print(f"{i}\t", end="")
        for j in range(len(ATTR_TO_VALS_CL[CL])):
            print(f"{matrix[i][j]}\t", end="")
        print()
    print()
    return matrix

def iterative_test():
    global TREE
    accuracies = []
    for _ in range(NUM_TESTS):
        accuracies.append(test())
        generate_train_test()
        # TREE = build_knn_tree()
    print(f"Accuracy: {avg(accuracies)}")

classify = classify_lattice
NUM_TESTS = 100
start = perf_counter()
TREE = build_knn_tree()
print(f"Built KNN tree in {perf_counter() - start:.4f}s")

start = perf_counter()
LATTICE = generate_lattice()
# LATTICE = load_lattice()
print(f"Generated lattice in {perf_counter() - start:.4f}s")

# start = perf_counter()
# print(f"{K=}, Accuracy: {test()}")
# print(f"Time: {perf_counter() - start:.4f}s")

# start = perf_counter()
# iterative_test()
# print(f"Time: {perf_counter() - start:.4f}s")

ATTR_TO_VALS_CL = {i: set(inst[i] for inst in TRAIN) for i in range(CL + 1)}
X_test = [inst[:CL] for inst in TEST]
y_test = [inst[CL] for inst in TEST]
confusion_matrix(classify)


"""
RESULTS

Lattice 3-per-dim build time: 5.2s
Lattice 4-per-dim build time: 220s

1000 tests:
Normal: 1950s
Proximity: 17.5s
Lattice: 5.5s

10000 tests:
Normal: 19500s, 0.808
Proximity: 171s, 0.799
Lattice: 51s, 3 splits: 0.777, 4 splits: 0.796

Confusion matrices:
Default KNN
        0       1
0       254     77
1       59      260

Proximity
        0       1
0       268     75
1       58      249

Lattice
        0       1
0       253     90
1       44      263
"""