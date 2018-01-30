import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import argparse


DEBUG = False


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    return parser


def fetch_data():
    data = pd.read_csv('Features_Variant_1.csv', header=None)
    # data = pd.read_csv('test.csv', header=None)

    # Create target vector and data matrix
    y = data.iloc[:, -1].values

    # Drop the 38th column
    data = data.drop(37, axis=1)

    # Add X0 = 1 for all rows
    data.insert(0, 'X0', 1)
    X = data.iloc[:, :-1].values

    if DEBUG:
        print("X Shape: ", X.shape)
        print("Y Shape: ", y.shape)

    return X, y


def SSE(X, w, y):
    """
    Sum of Squared Errors calculation
    :param X: Data
    :param w: Weights
    :param y: Labels
    :return: float
    """
    err = np.matmul(X, w) - y
    sq_err = err * err

    # if DEBUG:
    #     print("Error: ", err)
    #     print("Square error:", sq_err)
    #     print("SSE:", sum(sq_err))
    #     print("Max:", max(sq_err))

    return sum(sq_err)


def calcW(X, y):
    # w = inv(X.T * X) * X.T * y
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    # return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def plot_performance(sseA, sseB, sseC, sseAs, sseBs, sseCs):
    fig, ax = plt.subplots()
    ax.scatter(range(3), [sseA, sseB, sseC])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_title('SSE')

    plt.savefig('sse.png')

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(range(3), [sseAs, sseBs, sseCs])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_title('Normalized Data SSE')

    plt.savefig('normalized_sse.png')

    plt.close(fig)


def plot_r2(r2A, r2B, r2C, r2As, r2Bs, r2Cs):
    fig, ax = plt.subplots()
    ax.scatter(range(3), [r2A, r2B, r2C])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_title('R2 Scores')

    plt.savefig('r2.png')

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(range(3), [r2As, r2Bs, r2Cs])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_title('Normalized Data R2 Scores')

    plt.savefig('normalized_r2.png')

    plt.close(fig)


def main():
    X, y = fetch_data()

    # Test Train Splits:
    # A = 80 train 20 test
    # B = 50 train 50 test
    # C = 20 train 80 test
    XtrainA, XtestA, YtrainA, YtestA = train_test_split(X, y, test_size=0.2, random_state=123456)
    XtrainB, XtestB, YtrainB, YtestB = train_test_split(X, y, test_size=0.5, random_state=123456)
    XtrainC, XtestC, YtrainC, YtestC = train_test_split(X, y, test_size=0.8, random_state=123456)

    if DEBUG:
        print("X train/test As Shapes:", XtrainA.shape, XtestA.shape)
        print("X train/test Bs Shapes:", XtrainB.shape, XtestB.shape)
        print("X train/test Cs Shapes:", XtrainC.shape, XtestC.shape)

    wA = calcW(XtrainA, YtrainA)
    wB = calcW(XtrainB, YtrainB)
    wC = calcW(XtrainC, YtrainC)

    sseA = SSE(XtestA, wA, YtestA)
    sseB = SSE(XtestB, wB, YtestB)
    sseC = SSE(XtestC, wC, YtestC)

    print("SSE:", sseA, sseB, sseC)

    # Feature scaling
    sc = StandardScaler()

    sc.fit(XtrainA)
    XtrainAs = sc.transform(XtrainA)
    XtestAs = sc.transform(XtestA)

    sc.fit(XtrainB)
    XtrainBs = sc.transform(XtrainB)
    XtestBs = sc.transform(XtestB)

    sc.fit(XtrainC)
    XtrainCs = sc.transform(XtrainC)
    XtestCs = sc.transform(XtestC)

    # print(XtrainA[:, 1], XtrainAs[:, 1])

    # Reset X0 to 1
    XtrainAs[:, 0] = [1] * XtrainAs.shape[0]
    XtestAs[:, 0] = [1] * XtestAs.shape[0]
    XtrainBs[:, 0] = [1] * XtrainBs.shape[0]
    XtestBs[:, 0] = [1] * XtestBs.shape[0]
    XtrainCs[:, 0] = [1] * XtrainCs.shape[0]
    XtestCs[:, 0] = [1] * XtestCs.shape[0]

    if DEBUG:
        print("X train/test As Shapes:", XtrainAs.shape, XtestAs.shape)
        print("X train/test Bs Shapes:", XtrainBs.shape, XtestBs.shape)
        print("X train/test Cs Shapes:", XtrainCs.shape, XtestCs.shape)

    wAs = calcW(XtrainAs, YtrainA)
    wBs = calcW(XtrainBs, YtrainB)
    wCs = calcW(XtrainCs, YtrainC)

    # if DEBUG:
    #     print("W1s:", wAs)

    sseAs = SSE(XtestAs, wAs, YtestA)
    sseBs = SSE(XtestBs, wBs, YtestB)
    sseCs = SSE(XtestCs, wCs, YtestC)

    print("Scaled SSE:", sseAs, sseBs, sseCs)

    varA = np.var(YtestA)
    varB = np.var(YtestB)
    varC = np.var(YtestC)

    R2A = 1 - sseA / varA
    R2B = 1 - sseB / varB
    R2C = 1 - sseC / varC
    R2As = 1 - sseAs / varA
    R2Bs = 1 - sseBs / varB
    R2Cs = 1 - sseCs / varC

    print("R2 Scores:", R2A, R2As, R2B, R2Bs, R2C, R2Cs)

    plot_performance(sseA, sseB, sseC, sseAs, sseBs, sseCs)
    plot_r2(R2A, R2B, R2C, R2As, R2Bs, R2Cs)


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    if args.debug:
        DEBUG = True

    main()
