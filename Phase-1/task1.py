import numpy as np
from scipy.stats import norm


def normalize(df, lower=-1, upper=1):
    df = df.T
    df = (df - df.min()) * (upper - lower) / (df.max() - df.min()) + lower
    return df.T


def get_lengths(r, mean=0, std=0.25):
    x = np.linspace(-1, 1, 200)
    gaussian = norm(mean, std).pdf(x)
    band_points = np.concatenate((np.linspace(-1, 0, r+1), np.linspace(0, 1, r+1)[1:]))
    lengths = []
    for i in range(len(band_points)-1):
        nx = x[(x >= band_points[i]) & (x < band_points[i+1])]
        ny = norm(mean, std).pdf(nx)
        lengths.append(ny.sum()/gaussian.sum())
    return lengths


def quantize(data, lengths):
    lengths = np.array(lengths) * 2
    bins = [-1]
    cv = -1
    for k in range(len(lengths) - 1):
        cv = cv + lengths[k]
        bins.append(cv)
    bins.append(1)
    for i in range(len(data)):
        ft = data.iloc[i, :]
        ft1 = np.digitize(ft, bins)
        data.iloc[i, :] = ft1
    return data


def windowing(df, w, s, f):
    all_feats = []
    for i in range(df.shape[0]):
        feat_windowed = []
        for j in range(0, df.shape[1], s):
            idx = (f, s, j)
            win = df.values[i, j:j+w].astype(int).tolist()
            feat_windowed.append((idx, win))
        all_feats.append(feat_windowed)
    return all_feats
