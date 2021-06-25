# -*- coding: utf-8 -*-
"""
Utilities for ROSES 2021
Adapted from (https://github.com/bgoutorbe/seismic-noise-tomography)
"""

import numpy as np
import itertools as it
from numba import jit

from . import psutils
from .pstomo import Grid

def make_G(paths, grid, v0):
    """
    Makes the matrix G for a given set of paths:
    """

    G = np.zeros((len(paths), grid.n_nodes()))
    for ipath, path in enumerate(paths):
        lon_M, lat_M = path[:, 0], path[:, 1]
        xyzM = psutils.geo2cartesian(lon_M, lat_M)

        iA, iB, iC = grid.indexes_delaunay_triangle(lon_M, lat_M)
        lonlatA, lonlatB, lonlatC = [grid.xy(index_) for index_ in (iA, iB, iC)]
        xyzA, xyzB, xyzC = [psutils.geo2cartesian(lon, lat)
                            for lon, lat in (lonlatA, lonlatB, lonlatC)]

        xyzMp = psutils.projection(xyzM, xyzA, xyzB, xyzC)
        wA, wB, wC = psutils.barycentric_coords(xyzMp, xyzA, xyzB, xyzC)

        # attributing weights to grid nodes along path:
        # w[j, :] = w_j(r) = weights of node j along path
        nM = path.shape[0]
        w = np.zeros((grid.n_nodes(), nM))
        w[iA, range(nM)] = wA
        w[iB, range(nM)] = wB
        w[iC, range(nM)] = wC

        # ds = array of infinitesimal distances along path
        ds = psutils.dist(lons1=lon_M[:-1], lats1=lat_M[:-1],
                          lons2=lon_M[1:], lats2=lat_M[1:])

        # integrating w_j(r) / v0 along path using trapeze formula
        G[ipath, :] = np.sum(0.5 * (w[:, :-1] + w[:, 1:]) / v0 * ds, axis=-1)

    G = np.matrix(G)

    return G

def make_G_slow(paths, grid, v0):
    """
    Makes the matrix G for a given set of paths:
    """

    G = np.zeros((len(paths), grid.n_nodes()))
    for ipath, path in enumerate(paths):
        lon_M, lat_M = path[:, 0], path[:, 1]
        xyzM = psutils.geo2cartesian(lon_M, lat_M)

        iA, iB, iC = grid.indexes_delaunay_triangle(lon_M, lat_M)
        lonlatA, lonlatB, lonlatC = [grid.xy(index_) for index_ in (iA, iB, iC)]
        xyzA, xyzB, xyzC = [psutils.geo2cartesian(lon, lat)
                            for lon, lat in (lonlatA, lonlatB, lonlatC)]

        xyzMp = psutils.projection(xyzM, xyzA, xyzB, xyzC)
        wA, wB, wC = psutils.barycentric_coords(xyzMp, xyzA, xyzB, xyzC)

        # attributing weights to grid nodes along path:
        # w[j, :] = w_j(r) = weights of node j along path
        nM = path.shape[0]
        w = np.zeros((grid.n_nodes(), nM))
        w[iA, range(nM)] = wA
        w[iB, range(nM)] = wB
        w[iC, range(nM)] = wC

        # ds = array of infinitesimal distances along path
        ds = psutils.dist(lons1=lon_M[:-1], lats1=lat_M[:-1],
                          lons2=lon_M[1:], lats2=lat_M[1:])

        # integrating w_j(r) / v0 along path using trapeze formula
        G[ipath, :] = np.sum(0.5 * (w[:, :-1] + w[:, 1:]) / v0 * ds, axis=-1)

    G = np.matrix(G)

    return G


def checkerboard_func(grid, vmid, vmin, vmax, squaresize, shape='cos'):
    """
    Returns a checkerboard function, f(lons, lats), whose background
    value is *vmid*, and alternating min/max values are *vmin* and
    *vmax*. The centers of the anomalies are separated by *squaresize*
    (in km), and their shape is either 'gaussian' or 'cos'.

    @rtype: function
    """
    # converting square size from km to degrees
    d2rad = np.pi / 180.0
    midlat = 0.5 * (grid.ymin + grid.get_ymax())
    latwidth = squaresize / 6371.0 / d2rad
    lonwidth = squaresize / (6371.0 * np.cos(midlat * d2rad)) / d2rad

    # Basis function defining an anomaly of
    # unit height centered at (*lon0*, *lat0*).
    if shape.lower().strip() == 'gaussian':
        def basis_func(lons, lats, lon0, lat0):
            """
            Gausian anomaly , with sigma-parameter such that 3 sigma
            is the distance between the center and the border of
            the square, that is, half the distance between 2
            centers.
            """
            n = len(lons)
            r = psutils.dist(lons1=lons, lats1=lats, lons2=n*[lon0], lats2=n*[lat0])
            sigma = squaresize / 6.0
            return np.exp(- r**2 / (2 * sigma**2))
    elif shape.lower().strip() == 'cos':
        def basis_func(lons, lats, lon0, lat0):
            """
            Cosinus anomaly
            """
            x = (lons - lon0) / lonwidth
            y = (lats - lat0) / latwidth
            outside_square = (np.abs(x) >= 0.5) | (np.abs(y) >= 0.5)
            return np.where(outside_square, 0.0, np.cos(np.pi*x) * np.cos(np.pi*y))
    else:
        raise Exception("Unknown shape anomaly: " + shape)

    # coordinates of the center of the anomalies
    startlon = grid.xmin + lonwidth / 2.0
    stoplon = grid.get_xmax() + lonwidth
    centerlons = list(np.arange(startlon, stoplon, lonwidth))
    startlat = grid.ymin + latwidth / 2.0
    stoplat = grid.get_ymax() + latwidth
    centerlats = list(np.arange(startlat, stoplat, latwidth))
    centerlonlats = list(it.product(centerlons, centerlats))

    # factors by which multiply the basis function associated
    # with each center (to alternate lows and highs)
    polarities = [(centerlons.index(lon) + centerlats.index(lat)) % 2
                  for lon, lat in centerlonlats]
    factors = np.where(np.array(polarities) == 1, vmax - vmid, vmin - vmid)

    def func(lons, lats):
        """
        Checkboard function: sum of the basis functions along
        the centers defined above, times the high/low factor,
        plus background velocity.
        """
        lowhighs = [f * basis_func(lons, lats, lon0, lat0) for f, (lon0, lat0)
                    in zip(factors, centerlonlats)]
        return vmid + sum(lowhighs)

    return func


def path_density(grid, paths, window):
    """
    Returns the path density, that is, on each node of the
    grid, the number of paths that cross the rectangular
    cell of size (window[0], window[1]) centered on
    the node.
    """
    # initializing path density
    density = np.zeros(grid.n_nodes())

    # coordinates of grid nodes and associated windows
    lons_nodes, lats_nodes = grid.xy_nodes()
    lons_min = np.expand_dims(lons_nodes - window[0] / 2.0, axis=-1)
    lons_max = np.expand_dims(lons_nodes + window[0] / 2.0, axis=-1)
    lats_min = np.expand_dims(lats_nodes - window[1] / 2.0, axis=-1)
    lats_max = np.expand_dims(lats_nodes + window[1] / 2.0, axis=-1)

    for path in paths:
        lons_path, lats_path = path[:, 0], path[:, 1]
        # are points of paths in windows?
        # 1st dim = grid nodes; 2nd dim = points along path
        points_in_windows = (lons_path >= lons_min) & (lons_path <= lons_max) & \
                            (lats_path >= lats_min) & (lats_path <= lats_max)
        density += np.any(points_in_windows, axis=-1)

    return density


def invert4model(alpha, beta, lambda_, correlation_length, lonstep, latstep,
                 tol, disp_curves):
    """
    A function to wrap all the calculations when inverting for the tomographic
    model.
    """

    # Setup the inversion grid
    # ----------------------------------------------------------------------- #

    # Get the smallest longitude
    min_rcv_lon = np.min(disp_curves["receiver_lon"])
    min_src_lon = np.min(disp_curves["source_lon"])
    min_lon = np.min([min_rcv_lon, min_src_lon]) - tol

    # Get the smallest latitude
    min_rcv_lat = np.min(disp_curves["receiver_lat"])
    min_src_lat = np.min(disp_curves["source_lat"])
    min_lat = np.min([min_rcv_lat, min_src_lat]) - tol

    # Get the largest longitude
    max_rcv_lon = np.max(disp_curves["receiver_lon"])
    max_src_lon = np.max(disp_curves["source_lon"])
    max_lon = np.max([max_rcv_lon, max_src_lon])

    # Get the largest latitude
    max_rcv_lat = np.max(disp_curves["receiver_lat"])
    max_src_lat = np.max(disp_curves["source_lat"])
    max_lat = np.max([max_rcv_lat, max_src_lat])

    nlon = np.ceil((max_lon + tol - min_lon) / lonstep)
    nlat = np.ceil((max_lat + tol - min_lat) / latstep)

    # Create a grid object, from pysismo
    grid = Grid(min_lon, lonstep, nlon, min_lat, latstep, nlat)
    # ----------------------------------------------------------------------- #

    # Here I am just appending to a list. This is not necessarily fast, but
    # I'm only appending a few values so it's ok here. Usually it's better to
    # predefine the array.
    paths = []
    dists = []

    for idx, row in disp_curves.iterrows():
        dist = psutils.dist(row.source_lon, row.source_lat,
                            row.receiver_lon, row.receiver_lat)
        npts = np.max([np.ceil(dist) + 1, 100])
        source_coords = (row.source_lon, row.source_lat)
        receiver_coords = (row.receiver_lon, row.receiver_lat)
        path = psutils.geodesic(source_coords, receiver_coords, npts)

        paths.append(path)
        dists.append(dist)
    dists = np.array(dists)
    paths = np.array(paths, dtype="object")
    vels = disp_curves["group_velocity"]
    s = (dists / vels).sum() / dists.sum()
    v0 = 1.0 / s
    G = make_G(paths, grid, v0)
    dobs = np.matrix(dists / vels - dists / v0).T
    density = path_density(grid, paths, window=(lonstep, latstep))
    sigmav = np.ones((len(vels))) * 0.3
    sigmad = sigmav * dists / vels**2
    Cinv = np.matrix(np.zeros((len(sigmav), len(sigmav))))
    np.fill_diagonal(Cinv, 1.0 / sigmad**2)
    d = grid.to_2D_array(density)

    dists_mat = np.zeros((grid.n_nodes(), grid.n_nodes()))
    i_upper, j_upper = np.triu_indices_from(dists_mat)
    lons_i, lats_i = grid.xy(i_upper)
    lons_j, lats_j = grid.xy(j_upper)

    dists_mat[i_upper, j_upper] = psutils.dist(lons1=lons_i, lats1=lats_i,
                                               lons2=lons_j, lats2=lats_j)
    dists_mat += dists_mat.T

    # Calculate the smoothing kernel
    S = np.exp(- dists_mat**2 / (2 * correlation_length**2))
    S /= S.sum(axis=-1) - np.diag(S)  # normalization of non-diagonal terms

    # setting up spatial regularization matrix F
    F = np.matrix(-S)
    F[np.diag_indices_from(F)] = 1
    F *= alpha

    # Calculate regularization matrix Q
    # ----------------------------------------------------------------------- #
    Q = F.T * F

    for i, pd in enumerate(density):
        Q[i, i] += beta ** 2 * np.exp(-2 * lambda_ * pd)
    # ----------------------------------------------------------------------- #
    covmopt = np.linalg.inv(G.T * Cinv * G + Q)
    Ginv = covmopt * G.T  # Calculate the generalized inverse
    mopt = Ginv * Cinv * dobs
    R = Ginv * Cinv * G
    v = grid.to_2D_array(v0 / (1 + mopt))

    return v, d, R, grid
