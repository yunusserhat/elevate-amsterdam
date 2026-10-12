# -*- coding: utf-8 -*-
"""Shared pieces for the synthetic WP2 data (steps 3.3 and 3.4): routing graphs and exposure.

Routing uses the full OpenStreetMap graph of step 1.2 (network_type "all", read from the
OSMnx cache), which keeps the separately mapped sidewalks and crossings that the WP3
segments leave out; without them the walking network falls apart into pieces. The City's
ferries are added as walk and bicycle edges. Exposure uses
the WP3 street segments: the mean indicator of the observed segments within 25 m of a point
(the protocol's buffered street-segment assignment).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

PROJ = Path(__file__).resolve().parents[1]
METRIC = "EPSG:28992"
NO_WALK = {"motorway", "motorway_link", "trunk", "trunk_link", "busway", "construction", "proposed", "raceway"}
NO_BIKE = NO_WALK | {"steps", "elevator", "corridor"}
DRIVE = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
         "unclassified", "living_street", "service", "road"}
YES = {"yes", "designated", "permissive"}


def _flat(v):
    if isinstance(v, (list, tuple)):
        return "|".join(sorted(str(x) for x in v))
    return v if isinstance(v, str) else ""


def osm_graph_tables():
    """Nodes (osmid -> x, y in EPSG:28992) and edges with tags and metric geometry."""
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(PROJ / "data/raw/osmnx_cache")
    ox.settings.requests_timeout = 600
    ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + [
        "footway", "service", "access", "foot", "bicycle", "cycleway", "sidewalk", "surface", "busway"]
    gemeente = gpd.read_parquet(PROJ / "data/derived/gemeente.parquet")
    boundary = gemeente.to_crs(METRIC).geometry.union_all()
    poly = gpd.GeoSeries([boundary.buffer(500)], crs=METRIC).to_crs(4326).iloc[0]
    graph = ox.graph_from_polygon(poly, network_type="all", simplify=True, retain_all=True)
    nodes, edges = ox.graph_to_gdfs(graph)
    nodes = nodes.to_crs(METRIC)
    edges = edges.reset_index().to_crs(METRIC)
    for c in ["highway", "footway", "foot", "bicycle", "access", "service"]:
        edges[c] = edges[c].map(_flat) if c in edges.columns else ""
    edges = pd.concat([edges, ferry_edges(poly, nodes)], ignore_index=True)
    return nodes, edges


def ferry_edges(poly, nodes) -> gpd.GeoDataFrame:
    """The City's free ferries (OSM route=ferry, Pampus excluded) as walk and bicycle edges.

    Each ferry way joins the nearest graph nodes (within 150 m) at its two ends. Its length gets
    150 m added for boarding and waiting. Without the ferries, trips between Amsterdam-Noord and
    the centre detoured over distant bridges.
    """
    f = ox.features_from_polygon(poly, {"route": "ferry"}).reset_index()
    f = f[f.geometry.geom_type.isin(["LineString", "MultiLineString"])].to_crs(METRIC)
    f = f[~f.get("name", pd.Series(index=f.index, dtype=object)).fillna("").str.contains("Pampus")]
    tree = cKDTree(np.c_[nodes.geometry.x, nodes.geometry.y])
    rows = []
    for g in f.geometry:
        if g.geom_type == "MultiLineString":
            g = shapely.line_merge(g)
            if g.geom_type != "LineString":
                g = max(g.geoms, key=lambda x: x.length)
        (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
        d0, i0 = tree.query([x0, y0])
        d1, i1 = tree.query([x1, y1])
        if d0 > 150 or d1 > 150 or i0 == i1:
            continue
        u, v = nodes.index[i0], nodes.index[i1]
        line = shapely.LineString([nodes.geometry.iloc[i0].coords[0], *g.coords, nodes.geometry.iloc[i1].coords[0]])
        rows.append({"u": u, "v": v, "key": 0, "highway": "ferry", "footway": "", "foot": "yes", "bicycle": "yes",
                     "access": "", "service": "", "length": float(line.length) + 150.0, "geometry": line})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC)


def mode_masks(edges) -> dict:
    hw = edges.highway.str.split("|")
    foot, bike, access, fw, svc = edges.foot, edges.bicycle, edges.access, edges.footway, edges.service
    private = access.str.contains("private|no", regex=True)
    only = lambda allowed: hw.map(lambda parts: set(parts) <= allowed)  # noqa: E731
    any_in = lambda bad: hw.map(lambda parts: bool(set(parts) & bad))  # noqa: E731
    walk = ~any_in(NO_WALK) & ~(only({"cycleway"}) & ~foot.isin(YES)) & ~(private & ~foot.isin(YES))
    cyc = (~any_in(NO_BIKE) & ~(only({"footway", "pedestrian"}) & ~bike.isin(YES)) & ~fw.isin(["sidewalk", "crossing"])
           & ~(private & ~bike.isin(YES)))
    drive = any_in(DRIVE) & ~private & ~svc.str.contains("parking_aisle|driveway", regex=True)
    return {"walk": walk.to_numpy(), "bike": cyc.to_numpy(), "drive": drive.to_numpy()}


class ModeGraph:
    """Undirected graph of the edges a mode may use; routes on its largest component."""

    def __init__(self, nodes, edges, mask):
        self.ids = nodes.index.to_numpy()
        self.xy = np.c_[nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()]
        pos = {o: i for i, o in enumerate(self.ids)}
        e = edges[mask]
        u = e.u.map(pos).to_numpy()
        v = e.v.map(pos).to_numpy()
        w = e.length.to_numpy(float)
        eid = e.index.to_numpy()
        rows, cols = np.r_[u, v], np.r_[v, u]
        data, ids, rev = np.r_[w, w], np.r_[eid, eid], np.r_[np.zeros(len(u), bool), np.ones(len(u), bool)]
        order = np.lexsort((data, cols, rows))  # keep the shortest of parallel edges
        rows, cols, data, ids, rev = rows[order], cols[order], data[order], ids[order], rev[order]
        first = np.r_[True, (np.diff(rows) != 0) | (np.diff(cols) != 0)]
        rows, cols, data, ids, rev = rows[first], cols[first], data[first], ids[first], rev[first]
        n = len(self.ids)
        self.csr = csr_matrix((np.maximum(data, 0.01), (rows, cols)), shape=(n, n))
        self.base = self.csr.data.copy()
        key = rows.astype(np.int64) * n + cols
        self._edge = dict(zip(key.tolist(), zip(ids.tolist(), rev.tolist())))
        self.n = n
        _, labels = connected_components(self.csr, directed=False)
        used = np.zeros(n, bool)
        used[rows] = True
        main = np.bincount(labels[used]).argmax()
        self.main = np.flatnonzero((labels == main) & used)
        self.tree = cKDTree(self.xy[self.main])
        self.geoms = {}  # edge id -> coordinates (filled lazily)
        self.edges = edges

    def nearest(self, xy) -> np.ndarray:
        _, i = self.tree.query(np.atleast_2d(xy))
        return self.main[i]

    def _coords(self, eid):
        c = self.geoms.get(eid)
        if c is None:
            c = np.asarray(self.edges.geometry.loc[eid].coords)[:, :2]
            self.geoms[eid] = c
        return c

    def route(self, src: int, dst: int, rng) -> np.ndarray:
        """Coordinates (EPSG:28992) of a shortest path with edge lengths scaled by U(1.0, 1.3)."""
        if src == dst:
            return self.xy[[src, src]]
        g = self.csr.copy()
        g.data = self.base * rng.uniform(1.0, 1.3, size=len(self.base))
        straight = float(np.hypot(*(self.xy[src] - self.xy[dst])))
        for limit in (2.5 * straight + 3000.0, np.inf):
            dist, pred = dijkstra(g, indices=src, return_predecessors=True, limit=limit)
            if np.isfinite(dist[dst]):
                break
        path = [int(dst)]
        while path[-1] != src:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        parts = []
        for a, b in zip(path[:-1], path[1:]):
            eid, _ = self._edge[int(a) * self.n + int(b)]
            c = self._coords(eid)
            if np.hypot(*(c[0] - self.xy[a])) > np.hypot(*(c[-1] - self.xy[a])):
                c = c[::-1]  # traversed against the stored direction
            parts.append(c if not parts else c[1:])
        return np.vstack(parts)


class Exposure:
    """Mean of a segment indicator over observed WP3 segments within `buffer_m` of each point."""

    def __init__(self, column: str = "vegetation_pixel_fraction__mean", buffer_m: float = 25.0):
        seg = gpd.read_parquet(PROJ / "outputs/wp3/indicators_segment.parquet", columns=["segment_id", "n_images", column, "geometry"])
        seg = seg[(seg.n_images > 0) & seg[column].notna()].to_crs(METRIC)
        self.column, self.buffer_m = column, buffer_m
        self.values = seg[column].to_numpy(float)
        self.tree = shapely.STRtree(seg.geometry.to_numpy())

    def at(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        ok = np.isfinite(xy).all(axis=1)
        if not ok.any():
            return out
        pts = shapely.points(xy[ok])
        ip, iseg = self.tree.query(pts, predicate="dwithin", distance=self.buffer_m)
        s = np.bincount(ip, weights=self.values[iseg], minlength=len(pts))
        c = np.bincount(ip, minlength=len(pts))
        val = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        out[np.flatnonzero(ok)] = val
        return out
