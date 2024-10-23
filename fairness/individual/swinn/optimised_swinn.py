from __future__ import annotations

import collections
import heapq
import math
import time
import typing
from collections import deque
from enum import Enum
from itertools import groupby
from multiprocessing import Pool
from pstats import SortKey

import numpy as np
import scipy.spatial.distance
from river import neighbors
from river.neighbors.ann.nn_vertex import Vertex
from river.neighbors.base import FunctionWrapper, DistanceFunc

from fairness.individual.swinn.vertex import OptimisedVertex
from scenario import CombinedState


class OptimisedSWINN(neighbors.SWINN):
    """Optimised version of SWINN

    TODO: ASSUMPTION: for some methods, the item requesting a search query is in the graph itself:
        retrieve its nearest neighbours. For items that are explicitly mentioned to (possibly) NOT be in the graph,
        the default implementations are used.
    """

    def __init__(self, graph_k: int = 20,
                 dist_func: DistanceFunc | FunctionWrapper | None = None,
                 maxlen: int = 1000,
                 warm_up: int = 500,
                 max_candidates: int = None,
                 delta: float = 0.0001,
                 prune_prob: float = 0.0,
                 n_iters: int = 10,
                 seed: int = None, ):
        # Super call
        super(OptimisedSWINN, self).__init__(graph_k, dist_func, maxlen, warm_up, max_candidates,
                                             delta, prune_prob, n_iters, seed)
        # Store heap of sorted neighbours for linear scan during warm-up period
        self.individual_heaps = {}
        self.individual_comparisons = {}
        self._t = 0
        self._len_data = 0
        self._uuid = 0
        self._min_uuid = 0

        self.e_calls = 0
        self.re_calls = 0

    def _init_graph(self):
        """Create a nearest neighbor graph from stored info. Original creates a random first.
        This starts the graph from already computed distances/neighbours and continues the
        standard refinement process."""
        for vertex in self._data:
            # Based on Vertex.fill(...) method
            for (dist, t, n) in self.individual_heaps[vertex.uuid][:self.graph_k]:
                vertex.edges[n] = dist
                vertex.flags.add(n)
                n.r_edges[vertex] = dist
                #
                vertex.ur_edges[n.uuid] = dist
                n.uedges[vertex.uuid] = dist
                #
                vertex.neighbours.add(n)
                n.neighbours.add(vertex)
                #
                n.not_up_to_date()
            vertex.not_up_to_date()
            # Neighbors are ordered by distance
            vertex.worst_edge = n
        self.individual_heaps = {}

    def _fix_graph(self):
        """Connect every isolated node in the graph to their nearest neighbors."""
        # for node in list(OptimisedVertex._isolated):
        for node in OptimisedVertex._isolated:
            if not node.is_isolated():
                continue
            neighbors, dists = self._search(node.item, self.graph_k)
            node.fill(neighbors, dists)

        # Update class property
        OptimisedVertex._isolated.clear()

    def append(self, item: typing.Any, **kwargs):
        node = OptimisedVertex(item, self._uuid)

        if not self._index:
            # Add individual to heaps
            self.individual_heaps[node.uuid] = []
            for i, neighbour in enumerate(self._data):
                # Add this neighbour to the node
                dist = self.dist_func(node.item, neighbour.item)
                heapq.heappush(self.individual_heaps[node.uuid],
                               (dist, self._t + i, neighbour))  # uuid influences tie-breakers
                # Add node to the neighbour
                heapq.heappush(self.individual_heaps[neighbour.uuid],
                               (dist, self._t + i, node))  # uuid influences tie-breakers
                try:
                    self.individual_comparisons[node.uuid][neighbour.uuid] = dist
                except KeyError:
                    self.individual_comparisons[node.uuid] = {neighbour.uuid: dist}
            self._t += 1

            self._data.append(node)
            self._len_data += 1
            self._uuid += 1

            if self._len_data >= self.warm_up:
                self._init_graph()
                self._refine()
                self._index = True
            return

        # A slot will be replaced, so let's update the search graph first
        if self._len_data == self.maxlen:
            self._safe_node_removal()

        # Assign the closest neighbors to the new item
        neighbors, dists = self._search(node.item, self.graph_k)

        # Add the new element to the buffer
        self._data.append(node)
        node.fill(neighbors, dists)
        # The current neighbours have been updated
        node.search_up_to_date = True
        node._search_neighbours = [n.item for n in neighbors]
        node._search_distributions = dists
        node._search_n_actions = [n.item[1] for n in neighbors]
        node._search_mean_n_actions = sum(node._search_n_actions)/len(node._search_n_actions)
        #

        if self._uuid >= self.maxlen:
            self._min_uuid = self._uuid - self.maxlen - 1
        if self._len_data < self.maxlen:
            self._len_data += 1

        self._uuid += 1

    def get_nn_for_all(self, k, epsilon: float = 0.1, return_distances=False, return_as_array=False, representatives=None):
        """Get the nearest neighbours for all individuals in the graph"""
        individuals = self._data if representatives is None else [i for i in self._data if i.uuid in representatives]
        range_individuals = range(self._len_data) if representatives is None else [n for n, i in enumerate(self._data) if i.uuid in representatives]
        if self._len_data < self.warm_up:
            # Linear scan
            if return_distances:
                neighbours_p = [heapq.nsmallest(k, self.individual_heaps[item_id]) for item_id in range_individuals]
                nn = [[p[-1].item for p in np] for np in neighbours_p]
                dists = [[p[0] for p in np] for np in neighbours_p]
                return [(self._data[item_id].item, (nn[item_id], dists[item_id]))
                        for item_id in range_individuals]
            else:
                return [(self._data[item_id].item, [p[-1].item for p in heapq.nsmallest(k, self.individual_heaps[item_id])])
                        for item_id in range_individuals]
        else:
            return [(p.item, self._search(p.item, k, epsilon, seed=p, return_dists=return_distances,
                                          return_as_array=return_as_array)) for p in individuals]

    def get_actions_n_actions(self, k, epsilon: float = 0.1, return_distances=False, return_as_array=False, representatives=None):
        individuals = self._data if representatives is None else [i for i in self._data if i.uuid in representatives]
        range_individuals = range(self._len_data) if representatives is None else [n for n, i in enumerate(self._data)
                                                                                   if i.uuid in representatives]
        #
        actions = [self._data[item_id].item[1] for item_id in range_individuals]
        # Linear scan
        if self._len_data < self.warm_up:
            neighbours_p = [heapq.nsmallest(k, self.individual_heaps[item_id]) for item_id in range_individuals]
            n_actions = np.array([np.mean([n.item[1] for n in nn]) for nn in neighbours_p])
        else:
            neighbours_p = [self._search(p.item, k, epsilon, seed=p, return_dists=False, return_as_array=return_as_array)
                            for p in individuals]
            n_actions = np.array([np.mean([n[1] for n in nn]) for nn in neighbours_p])
            # n_actions = np.array([sum([n[1]/len(nn) for n in nn]) for nn in neighbours_p])
        return actions, n_actions

    def get_actions_n_actions2(self, k, epsilon: float = 0.1, return_distances=False, return_as_array=False, representatives=None):
        individuals = self._data if representatives is None else [i for i in self._data if i.uuid in representatives]
        range_individuals = range(self._len_data) if representatives is None else [n for n, i in enumerate(self._data)
                                                                                   if i.uuid in representatives]
        #
        actions = [self._data[item_id].item[1] for item_id in range_individuals]
        # Linear scan
        if self._len_data < self.warm_up:
            neighbours_p = [heapq.nsmallest(k, self.individual_heaps[item_id]) for item_id in range_individuals]
            # n_actions = np.array([np.mean([n[1] for n in nn]) for nn in neighbours_p])
            n_actions = np.array([sum([n[1] for n in nn])/len(nn) for nn in neighbours_p])
        else:
            neighbours_p = [self._search_actions_only(p.item, k, epsilon, seed=p, return_mean_action=False, return_as_array=True)
                           for p in individuals]
            # n_actions = np.array([sum(nn)/len(nn) for nn in neighbours_p])
            n_actions = np.array(neighbours_p, dtype=object).mean(axis=1)
        return actions, n_actions

    def get_actions_n_actions3(self, k, epsilon: float = 0.1, return_distances=False, return_as_array=False, representatives=None):
        individuals = self._data if representatives is None else [i for i in self._data if i.uuid in representatives]
        range_individuals = range(self._len_data) if representatives is None else [n for n, i in enumerate(self._data)
                                                                                   if i.uuid in representatives]
        #
        actions = [self._data[item_id].item[1] for item_id in range_individuals]
        # actions = np.array(map(self._data[item_id].item[1] for item_id in range_individuals))
        # Linear scan
        if self._len_data < self.warm_up:
            neighbours_p = [heapq.nsmallest(k, self.individual_heaps[item_id]) for item_id in range_individuals]
            # n_actions = np.array([np.mean([n[1] for n in nn]) for nn in neighbours_p])
            n_actions = np.array([sum([n[1] for n in nn])/len(nn) for nn in neighbours_p])
        else:
            # n_actions = [self._search_actions_only(p.item, k, epsilon, seed=p, return_mean_action=True, return_as_array=True)
            #     for p in individuals]
            # n_actions = np.array([sum(nn) / len(nn) for nn in neighbours_p])
            # n_actions = np.array(neighbours_p, dtype=object).mean(axis=1)
            #
            neighbours_p = [self._search_actions_only(p.item, k, epsilon, seed=p, return_mean_action=False, return_as_array=True)
                for p in individuals]
            n_actions = np.array(neighbours_p, dtype=object).mean(axis=1)
        return actions, n_actions

    def _linear_scan(self, item, k):
        # Lazy search while the warm-up period is not finished
        return super(OptimisedSWINN, self)._linear_scan(item, k)

    def _search(self, item, k, epsilon: float = 0.1, seed=None, exclude=None, return_dists=True, return_as_array=False) -> tuple[list, list]:
        # Search has already been performed and the seed's neighbours didn't change
        if return_as_array and seed is not None and seed.search_up_to_date:
            if return_dists:
                return seed._search_neighbours[:k], seed._search_distributions[:k]
            else:
                return seed._search_neighbours[:k]
        # Limiter for the distance bound
        distance_scale = 1 + epsilon
        # Distance threshold for early stops
        distance_bound = math.inf

        if exclude is None:
            exclude = set()
        else:
            exclude = {exclude.uuid}

        if seed is None:
            # Make sure the starting point for the search is valid
            while True:
                # Random seed point to start the search
                seed = self._rng.choice(self._data)
                if not seed.is_isolated() and seed.uuid not in exclude:
                    break

        # To avoid computing distances more than once for a given node
        visited = {seed.uuid}
        visited |= exclude
        # visited = {v: True for v in visited}

        # Search pool is a minimum heap
        pool = []
        pool_len = 0

        # Results are stored in a maximum heap
        result = []
        result_len = 0

        # c_dist, c_n = heapq.heappop(pool)
        c_dist, c_n = 0, seed
        while c_dist < distance_bound:
            for n in c_n.all_neighbors():
                if n.uuid in visited:
                    continue

                # TODO: assumption that seed.item is item in most searches, or a comparison with current node
                #  has been made in the past for given neighbour
                if c_n.uuid == seed.uuid:
                    # _, _, dist = seed.get_edge(n)
                    # try:
                    #     dist = seed.edges[n]
                    #     self.e_calls += 1
                    # except KeyError:
                    #     dist = seed.r_edges[n]
                    #     self.re_calls += 1
                    try:
                        dist = c_n.uedges[n.uuid]
                        self.e_calls += 1
                    except KeyError:
                        # print(seed.uuid, c_n.uuid, n.uuid)
                        # print(c_n.ur_edges, c_n.uedges)
                        # print("+"*20)
                        # print(c_n.r_edges, c_n.edges)
                        dist = c_n.ur_edges[n.uuid]
                        self.re_calls += 1
                elif seed.uuid in self.individual_comparisons:
                    try:
                        dist = self.individual_comparisons[seed.uuid][n.uuid]
                    except KeyError:
                        dist = self.dist_func(item, n.item)
                        self.individual_comparisons[seed.uuid][n.uuid] = dist
                    # try:
                    #     self.individual_comparisons[n.uuid][seed.uuid] = dist
                    # except KeyError:
                    #     self.individual_comparisons[n.uuid] = {seed.uuid: dist}

                elif n.uuid in self.individual_comparisons:
                    try:
                        dist = self.individual_comparisons[n.uuid][seed.uuid]
                    except KeyError:
                        dist = self.dist_func(item, n.item)
                        self.individual_comparisons[n.uuid][seed.uuid] = dist
                    # try:
                    #     self.individual_comparisons[seed.uuid][n.uuid] = dist
                    # except KeyError:
                    #     self.individual_comparisons[seed.uuid] = {n.uuid: dist}

                #
                else:
                    dist = self.dist_func(item, n.item)
                    self.individual_comparisons[n.uuid] = {seed.uuid: dist}
                    # TODO: added
                    self.individual_comparisons[seed.uuid] = {n.uuid: dist}

                if result_len < k:
                    heapq.heappush(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                    result_len += 1
                    pool_len += 1
                elif dist < -result[0][0]:
                    heapq.heapreplace(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                    pool_len += 1

                # if result_len < k:
                #     heapq.heappush(result, (-dist, n.uuid))
                #     heapq.heappush(pool, (dist, n.uuid))
                #     distance_bound = distance_scale * -result[0][0]
                #     result_len += 1
                #     pool_len += 1
                # elif dist < -result[0][0]:
                #     heapq.heapreplace(result, (-dist, n.uuid))
                #     heapq.heappush(pool, (dist, n.uuid))
                #     distance_bound = distance_scale * -result[0][0]
                #     pool_len += 1

                visited.add(n.uuid)
                # visited[n.uuid] = True


            if pool_len == 0:
                break
            c_dist, c_n = heapq.heappop(pool)
            # TODO:
            # orig_c_n = c_n
            # try:
            #     c_n = self._data[c_n - self._uuid]
            # except IndexError as e:
            #     print(c_n - self._uuid, c_n, self._uuid)
            #     raise e

            # assert c_n.uuid == orig_c_n
            # idx = c_n.uuid - self._uuid
            # assert c_n.uuid == self._data[idx], f"{[c_n, self._data[idx]]}, {c_n.uuid, self._uuid, self._min_uuid, self.maxlen, self._len_data}"
            # c_n = self._data
            pool_len -= 1

        if len(result) == 0:
            return ([], []) if return_dists else []

        result.sort(reverse=True)
        if return_dists:
            neighbors, dists = map(list, zip(*((r[1], -r[0]) for r in result)))
            # neighbors, dists = map(list, zip(*((self._data[r[1] - self._uuid], -r[0]) for r in result)))
            # The current neighbours have been updated
            nneighbors = [r[1].item for r in result]
            if seed is not None:
                # nneighbors = [r[1].item for r in result]
                seed.search_up_to_date = True
                seed._search_neighbours = nneighbors
                seed._search_distributions = dists
                seed._search_n_actions = [n[1] for n in nneighbors]
            if return_as_array:
                neighbors = nneighbors
            return neighbors, dists
        else:
            # neighbors = [r[1] for r in result]
            neighbors = [r[1].item for r in result]
            # neighbors = [self._data[r[1] - self._uuid] for r in result]
            # The current neighbours have been updated
            if seed is not None:
                seed.search_up_to_date = True
                seed._search_neighbours = neighbors
                seed._search_distributions = None
                seed._search_n_actions = [n[1] for n in neighbors]
            return neighbors

    def _search_actions_only(self, item, k, epsilon: float = 0.1, seed=None, exclude=None, return_mean_action=True, return_as_array=False) -> tuple[list, list]:
        # Search has already been performed and the seed's neighbours didn't change
        if return_as_array and seed is not None and seed.search_up_to_date:
            n_actions = seed._search_n_actions[:k]
            if return_mean_action:
                return sum(n_actions)/len(n_actions)
            else:
                return n_actions
        # Limiter for the distance bound
        distance_scale = 1 + epsilon
        # Distance threshold for early stops
        distance_bound = math.inf

        if exclude is None:
            exclude = set()
        else:
            exclude = {exclude.uuid}

        if seed is None:
            # Make sure the starting point for the search is valid
            while True:
                # Random seed point to start the search
                seed = self._rng.choice(self._data)
                if not seed.is_isolated() and seed.uuid not in exclude:
                    break

        # To avoid computing distances more than once for a given node
        visited = {seed.uuid}
        visited |= exclude
        # visited = {v: True for v in visited}

        # Search pool is a minimum heap
        pool = []
        pool_len = 0

        # Results are stored in a maximum heap
        result = []
        result_len = 0

        # c_dist, c_n = heapq.heappop(pool)
        c_dist, c_n = 0, seed
        while c_dist < distance_bound:
            for n in c_n.all_neighbors():
                if n.uuid in visited:
                    continue

                # TODO: assumption that seed.item is item in most searches, or a comparison with current node
                #  has been made in the past for given neighbour
                if c_n.uuid == seed.uuid:
                    try:
                        dist = c_n.uedges[n.uuid]
                        self.e_calls += 1
                    except KeyError:
                        dist = c_n.ur_edges[n.uuid]
                        self.re_calls += 1
                elif seed.uuid in self.individual_comparisons:
                    try:
                        dist = self.individual_comparisons[seed.uuid][n.uuid]
                    except KeyError:
                        dist = self.dist_func(item, n.item)
                        self.individual_comparisons[seed.uuid][n.uuid] = dist
                    # try:
                    #     self.individual_comparisons[n.uuid][seed.uuid] = dist
                    # except KeyError:
                    #     self.individual_comparisons[n.uuid] = {seed.uuid: dist}

                elif n.uuid in self.individual_comparisons:
                    try:
                        dist = self.individual_comparisons[n.uuid][seed.uuid]
                    except KeyError:
                        dist = self.dist_func(item, n.item)
                        self.individual_comparisons[n.uuid][seed.uuid] = dist
                    # try:
                    #     self.individual_comparisons[seed.uuid][n.uuid] = dist
                    # except KeyError:
                    #     self.individual_comparisons[seed.uuid] = {n.uuid: dist}

                #
                else:
                    dist = self.dist_func(item, n.item)
                    self.individual_comparisons[n.uuid] = {seed.uuid: dist}
                    # TODO: added
                    self.individual_comparisons[seed.uuid] = {n.uuid: dist}

                if result_len < k:
                    heapq.heappush(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                    result_len += 1
                    pool_len += 1
                elif dist < -result[0][0]:
                    heapq.heapreplace(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                    pool_len += 1

                # if result_len < k:
                #     heapq.heappush(result, (-dist, n.uuid))
                #     heapq.heappush(pool, (dist, n.uuid))
                #     distance_bound = distance_scale * -result[0][0]
                #     result_len += 1
                #     pool_len += 1
                # elif dist < -result[0][0]:
                #     heapq.heapreplace(result, (-dist, n.uuid))
                #     heapq.heappush(pool, (dist, n.uuid))
                #     distance_bound = distance_scale * -result[0][0]
                #     pool_len += 1

                visited.add(n.uuid)
                # visited[n.uuid] = True

            if pool_len == 0:
                break
            c_dist, c_n = heapq.heappop(pool)
            pool_len -= 1

        if len(result) == 0:
            return []

        result.sort(reverse=True)
        neighbors, n_actions, dists = map(list, zip(*((r[1].item, r[1].item[1], -r[0]) for r in result)))
        if seed is not None:
            # nneighbors = [r[1].item for r in result]
            seed.search_up_to_date = True
            seed._search_neighbours = neighbors
            seed._search_distributions = dists
            seed._search_n_actions = n_actions
            seed._search_mean_n_actions = sum(seed._search_n_actions) / len(seed._search_n_actions)
        if return_mean_action:
            if seed is not None:
                return seed._search_mean_n_actions
            return sum(n_actions)/len(n_actions)
        else:
            return n_actions

    def _safe_node_removal(self):
        """Remove the oldest data point from the search graph.

        Make sure nodes are accessible from any given starting point after removing the oldest
        node in the search graph. New traversal paths will be added in case the removed node was
        the only bridge between its neighbors.

        """
        node = self._data.popleft()

        ########### added
        # Delete all comparisons of this node with others  TODO: does not always get computed, just remove when that node gets removed
        # try:
        # for n in self.individual_comparisons[node.uuid]:
        #     try:
        #         del self.individual_comparisons[n][node.uuid]
        #     except KeyError as e:
        #         # print(node)
        #         # print(n)
        #         # print(self.individual_comparisons)
        #         # print(self._data)
        #         # raise e
        #         pass
        # except KeyError as e:
        #     pass
            # print(node.uuid, node)
            # print(self.individual_comparisons.keys())
            # raise e
        # Delete the nodes comparisons
        # try:
        del self.individual_comparisons[node.uuid]
        # except KeyError as e:
        #     pass
        ##########

        # Get previous neighborhood info
        rns = node.r_neighbors()[0]
        ns = node.neighbors()[0]
        node.not_up_to_date()
        node.farewell()

        # Nodes whose only direct neighbor was the removed node
        rns = {rn for rn in rns if not rn.has_neighbors()}
        # Nodes whose only reverse neighbor was the removed node
        ns = {n for n in ns if not n.has_rneighbors()}

        affected = list(rns | ns)
        isolated = rns.intersection(ns)

        # First we handle the unreachable nodes
        for al in isolated:
            # if al == node:
            #     continue  # TODO:
            neighbors, dists = self._search(al.item, self.graph_k)
            al.fill(neighbors, dists)
            al.search_up_to_date = True
            al._search_neighbours = [n.item for n in neighbors]
            al._search_distributions = dists
            al._search_n_actions = [n.item[1] for n in neighbors]
            al._search_mean_n_actions = sum(al._search_n_actions) / len(al._search_n_actions)

        rns -= isolated
        ns -= isolated
        ns = tuple(ns)
        len_ns = len(ns)

        # Nodes with no direct neighbors
        for rn in rns:
            seed = None
            # Check the group of nodes without reverse neighborhood for seeds
            # Thus we can join two separate groups
            if len_ns > 0:
                seed = self._rng.choice(ns)

            # Use the search index to create new connections
            neighbors, dists = self._search(rn.item, self.graph_k, seed=seed, exclude=rn, return_dists=True)
            rn.fill(neighbors, dists)
            rn.search_up_to_date = True
            rn._search_neighbours = [n.item for n in neighbors]
            rn._search_distributions = dists
            rn._search_n_actions = [n.item[1] for n in neighbors]
            rn._search_mean_n_actions = sum(rn._search_n_actions) / max(1, len(rn._search_n_actions))

        self._refine(affected)  # TODO: with search

    def _refine(self, nodes: list[OptimisedVertex] = None):
        """Update the nearest neighbor graph to improve the edge distances.

        Parameters
        ----------
        nodes
            The list of nodes for which the neighborhood refinement will be applied.
            If `None`, all nodes will have their neighborhood enhanced.
        """

        if nodes is None:
            nodes = [n for n in self]

        min_changes = self.delta * self.graph_k * len(nodes)

        tried = set()
        for _ in range(self.n_iters):
            total_changes = 0

            new = collections.defaultdict(set)
            old = collections.defaultdict(set)

            # Expand undirected neighborhood
            for node in nodes:
                neighbors = node.neighbors()[0]
                flags = node.sample_flags

                for neigh, flag in zip(neighbors, flags):
                    # To avoid evaluating previous neighbors again
                    tried.add((node.uuid, neigh.uuid))
                    if flag:
                        new[node].add(neigh)
                        new[neigh].add(node)
                    else:
                        old[node].add(neigh)
                        old[neigh].add(node)

            # Limits the maximum number of edges to explore and update sample flags
            for node in nodes:
                if len(new[node]) > self.max_candidates:
                    # new[node] = self._rng.sample(tuple(new[node]), self.max_candidates)  # type: ignore
                    new[node] = self._rng.sample(new[node], self.max_candidates)  # type: ignore
                # else:
                #     new[node] = new[node]

                if len(old[node]) > self.max_candidates:
                    # old[node] = self._rng.sample(tuple(old[node]), self.max_candidates)  # type: ignore
                    old[node] = self._rng.sample(old[node], self.max_candidates)  # type: ignore
                # else:
                #     old[node] = old[node]

                node.sample_flags = new[node]

            # Perform local joins an attempt to improve the neighborhood
            for node in nodes:
                # The origin of the join must have a boolean flag set to true
                for n1 in new[node]:
                    # Consider connections between vertices whose boolean flags are both true
                    # n2s = new[node].union(old[node])
                    for n2 in new[node]:
                    # for n2 in n2s:
                        if n1.uuid == n2.uuid or n1.is_neighbor(n2):
                            continue

                        if (n1.uuid, n2.uuid) in tried or (n2.uuid, n1.uuid) in tried:
                            continue

                        # Distance may already have been computed
                        try:
                            dist = self.individual_comparisons[n1.uuid][n2.uuid]
                        except KeyError:
                            try:
                                dist = self.individual_comparisons[n2.uuid][n1.uuid]
                            except KeyError:
                                dist = self.dist_func(n1.item, n2.item)
                                try:
                                    self.individual_comparisons[n1.uuid][n2.uuid] = dist
                                except KeyError:
                                    self.individual_comparisons[n1.uuid] = {n2.uuid: dist}
                                try:
                                    self.individual_comparisons[n2.uuid][n1.uuid] = dist
                                except KeyError:
                                    self.individual_comparisons[n2.uuid] = {n1.uuid: dist}
                        # dist = self.dist_func(n1.item, n2.item)
                        total_changes += n1.push_edge(n2, dist, self.graph_k)
                        total_changes += n2.push_edge(n1, dist, self.graph_k)

                        tried.add((n1.uuid, n2.uuid))

                    # Or one of the connections has a boolean flag set to false
                    for n2 in old[node]:
                        if n1.uuid == n2.uuid or n1.is_neighbor(n2):
                            continue

                        if (n1.uuid, n2.uuid) in tried or (n2.uuid, n1.uuid) in tried:
                            continue

                        try:
                            dist = self.individual_comparisons[n1.uuid][n2.uuid]
                        except KeyError:
                            try:
                                dist = self.individual_comparisons[n2.uuid][n1.uuid]
                            except KeyError:
                                dist = self.dist_func(n1.item, n2.item)
                                try:
                                    self.individual_comparisons[n1.uuid][n2.uuid] = dist
                                except KeyError:
                                    self.individual_comparisons[n1.uuid] = {n2.uuid: dist}
                                try:
                                    self.individual_comparisons[n2.uuid][n1.uuid] = dist
                                except KeyError:
                                    self.individual_comparisons[n2.uuid] = {n1.uuid: dist}
                        # dist = self.dist_func(n1.item, n2.item)
                        total_changes += n1.push_edge(n2, dist, self.graph_k)
                        total_changes += n2.push_edge(n1, dist, self.graph_k)

                        tried.add((n1.uuid, n2.uuid))

            # Stopping criterion
            if total_changes <= min_changes:
                break

        # Reduce the number of edges, if needed
        for n in nodes:
            n.prune(self.prune_prob, self.max_candidates, self._rng)

        # Ensure that no node is isolated in the graph
        self._fix_graph()

    def get_graph(self, features):
        """Get the graph structure"""
        from scenario.job_hiring.features import HiringFeature
        from scenario.fraud_detection.env import FraudFeature

        first_feature = list(features)[0]
        if isinstance(first_feature, HiringFeature):
            env_name = "job"
        elif isinstance(first_feature, FraudFeature):
            env_name = "fraud"
        else:
            raise ValueError(f"expected supported feature type. Given: {features}")

        nodes = []
        edges = []
        for i, node in enumerate(self._data):
            # print(node.uuid)
            ind, action = node.item
            # Get node id for display
            # TODO: abstract to work for other environments
            individual = {f: v for f, v in zip(features, ind)}

            if env_name == "job":
                nat = "belgian" if individual[HiringFeature.nationality] == 0.0 else "foreign"
                gen = "man" if individual[HiringFeature.gender] == 0.0 else "woman"
                selector_id = f"{nat}_{gen}"
                suffix_order = [HiringFeature.married, HiringFeature.degree, HiringFeature.extra_degree,
                                HiringFeature.language_dutch, HiringFeature.language_french,
                                HiringFeature.language_english, HiringFeature.language_german]
                selector_suffixes = {f: f"_{f.name.replace('_', '-')}" for f in suffix_order}
                for f, ss in selector_suffixes.items():
                    if individual[f]:
                        selector_id += ss
                hired = "rejected" if action == 0 else "hired"
                selector_id += f"_{hired}"
                label = f"(Node {node.uuid}) Age {int(individual[HiringFeature.age] * (65 - 18) + 18)}, Exp. {int(individual[HiringFeature.experience] * (65 - 18))}"
            else:

                pass

            # Add node
            nodes.append({"classes": ['node', "individual", selector_id],
                          "data": {"id": node.uuid, "label": label,
                                   "individual": ind,
                                   "action": action,
                                   }})

            # Add neighbours
            for n, distance in zip(*node.neighbors()):
                # Round for visibility
                dist = round(distance, 5)

                edge = {'classes': ['edge', "nn"],
                        'data': {'id': f"{node.uuid}_{n.uuid}", 'source': node.uuid, 'target': n.uuid,
                                 'weight': dist,  # 'opacity': max(dist, 0.25),
                                 'arrow_weight': 1.5}}
                edges.append(edge)

        output = {
            'id': 'cytoscape-neighbours',
            'elements': nodes + edges,
        }
        print(len(nodes), "nodes,", len(edges), "edges")

        return output

    def get_representatives_graph(self, max_degree, max_consider=5):
        """Get the graph representatives"""
        adjacency_matrix = np.zeros(shape=(len(self._data), len(self._data)))
        ids = {n.uuid: i for i, n in enumerate(self._data)}

        for node in self._data:
            node_id = ids[node.uuid]
            # Neighbours
            for n, dist in node.edges.items():
                n_id = ids[n.uuid]
                adjacency_matrix[node_id, n_id] = 1
            # Reverse neighbours will be added by n as neighbour

        total_paths = np.zeros_like(adjacency_matrix)
        for degree in range(1, max_degree + 1):
            adj_d = np.linalg.matrix_power(adjacency_matrix, degree)
            total_paths += adj_d

        total_sorted = np.argsort(-total_paths)  # most connections to least
        consider = total_sorted[:, :max_consider]

        # Store all (in)direct connections between nodes and representatives
        all_connections = {}
        representatives = set()
        for idx, row in enumerate(consider):
            representatives.update([self._data[n].uuid for n in row])
            for n in row:
                repr_node = self._data[n]
                try:
                    node = self._data[idx]
                except IndexError as e:
                    print(len(self._data), idx, n)
                    raise e
                all_connections[(repr_node.uuid, node.uuid)] = node.is_neighbor(repr_node)
        return representatives, all_connections


if __name__ == '__main__':
    from scenario.job_hiring.features import ApplicantGenerator, HiringFeature
    from scenario.job_hiring.env import JobHiringEnv

    sseed = 0
    n = 3000
    n_max = 1000

    warm_up = 100  # 500
    graph_k = 10
    k = 5
    epsilon = 0.1

    print(f"seed={sseed}, num_samples={n}, warm-up={warm_up}, graph_k={graph_k}, k={k}, epsilon={epsilon}")

    ag = ApplicantGenerator(seed=sseed, csv="../../../scenario/job_hiring/data/belgian_population.csv")
    population = ag.sample(n=n)
    env = JobHiringEnv(exclude_from_distance=(HiringFeature.age, HiringFeature.gender, HiringFeature.nationality, HiringFeature.married),
                       team_size=100, seed=sseed, episode_length=1000,  # Required ep length for pcn
                       diversity_weight=0, applicant_generator=ag,
                       )

    nearest_neighbours = OptimisedSWINN(graph_k=graph_k,
                                        dist_func=FunctionWrapper(scipy.spatial.distance.euclidean),
                                        # dist_func=FunctionWrapper(lambda state1, state2: env.minkowski_metric(state1, state2)),
                                        warm_up=warm_up, maxlen=n_max, seed=sseed)

    import cProfile
    with cProfile.Profile() as pr:
        t0 = time.time()
        for t, p in enumerate(population):
            individual = CombinedState(sample_context={}, sample_individual=p).to_array(individual_only=True)
            action = int(individual[1])
            nearest_neighbours.append((individual, action))
            # print(t)
            #
            nearest = nearest_neighbours.get_nn_for_all(k=k, epsilon=epsilon)

        n_actions = np.array([np.mean([n.item[1] for n in nn[1]]) for nn in nearest])  # TODO: extract item in method
        actions = np.array([n[0][1] for n in nearest])
        CON = - abs(actions - n_actions).mean()
        print(CON)

        t1 = time.time()
        print("time new", t1 - t0)

        pr.print_stats(SortKey.CUMULATIVE)
        # import pstats
        # p = pstats.Stats(pr)
        # p = p.sort_stats("line")
        # p.print_callers('__hash__')

        print(nearest_neighbours.e_calls, nearest_neighbours.re_calls, sum([nearest_neighbours.e_calls, nearest_neighbours.re_calls]))
