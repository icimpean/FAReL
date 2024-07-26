from __future__ import annotations

from river.neighbors.ann.nn_vertex import Vertex


class OptimisedVertex(Vertex):
    """Optimised SWINN Vertex"""
    _isolated: set[OptimisedVertex] = set()

    def __init__(self, item, uuid: int) -> None:
        # Super call
        super(OptimisedVertex, self).__init__(item, uuid)
        # Faster lookup during _search
        self.uedges: dict[int, float] = {}
        self.ur_edges: dict[int, float] = {}
        #
        self.neighbours: set[OptimisedVertex] = set()
        self.search_up_to_date = False
        self._search_neighbours = None
        self._search_distributions = None
        self._search_n_actions = None
        self._search_mean_n_actions = None

    # TODO: remove
    def __str__(self):
        return f"OptimisedVertex({self.uuid})"

    def __hash__(self) -> int:
        return self.uuid

    def __eq__(self, other) -> bool:
        return self.uuid == other.uuid

    def __lt__(self, other) -> bool:
        return self.uuid < other.uuid

    def not_up_to_date(self):
        self.search_up_to_date = False
        self._search_neighbours = None
        self._search_distributions = None
        self._search_n_actions = None
        self._search_mean_n_actions = None

    def fill(self, neighbors: list[OptimisedVertex], dists: list[float]):
        for n, dist in zip(neighbors, dists):
            self.edges[n] = dist
            n.r_edges[self] = dist
            #
            self.uedges[n.uuid] = dist
            n.ur_edges[self.uuid] = dist
            #
            n.neighbours.add(self)
            n.not_up_to_date()
        #
        if self.flags is None:
            print("here")
        self.flags.update(neighbors)
        self.neighbours.update(neighbors)

        # Neighbors are ordered by distance
        if len(neighbors) != 0:  # TODO
            self.worst_edge = n
        self.not_up_to_date()

    def farewell(self):
        # Super call
        # super(OptimisedVertex, self).farewell()
        for rn in list(self.r_edges):
            rn.rem_edge(self)

        for n in list(self.edges):
            self.rem_edge(n)

        self.flags = None
        self.worst_edge = None

        OptimisedVertex._isolated.discard(self)
        #
        self.neighbours = None
        self.not_up_to_date()

    def add_edge(self, vertex: OptimisedVertex, dist):
        # Super call
        super(OptimisedVertex, self).add_edge(vertex, dist)
        #
        self.uedges[vertex.uuid] = dist
        vertex.ur_edges[self.uuid] = dist
        #
        self.neighbours.add(vertex)
        vertex.neighbours.add(self)
        self.not_up_to_date()
        vertex.not_up_to_date()

    def rem_edge(self, vertex: OptimisedVertex):
        # Super call
        super(OptimisedVertex, self).rem_edge(vertex)
        if not self.has_rneighbors():
            OptimisedVertex._isolated.add(self)
        #
        if vertex.uuid in self.uedges:
            del self.uedges[vertex.uuid]
        if self.uuid in vertex.ur_edges:
            del vertex.ur_edges[self.uuid]
        #
        vertex.neighbours.discard(self)
        self.neighbours.discard(vertex)
        self.not_up_to_date()
        vertex.not_up_to_date()

    def get_edge(self, vertex: Vertex):
        if vertex in self.edges:
            return self, vertex, self.edges[vertex]
        return vertex, self, self.r_edges[vertex]

    def is_neighbor(self, vertex):
        return vertex in self.neighbours

    def all_neighbors(self):
        return self.neighbours
