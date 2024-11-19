from typing import List, Tuple
from mplanner.core.environment import Grid, GridNode
from mplanner.core.planner import Planner
from mplanner.core.utils import backtrack
from mplanner.core.priority_queue import PriorityQueueStructure


class Dijkstra3D(Planner):
    """
    Dijkstra's Algorithm for 3D Pathfinding.

    This class implements Dijkstra's algorithm, which finds the shortest path from a start node
    to a target node in a 3D grid. Dijkstra's algorithm does not use a heuristic function.

    Attributes:
        weight (int): Weight factor for the cost of moving between nodes.
        diagonal_movement (int): Defines if and how diagonal movement is allowed.
        time_limit (float): Maximum time (in seconds) before the search aborts.
        max_runs (int): Maximum allowable iterations before termination.
    """
    def __init__(
        self,
        weight: int = 1,
        diagonal_movement: int = 0,
        time_limit: float = float("inf"),
        max_runs: int = float("inf"),
    ):
        super().__init__(
            heuristic=None,  # No heuristic for Dijkstra's algorithm
            weight=weight,
            diagonal_movement=diagonal_movement,
            time_limit=time_limit,
            max_runs=max_runs,
        )

    def explore_neighbors(
        self,
        current_node: GridNode,
        grid: Grid,
        open_list: List,
    ):
        """
        Explore and process neighbors of the current node during Dijkstra's search.

        Args:
            current_node (GridNode): The node currently being processed.
            grid (Grid): The grid representing the search space.
            open_list (List): The priority queue containing nodes to be evaluated.
        """
        neighbors = self.get_neighboring_nodes(grid, current_node)

        for neighbor in neighbors:
            if neighbor.closed:
                continue  # Skip nodes already processed

            # Compute cost to reach the neighbor
            tentative_g_cost = current_node.g + grid.calc_cost(current_node, neighbor, self.weighted)

            if not neighbor.opened or tentative_g_cost < neighbor.g:
                # Update costs and parent if a better path is found
                neighbor.g = tentative_g_cost
                neighbor.f = tentative_g_cost  # For Dijkstra's, f is equivalent to g
                neighbor.parent = current_node

                if not neighbor.opened:
                    open_list.push_node(neighbor)
                    neighbor.opened = True

    def evaluate_neighbors(
        self,
        start_node: GridNode,
        target_node: GridNode,
        grid: Grid,
        open_list: List,
    ) -> List[GridNode]:
        """
        Process the node with the lowest cost (f) and evaluate its neighbors.

        Args:
            start_node (GridNode): The starting node of the search.
            target_node (GridNode): The target node.
            grid (Grid): The grid representing the search space.
            open_list (List): The priority queue of nodes to evaluate.

        Returns:
            List[GridNode]: A path to the target node if found, otherwise None.
        """
        current_node = open_list.pop_node()
        current_node.closed = True

        if current_node == target_node:
            # Path found
            return backtrack(current_node)

        # Explore and process neighbors
        self.explore_neighbors(current_node, grid, open_list)
        return None

    def get_planned_path(
        self, start_node: GridNode, target_node: GridNode, grid: Grid
    ) -> Tuple[List[GridNode], int]:
        """
        Executes Dijkstra's algorithm to find the shortest path.

        Args:
            start_node (GridNode): The starting node.
            target_node (GridNode): The destination node.
            grid (Grid): The grid representing the search space.

        Returns:
            Tuple[List[GridNode], int]: The path as a list of nodes and the number of iterations performed.
        """
        # Initialize costs and open list
        start_node.g = 0
        start_node.f = 0  # f = g for Dijkstra's
        start_node.opened = True

        open_list = self.create_open_list(grid, start_node)

        while len(open_list) > 0:
            self.keep_running()
            path = self.evaluate_neighbors(start_node, target_node, grid, open_list)
            if path:
                return path, self.runs

        # No path found
        return [], self.runs

    def create_open_list(self, grid: Grid, start_node: GridNode) -> PriorityQueueStructure:
        """
        Creates and initializes the open list with the starting node.

        Args:
            grid (Grid): The grid representing the search space.
            start_node (GridNode): The starting node.

        Returns:
            PriorityQueueStructure: Initialized priority queue with the start node.
        """
        return PriorityQueueStructure(start_node, grid)

    def __repr__(self):
        return (
            f"<Dijkstra3D(weight={self.weight}, diagonal_movement={self.diagonal_movement})>"
        )
