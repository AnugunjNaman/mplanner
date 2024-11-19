from typing import Callable, List, Optional, Tuple, Union
from mplanner.core.environment import DMove, Grid, GridNode
from mplanner.core.distance import manhattan, octile
from mplanner.core.utils import backtrack
from mplanner.core.planner import Planner
from mplanner.core.priority_queue import PriorityQueueStructure


class AStar(Planner):
    """
    A* Algorithm for 3D Pathfinding.

    This class implements a 3D version of the A* algorithm. It calculates the shortest path
    between a start and target node while considering movement costs and optional diagonal movement.

    Attributes:
        heuristic (Callable): Function to estimate the cost to the target node.
        weight (int): Weight factor for the cost of moving between nodes.
        diagonal_movement (int): Defines if and how diagonal movement is allowed.
        time_limit (float): Maximum time (in seconds) before the search aborts.
        max_runs (Union[int, float]): Maximum allowable iterations before termination.
    """
    def __init__(
        self,
        heuristic: Optional[Callable] = None,
        weight: int = 1,
        diagonal_movement: int = DMove.NEVER,
        time_limit: float = float("inf"),
        max_runs: Union[int, float] = float("inf"),
    ):
        super().__init__(
            heuristic=heuristic or manhattan,
            weight=weight,
            diagonal_movement=diagonal_movement,
            time_limit=time_limit,
            max_runs=max_runs,
        )
        # Default heuristic assignment
        if not self.heuristic:
            self.heuristic = manhattan if diagonal_movement == DMove.NEVER else octile

    def explore_neighbors(
        self,
        current_node: GridNode,
        target_node: GridNode,
        grid: Grid,
        open_list: List,
    ) -> Optional[List[GridNode]]:
        """
        Explore and process neighboring nodes for the current node.

        Args:
            current_node (GridNode): The node currently being processed.
            target_node (GridNode): The destination node.
            grid (Grid): The grid representing the search space.
            open_list (List): The priority queue containing nodes to be evaluated.

        Returns:
            Optional[List[GridNode]]: The path as a list of nodes if the target is reached, otherwise None.
        """
        # Retrieve valid neighbors
        neighbors = self.get_neighboring_nodes(grid, current_node)

        for neighbor in neighbors:
            if neighbor.closed:
                continue  # Skip nodes already processed

            # Compute cost from current node to neighbor
            tentative_g_cost = current_node.g + grid.calc_cost(current_node, neighbor, self.weighted)

            if not neighbor.opened or tentative_g_cost < neighbor.g:
                # Update costs and parent if a better path is found
                neighbor.g = tentative_g_cost
                neighbor.h = self.distance_heuristic(neighbor, target_node)
                neighbor.f = neighbor.g + neighbor.h
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
    ) -> Optional[List[GridNode]]:
        """
        Process the node with the lowest cost (f) and evaluate its neighbors.

        Args:
            start_node (GridNode): The start node of the pathfinding process.
            target_node (GridNode): The target (goal) node.
            grid (Grid): The grid representing the search space.
            open_list (List): The priority queue containing nodes to be evaluated.

        Returns:
            Optional[List[GridNode]]: A path to the target node if found, otherwise None.
        """
        current_node = open_list.pop_node()
        current_node.closed = True

        if current_node == target_node:
            # Path found
            return backtrack(current_node)

        self.explore_neighbors(current_node, target_node, grid, open_list)
        return None

    def get_planned_path(
        self, start_node: GridNode, target_node: GridNode, grid: Grid
    ) -> Tuple[List[GridNode], int]:
        """
        Executes the A* search to find a path from start to target.

        Args:
            start_node (GridNode): Starting node of the path.
            target_node (GridNode): Destination node of the path.
            grid (Grid): The 3D grid representing the search space.

        Returns:
            Tuple[List[GridNode], int]: Path as a list of nodes and the number of iterations performed.
        """
        # Initialize costs and open list
        start_node.g = 0
        start_node.f = 0
        start_node.h = self.distance_heuristic(start_node, target_node)
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
            f"<AStar3D(heuristic={self.heuristic.__name__}, weight={self.weight}, "
            f"diagonal_movement={self.diagonal_movement})>"
        )


