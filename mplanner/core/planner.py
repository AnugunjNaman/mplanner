import time  # For enforcing time constraints
from typing import Callable, List, Optional, Tuple, Union

from mplanner.core.environment import DMove, Grid, GridNode
from mplanner.core.priority_queue import PriorityQueueStructure


MAX_RUNS = float("inf")
TIME_LIMIT = float("inf")
FORWARD_SEARCH = 1
BACKWARD_SEARCH = 2

class Planner:
    """
    A base class for pathfinding algorithms. Serves as a template for implementing
    specific pathfinding strategies like A*, Dijkstra, etc.

    Attributes:
        time_limit (float): Maximum runtime in seconds before aborting the search.
        max_runs (Union[int, float]): Maximum iterations before aborting the search.
        weighted (bool): Indicates whether the algorithm supports weighted nodes.
        diagonal_movement (int): Enum specifying if diagonal movement is allowed.
        weight (int): Weight for the edges in the pathfinding algorithm.
        heuristic (Optional[Callable]): Function to calculate heuristic distance between two nodes.
        start_time (float): Tracks the start time of the algorithm.
        runs (int): Counter for the number of iterations.
    """

    def __init__(
        self,
        heuristic: Optional[Callable] = None,
        weight: int = 1,
        diagonal_movement: int = DMove.NEVER,
        weighted: bool = True,
        time_limit: float = TIME_LIMIT,
        max_runs: Union[int, float] = MAX_RUNS,
    ):
        self.time_limit = time_limit
        self.max_runs = max_runs
        self.weighted = weighted
        self.diagonal_movement = diagonal_movement
        self.weight = weight
        self.heuristic = heuristic

        self.start_time: float = 0.0  # Track the start time of the algorithm
        self.runs: int = 0  # Counter for the number of iterations

    def distance_heuristic(
        self, node_a: GridNode, node_b: GridNode, heuristic: Optional[Callable] = None
    ) -> float:
        """
        Calculates the heuristic distance between two nodes.

        Args:
            node_a (GridNode): The first node.
            node_b (GridNode): The second node.
            heuristic (Optional[Callable]): Custom heuristic function. Defaults to None.

        Returns:
            float: Heuristic value representing the estimated cost.
        """
        if not heuristic:
            heuristic = self.heuristic
        return heuristic(
            abs(node_a.x - node_b.x),
            abs(node_a.y - node_b.y),
            abs(node_a.z - node_b.z),
        )

    def get_neighboring_nodes(
        self, grid: Grid, current_node: GridNode, diagonal_movement: Optional[int] = None
    ) -> List[GridNode]:
        """
       Identifies the neighbors of a node on the grid.

        Args:
            grid (Grid): The grid containing the nodes.
            current_node (GridNode): The node whose neighbors are being searched.
            diagonal_movement (Optional[int]): Specifies if diagonal movement is allowed. Defaults to None.

        Returns:
            List[GridNode]: List of neighboring nodes.
        """
        if diagonal_movement is None:
            diagonal_movement = self.diagonal_movement
        return grid.neighbors(current_node, diagonal_movement=diagonal_movement)

    def keep_running(self):
        if self.runs >= self.max_runs:
            raise Exception(
                f"{self.__class__.__name__} exceeded {self.max_runs} iterations without finding the destination."
            )

        if time.time() - self.start_time >= self.time_limit:
            raise Exception(
                f"{self.__class__.__name__} exceeded the time limit of {self.time_limit} seconds, aborting!"
            )

    def process_node(
        self,
        grid: Grid,
        neighbor_node: GridNode,
        parent_node: GridNode,
        target_node: GridNode,
        open_list: List,
        open_value: int = 1,
    ):
        """
        Evaluates a node to determine if it should be added to the open list.

        Args:
            grid (Grid): The grid containing the nodes.
            neighbor_node (GridNode): The neighbor node being evaluated.
            parent_node (GridNode): The current node processing the neighbor.
            target_node (GridNode): The target node (goal) in the pathfinding process.
            open_list (List): The list of nodes to be processed next.
            open_value (int): Value to mark the node as opened. Defaults to 1.
        """
        # Calculate the cost to move from the parent node to the neighbor node
        cost_to_neighbor = parent_node.g + grid.calc_cost(parent_node, neighbor_node, self.weighted)

        if not neighbor_node.opened or cost_to_neighbor < neighbor_node.g:
            # Update node values if it is not opened or a shorter path is found
            previous_cost = neighbor_node.f
            neighbor_node.g = cost_to_neighbor
            neighbor_node.h = neighbor_node.h or self.distance_heuristic(neighbor_node, target_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            neighbor_node.parent = parent_node

            if not neighbor_node.opened:
                open_list.push_node(neighbor_node)
                neighbor_node.opened = open_value
            else:
                # Update node's position in the open list if the cost has been improved
                open_list.remove_node(neighbor_node, previous_cost)
                open_list.push_node(neighbor_node)

    def evaluate_neighbors(
        self,
        start_node: GridNode,
        target_node: GridNode,
        grid: Grid,
        open_list: List,
        open_value: int = 1,
        backtrace_by=None,
    ) -> Optional[List[GridNode]]:
        """
        Evaluates the neighbors of the current node and determines the next step.

        Args:
            start_node (GridNode): The start node of the search.
            target_node (GridNode): The goal node of the search.
            grid (Grid): The grid containing the nodes.
            open_list (List): The list of nodes to be processed next.
            open_value (int): Value to mark the node as opened. Defaults to 1.
            backtrace_by (optional): Custom backtrace method. Defaults to None.

        Returns:
            Optional[List[GridNode]]: The path if the target node is reached, otherwise None.
        """

        raise NotImplementedError("Please implement evaluate_neighbors in your planner subclass")

    def get_planned_path(self, start_node: GridNode, target_node: GridNode, grid: Grid) -> Tuple[List, int]:
        """
        Finds the shortest path between two nodes on a grid.

        Args:
            start_node (GridNode): The starting node.
            target_node (GridNode): The goal node.
            grid (Grid): The grid containing the nodes.

        Returns:
            Tuple[List, int]: A tuple containing the path as a list of nodes and the number of iterations performed.
        """
        self.start_time = time.time()  
        self.runs = 0 
        start_node.opened = True

        open_list = PriorityQueueStructure(start_node, grid)

        while len(open_list) > 0:
            self.runs += 1
            self.keep_running()

            path = self.evaluate_neighbors(start_node, target_node, grid, open_list)
            if path:
                return path, self.runs

        # Return empty path if no solution is found
        return [], self.runs

    def __repr__(self):
        return f"<{self.__class__.__name__}(diagonal_movement={self.diagonal_movement})>"
