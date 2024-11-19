import heapq
from typing import Callable, Union

from mplanner.core.environment import Grid, GridNode, World


class PriorityQueueStructure:
    """
    A priority queue data structure optimized for pathfinding algorithms.

    Manages an open list of nodes, allows efficient retrieval of the lowest-cost node,
    and supports marking nodes as removed for handling updates during pathfinding.

    Attributes:
        grid (Union[Grid, World]): The grid or world containing the nodes.
        open_list (list): The priority queue containing nodes as tuples.
        removed_node_tuples (set): Tracks nodes that are logically removed but not yet physically removed.
        priority queue_order (dict): Maps node identifiers to their priority queue order for tie-breaking.
        number_pushed (int): Counter to maintain priority queue stability during tie-breaking.
    """

    def __init__(self, node: GridNode, grid: Union[Grid, World]):
        self.grid = grid
        self._get_node_tuple = self._determine_node_retrieval_function()
        self._get_node = self._determine_node_function()
        self.open_list = [self._get_node_tuple(node, 0)]  # PriorityQueue of nodes (priority queue)
        self.removed_node_tuples = set()  # Tracks logically removed nodes
        self.priority_queue_order = {}  # Tracks the order nodes are added (for tie-breaking)
        self.number_pushed = 0  # Counter for tie-breaking stability

    def _determine_node_retrieval_function(self) -> Callable:
        """
        Determines the function to create tuples for nodes based on the grid type.

        Returns:
            Callable: A function generating tuples for use in the priority queue.

        Raises:
            ValueError: If the grid type is unsupported.
        """
        if isinstance(self.grid, Grid):
            return lambda node, priority_queue_order: (node.f, priority_queue_order, *node.identifier)

        if isinstance(self.grid, World):
            return lambda node, priority_queue_order: (node.f, priority_queue_order, *node.identifier)

        raise ValueError("Unsupported grid type")

    def _determine_node_function(self) -> Callable:
        """
        Determines the function to retrieve a `GridNode` from its tuple representation.

        Returns:
            Callable: A function retrieving a `GridNode` from its tuple.

        Raises:
            ValueError: If the grid type is unsupported.
        """
        if isinstance(self.grid, Grid):
            return lambda node_tuple: self.grid.node(*node_tuple[2:])

        if isinstance(self.grid, World):
            return lambda node_tuple: self.grid.grids[node_tuple[5]].node(*node_tuple[2:5])

        raise ValueError("Unsupported grid type")

    def pop_node(self) -> GridNode:
        """
        Removes and returns the node with the lowest cost (`f`) from the priority queue.

        Skips nodes that have been logically removed.

        Returns:
            GridNode: The node with the lowest cost in the priority queue.
        """
        node_tuple = heapq.heappop(self.open_list)
        while node_tuple in self.removed_node_tuples:
            node_tuple = heapq.heappop(self.open_list)

        return self._get_node(node_tuple)

    def push_node(self, node: GridNode):
        """
        Adds a node to the priority queue.

        Args:
            node (GridNode): The node to add. Must have an `f` attribute for cost evaluation.
        """
        self.number_pushed += 1  # Increment counter for tie-breaking
        node_tuple = self._get_node_tuple(node, self.number_pushed)

        self.priority_queue_order[node.identifier] = self.number_pushed  # Track the order of insertion
        heapq.heappush(self.open_list, node_tuple)  # Add node to the priority_queue

    def remove_node(self, node: GridNode, old_f: float):
        """
        Marks a node as removed from the priority_queue.

        This logical removal prevents the node from being returned when popped.
        This approach is efficient for handling updates during pathfinding.

        Args:
            node (GridNode): The node to mark as removed.
            old_f (float): The previous cost of the node, used to identify it in the priority_queue.
        """
        priority_queue_order = self.priority_queue_order[node.identifier]
        node_tuple = self._get_node_tuple(node, priority_queue_order)
        self.removed_node_tuples.add(node_tuple)

    def __len__(self) -> int:
        return len(self.open_list)
