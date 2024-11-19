import math
import numpy as np
from typing import List, Tuple, Union, Dict
from mplanner.core.environment import GridNode

def tuple_ceil(input_tuple: Tuple[float, ...]) -> Tuple[int, ...]:
    """
    Takes a tuple of numbers and returns a tuple with the ceiling of each element.
    
    Args:
        input_tuple (Tuple[float, ...]): A tuple of numbers (floats).
    
    Returns:
        Tuple[int, ...]: A tuple with the ceiling values of each number in the input.
    """
    if not isinstance(input_tuple, tuple):
        raise ValueError("Input must be a tuple.")
    
    return tuple(math.ceil(element) for element in input_tuple)


def add_offset_to_tuple(input_tuple: Tuple[int, ...], offset: Union[int, float]) -> Tuple[int, ...]:
    """
    Adds an offset to each element of a tuple.
    
    Args:
        input_tuple (Tuple[int, ...]): A tuple of integers.
        offset (Union[int, float]): The value to be added to each element in the tuple.
    
    Returns:
        Tuple[int, ...]: A new tuple with the offset added to each element.
    """
    if not isinstance(input_tuple, tuple):
        raise ValueError("Input must be a tuple.")
    if not isinstance(offset, (int, float)):
        raise ValueError("Offset must be an integer or float.")
    
    return tuple(int(element + offset) for element in input_tuple)


def backtrack(node: GridNode) -> List[GridNode]:
    """
    Generate a path by backtracking from the given node to its root (start) node.

    Args:
        node (GridNode): The node from which to start backtracking.

    Returns:
        List[GridNode]: A list of nodes representing the path, ordered from start to the given node.
    """
    path = [node]
    while node.parent:
        node = node.parent
        path.append(node)
    path.reverse()  # Ensure the path is ordered from start to end
    return path


def bidirectional_backtrack(node_a: GridNode, node_b: GridNode) -> List[GridNode]:
    """
    Generate a complete path for bi-directional A* by backtracking from both start and end nodes.

    Args:
        node_a (GridNode): The node from which to backtrack to reconstruct the path from the start.
        node_b (GridNode): The node from which to backtrack to reconstruct the path to the end.

    Returns:
        List[GridNode]: A combined path from the start node to the end node, passing through the meeting point.
    """
    path_a = backtrack(node_a)  # Path from start to meeting point
    path_b = backtrack(node_b)  # Path from end to meeting point
    path_b.reverse()  # Reverse the second path to ensure correct order
    return path_a + path_b  # Combine the two paths into one

def calculate_grid_size(bounds: Dict, resolution: float) -> Tuple:
    """
    Calculates the grid size based on environment bounds and resolution.

    Args:
        bounds (dict): Dictionary containing the 'min' and 'max' bounds of the environment in (x, y, z).
        resolution (float): Size of each grid cell.

    Returns:
        tuple: Grid dimensions (width, height, depth).
    """
    grid_size = tuple(
        math.ceil((bounds["max"][i] - bounds["min"][i]) / resolution) for i in range(3)
    )
    return grid_size


def to_grid_coords(position: Tuple, resolution: float, origin: Tuple) -> Tuple:
    """
    Converts a position in decimal coordinates to grid indices.

    Args:
        position (tuple): Decimal (X, Y, Z) coordinates.
        resolution (float): Size of each grid cell.
        origin (tuple): Origin (X, Y, Z) of the grid.

    Returns:
        tuple: Grid indices (x, y, z).
    """
    return tuple(int(round((p - o) / resolution)) for p, o in zip(position, origin))


def mark_obstacles(matrix: np.ndarray, position: Tuple, dimensions: Tuple, resolution: float, origin: Tuple):
    """
    Marks a cuboidal obstacle in the grid matrix.

    Args:
        matrix (numpy.ndarray): The grid matrix.
        position (tuple): Decimal (X, Y, Z) center of the obstacle.
        dimensions (tuple): Dimensions (X, Y, Z) of the obstacle.
        resolution (float): Size of each grid cell.
        origin (tuple): Origin (X, Y, Z) of the grid.
    """
    min_coords = [
        int((position[i] - dimensions[i] / 2 - origin[i]) / resolution) for i in range(3)
    ]
    max_coords = [
        int((position[i] + dimensions[i] / 2 - origin[i]) / resolution) for i in range(3)
    ]
    for x in range(max(0, min_coords[0]), min(matrix.shape[0], max_coords[0] + 1)):
        for y in range(max(0, min_coords[1]), min(matrix.shape[1], max_coords[1] + 1)):
            for z in range(max(0, min_coords[2]), min(matrix.shape[2], max_coords[2] + 1)):
                matrix[x, y, z] = 0  # Mark as obstacle