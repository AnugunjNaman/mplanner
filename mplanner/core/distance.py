import math
from typing import Union

def null(x: Union[int, float], y: Union[int, float], z: Union[int, float]) -> float:
    """
    Returns 0.0 for any input, representing no distance.

    Args:
        x (Union[int, float]): Distance along the x-axis.
        y (Union[int, float]): Distance along the y-axis.
        z (Union[int, float]): Distance along the z-axis.

    Returns:
        float: Always returns 0.0, indicating no distance.
    """
    return 0.0


def euclidean(x: Union[int, float], y: Union[int, float], z: Union[int, float]) -> float:
    """
    Calculates the Euclidean distance between two points in 3D space.

    The Euclidean distance is the straight-line distance between two points, given by:

    \\[ d = \\sqrt{x^2 + y^2 + z^2}  \\] 

    Args:
        x (Union[int, float]): Distance along the x-axis.
        y (Union[int, float]): Distance along the y-axis.
        z (Union[int, float]): Distance along the z-axis.

    Returns:
        float: The Euclidean distance between the points.
    """
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def manhattan(x: Union[int, float], y: Union[int, float], z: Union[int, float]) -> float:
    """
    Calculates the Manhattan distance between two points in 3D space.

    The Manhattan distance (or L1 distance) is the sum of the absolute differences along each axis:

    \\[ d = |x| + |y| + |z| \\]

    Args:
        x (Union[int, float]): Distance along the x-axis.
        y (Union[int, float]): Distance along the y-axis.
        z (Union[int, float]): Distance along the z-axis.

    Returns:
        float: The Manhattan distance between the points.
    """
    return abs(x) + abs(y) + abs(z)


def chebyshev(x: Union[int, float], y: Union[int, float], z: Union[int, float]) -> float:
    """
    Calculates the Chebyshev distance between two points in 3D space.

    The Chebyshev distance is the maximum absolute difference along any axis:

    \\[ d = \\max(|x|, |y|, |z|) \\]

    Args:
        x (Union[int, float]): Distance along the x-axis.
        y (Union[int, float]): Distance along the y-axis.
        z (Union[int, float]): Distance along the z-axis.

    Returns:
        float: The Chebyshev distance between the points.
    """
    return max(abs(x), abs(y), abs(z))


def octile(x: Union[int, float], y: Union[int, float], z: Union[int, float]) -> float:
    """
    Calculates the generalized octile distance between two points in 3D space.

    Octile distance is a heuristic metric used in 3D grid-based pathfinding. It combines straight,
    diagonal (2D), and diagonal (3D) moves with the following weights:
    - Straight moves: cost = 1
    - Diagonal moves in 2D: cost = sqrt(2) - 1 
    - Diagonal moves in 3D: cost = sqrt(3) - sqrt(2) 

    The formula is as follows:

    \\[ d = \\text{max_distance} + (\\sqrt{2} - 1) \\cdot \\text{mid_distance} + (\\sqrt{3} - \\sqrt{2}) \\cdot \\text{min_distance} \\]

    where:
        - \\[ \\text{max_distance} = \\max(|x|, |y|, |z|) \\]
        - \\[ \\text{mid_distance} = \\text{total_distance} - \\text{max_distance} - \\text{min_distance} \\]
        - \\[ \\text{min_distance} = \\min(|x|, |y|, |z|) \\]
        - \\[ \\text{total_distance} = |x| + |y| + |z| \\]

    Args:
        x (Union[int, float]): Distance along the x-axis.
        y (Union[int, float]): Distance along the y-axis.
        z (Union[int, float]): Distance along the z-axis.

    Returns:
        float: The generalized octile distance between the points.
    """
    x, y, z = abs(x), abs(y), abs(z)
    
    # Find the maximum, middle, and minimum distances among the axes
    max_distance = max(x, y, z)
    min_distance = min(x, y, z)
    mid_distance = x + y + z - max_distance - min_distance
    
    # Calculate the octile distance
    return max_distance + (math.sqrt(2) - 1) * mid_distance + (math.sqrt(3) - math.sqrt(2)) * min_distance
