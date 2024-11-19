import math
import warnings
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Union, Dict

USE_PLOTLY = True

MatrixType = Optional[Union[List[List[List[int]]], np.ndarray]]

class DMove(Enum):
    """
    Enumeration for diagonal movement options in pathfinding algorithms.

    Specifies whether diagonal movement is allowed in grid-based pathfinding.

    Attributes:
        ALWAYS (int): Indicates that diagonal movement is always allowed.
        NEVER (int): Indicates that diagonal movement is never allowed.
    """
    ALWAYS = 1
    NEVER = 0

@dataclass
class Node:
    """
    Basic node class to store calculated values for pathfinding algorithms.

    Attributes:
        h (float): Heuristic cost from this node to the goal (used in A*).
        g (float): Actual cost from the start node to this node.
        f (float): Total estimated cost of the path through this node (f = g + h).
        opened (int): Number of times this node has been opened during the search.
        closed (bool): Indicates whether this node has been processed and closed.
        parent (Optional[Node]): Reference to the parent node, used for backtracking the path.
        retain_count (int): Counter for recursion tracking in IDA*.
        tested (bool): Indicates if the node has been tested (used in IDA* and Jump-Point-Search).
    """
    h: float = field(default=0.0)
    g: float = field(default=0.0)
    f: float = field(default=0.0)
    opened: int = field(default=0)
    closed: bool = field(default=False)
    parent: Optional["Node"] = field(default=None)
    retain_count: int = field(default=0)
    tested: bool = field(default=False)

    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f

    def cleanup(self):
        """
        Resets all node values to their default state for reuse in pathfinding.
        """
        self.h = 0.0
        self.g = 0.0
        self.f = 0.0
        self.opened = 0
        self.closed = False
        self.parent = None
        self.retain_count = 0
        self.tested = False

@dataclass
class GridNode(Node):
    """
    Represents a node in a grid for pathfinding algorithms.

    Extends the basic `Node` class by including spatial coordinates and 
    grid-specific attributes, such as walkability, weight, and connections 
    to neighboring nodes.

    Attributes:
        x (int): The x-coordinate of the node in the grid.
        y (int): The y-coordinate of the node in the grid.
        z (int): The z-coordinate of the node in the grid.
        walkable (bool): Indicates whether this node is passable or blocked.
        weight (float): The traversal cost associated with this node, used in weighted algorithms.
        grid_id (Optional[int]): Identifier for the grid this node belongs to. Useful for managing multiple grids.
        connections (List[GridNode]): The list of neighboring nodes connected to this node.
    """
    x: int = 0
    y: int = 0
    z: int = 0
    
    walkable: bool = True
    weight: float = 1.0
    grid_id: Optional[int] = None

    connections: List["GridNode"] = field(default_factory=list)

    def __post_init__(self):
        super().__init__()
        self.identifier: Tuple = (
            (self.x, self.y, self.z) if self.grid_id is None else (self.x, self.y, self.z, self.grid_id)
        )

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
        if self.grid_id is not None:
            yield self.grid_id

    def connect(self, other_node: "GridNode"):
        """
        Connects this node to another node by adding the other node to its connections.
        """
        self.connections.append(other_node)


class Grid:
    """
    A class representing a 3D grid, which serves as a map for spatial navigation and pathfinding.
    
    Initializes a new grid instance.

    Args:
        width (int, optional): The width of the grid. Defaults to 0.
        height (int, optional): The height of the grid. Defaults to 0.
        depth (int, optional): The depth of the grid. Defaults to 0.
        matrix (optional, MatrixType): A 3D matrix (list or ndarray) where each element determines 
            if the corresponding node is walkable and its weight. If omitted, all nodes are set as walkable.
        grid_id (int, optional): A unique identifier for the grid. Defaults to None.
        inverse (bool, optional): If True, values in the matrix other than 0 are walkable; otherwise, 0 is walkable.
    
    """
    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        depth: int = 0,
        matrix: MatrixType = None,
        grid_id: Optional[int] = None,
        inverse: bool = False,
    ):
        self.width, self.height, self.depth = self._validate_dimensions(width, height, depth, matrix)
        self.nodes = (
            build_nodes(self.width, self.height, self.depth, matrix, inverse, grid_id)
            if self.is_valid_grid()
            else [[[]]]
        )

    def _validate_dimensions(self, width: int, height: int, depth: int, matrix: MatrixType) -> tuple:
        """
        Validates and determines the dimensions of the grid based on input parameters or matrix.

        Args:
            width (int): Specified width of the grid.
            height (int): Specified height of the grid.
            depth (int): Specified depth of the grid.
            matrix (MatrixType): A 3D matrix to infer dimensions from, if provided.

        Returns:
            tuple: The (width, height, depth) of the grid.

        Raises:
            ValueError: If the matrix is not a valid 3D structure or is empty.
        """
        if matrix is not None:
            if not (
                isinstance(matrix, (list, np.ndarray))
                and len(matrix) > 0
                and len(matrix[0]) > 0
                and len(matrix[0][0]) > 0
            ):
                raise ValueError("Provided matrix is not a 3D structure or is empty.")
            return len(matrix), len(matrix[0]), len(matrix[0][0])
        return width, height, depth

    def is_valid_grid(self) -> bool:
        """
        Determine if the grid has valid dimensions.
        """
        return self.width > 0 and self.height > 0 and self.depth > 0

    def node(self, x: int, y: int, z: int) -> Optional[GridNode]:
        """
        Retrieve the node at a specific position.
        """
        return self.nodes[x][y][z] if self.inside(x, y, z) else None

    def inside(self, x: int, y: int, z: int) -> bool:
        """
        Check if a position is within the bounds of the grid.
        """
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def walkable(self, x: int, y: int, z: int) -> bool:
        """
        Check if a node is within bounds and walkable.
        """
        return self.inside(x, y, z) and self.nodes[x][y][z].walkable

    @lru_cache(maxsize=128)
    def _calc_cost(self, x: int, y: int, z: int) -> float:
        """
        Compute the cost (Euclidean distance) between two adjacent nodes.
        """
        return math.sqrt(x * x + y * y + z * z)

    def calc_cost(self, node_a: GridNode, node_b: GridNode, weighted: bool = False) -> float:
        """
        Calculate the movement cost between two nodes.
        """
        x = node_b.x - node_a.x
        y = node_b.y - node_a.y
        z = node_b.z - node_a.z
        ng = self._calc_cost(x, y, z)
        if weighted:
            ng *= node_b.weight
        return ng

    def neighbors(
        self,
        node: GridNode,
        diagonal_movement: int = DMove.NEVER,
    ) -> List[GridNode]:
        """
        Retrieves all valid neighbors of a given node in a 3D grid.

        Directional Movement Flags:
            These variables represent the possible movement states in a 3D grid. They help determine whether 
            specific movements between nodes are allowed based on connectivity, obstacles, or boundaries.

        Naming Convention:
            - **`c`**: Refers to the **current plane** (Z-level of the node being analyzed).
            - **`u`**: Refers to the **upper plane** (Z-level above the current node).
            - **`l`**: Refers to the **lower plane** (Z-level below the current node).
            - **`s`**: Refers to **straight movement** along major axes (X, Y, or Z).
            - **`d`**: Refers to **diagonal movement** (combination of two or more axes, e.g., X+Y, X+Y+Z).
            - **`ut`**: Represents movement directly upward (+Z direction).
            - **`lb`**: Represents movement directly downward (-Z direction).

        Variable Definitions:
            ### Current Plane (Same Z-Level)
            - `cs0`: Straight movement in the negative Y direction.
            - `cs1`: Straight movement in the positive X direction.
            - `cs2`: Straight movement in the positive Y direction.
            - `cs3`: Straight movement in the negative X direction.
            - `cd0`: Diagonal movement in the positive X, negative Y direction.
            - `cd1`: Diagonal movement in the positive X, positive Y direction.
            - `cd2`: Diagonal movement in the negative X, positive Y direction.
            - `cd3`: Diagonal movement in the negative X, negative Y direction.

            ### Upper Plane (Z-Level Above)
            - `us0`: Straight movement in the negative Y direction on the upper plane.
            - `us1`: Straight movement in the positive X direction on the upper plane.
            - `us2`: Straight movement in the positive Y direction on the upper plane.
            - `us3`: Straight movement in the negative X direction on the upper plane.
            - `ud0`: Diagonal movement in the positive X, negative Y, and positive Z direction.
            - `ud1`: Diagonal movement in the positive X, positive Y, and positive Z direction.
            - `ud2`: Diagonal movement in the negative X, positive Y, and positive Z direction.
            - `ud3`: Diagonal movement in the negative X, negative Y, and positive Z direction.
            - `ut`: Movement directly upward in the positive Z direction.

            ### Lower Plane (Z-Level Below)
            - `ls0`: Straight movement in the negative Y direction on the lower plane.
            - `ls1`: Straight movement in the positive X direction on the lower plane.
            - `ls2`: Straight movement in the positive Y direction on the lower plane.
            - `ls3`: Straight movement in the negative X direction on the lower plane.
            - `ld0`: Diagonal movement in the positive X, negative Y, and negative Z direction.
            - `ld1`: Diagonal movement in the positive X, positive Y, and negative Z direction.
            - `ld2`: Diagonal movement in the negative X, positive Y, and negative Z direction.
            - `ld3`: Diagonal movement in the negative X, negative Y, and negative Z direction.
            - `lb`: Movement directly downward in the negative Z direction.

        Usage:
            These flags are set to `True` or `False` during the neighbor computation process, depending on whether 
            the corresponding movement is valid (e.g., not blocked by an obstacle or outside grid boundaries).

            For example:
            - If `cs0 = True`, straight movement in the negative Y direction on the current plane is valid.
            - If `ud1 = False`, diagonal movement in the positive X, positive Y, and positive Z direction on the 
            upper plane is not allowed.

        Args:
            node (GridNode): The node for which neighbors are retrieved.
            diagonal_movement (int, optional): Determines if diagonal movement is allowed. 
                See `DMove` enum for options. Defaults to DMove.NEVER.

        Returns:
            List[GridNode]: A list of all neighboring nodes.
        """

        x, y, z = node.x, node.y, node.z

        neighbors = []
        cs0 = cd0 = cs1 = cd1 = cs2 = cd2 = cs3 = cd3 = False
        us0 = ud0 = us1 = ud1 = us2 = ud2 = us3 = ud3 = ut = False  
        ls0 = ld0 = ls1 = ld1 = ls2 = ld2 = ls3 = ld3 = lb = False

        if self.walkable(x, y - 1, z):
            neighbors.append(self.nodes[x][y - 1][z])
            cs0 = True

        if self.walkable(x + 1, y, z):
            neighbors.append(self.nodes[x + 1][y][z])
            cs1 = True

        if self.walkable(x, y + 1, z):
            neighbors.append(self.nodes[x][y + 1][z])
            cs2 = True

        if self.walkable(x - 1, y, z):
            neighbors.append(self.nodes[x - 1][y][z])
            cs3 = True

        if self.walkable(x, y, z + 1):
            neighbors.append(self.nodes[x][y][z + 1])
            ut = True

        if self.walkable(x, y, z - 1):
            neighbors.append(self.nodes[x][y][z - 1])
            lb = True

        if node.connections:
            neighbors.extend(node.connections)

        if diagonal_movement == DMove.NEVER:
            return neighbors

        elif diagonal_movement == DMove.ALWAYS:
            cd0 = cd1 = cd2 = cd3 = True
            us0 = us1 = us2 = us3 = True
            ls0 = ls1 = ls2 = ls3 = True

        if cd0 and self.walkable(x + 1, y - 1, z):
            neighbors.append(self.nodes[x + 1][y - 1][z])
        else:
            cd0 = False

        if cd1 and self.walkable(x + 1, y + 1, z):
            neighbors.append(self.nodes[x + 1][y + 1][z])
        else:
            cd1 = False

        if cd2 and self.walkable(x - 1, y + 1, z):
            neighbors.append(self.nodes[x - 1][y + 1][z])
        else:
            cd2 = False

        if cd3 and self.walkable(x - 1, y - 1, z):
            neighbors.append(self.nodes[x - 1][y - 1][z])
        else:
            cd3 = False

        if us0 and self.walkable(x, y - 1, z + 1):
            neighbors.append(self.nodes[x][y - 1][z + 1])
        else:
            us0 = False

        if us1 and self.walkable(x + 1, y, z + 1):
            neighbors.append(self.nodes[x + 1][y][z + 1])
        else:
            us1 = False

        if us2 and self.walkable(x, y + 1, z + 1):
            neighbors.append(self.nodes[x][y + 1][z + 1])
        else:
            us2 = False

        if us3 and self.walkable(x - 1, y, z + 1):
            neighbors.append(self.nodes[x - 1][y][z + 1])
        else:
            us3 = False

        if ls0 and self.walkable(x, y - 1, z - 1):
            neighbors.append(self.nodes[x][y - 1][z - 1])
        else:
            ls0 = False

        if ls1 and self.walkable(x + 1, y, z - 1):
            neighbors.append(self.nodes[x + 1][y][z - 1])
        else:
            ls1 = False

        if ls2 and self.walkable(x, y + 1, z - 1):
            neighbors.append(self.nodes[x][y + 1][z - 1])
        else:
            ls2 = False

        if ls3 and self.walkable(x - 1, y, z - 1):
            neighbors.append(self.nodes[x - 1][y][z - 1])
        else:
            ls3 = False

        if diagonal_movement == DMove.ALWAYS:
            ud0 = ud1 = ud2 = ud3 = True
            ld0 = ld1 = ld2 = ld3 = True

        if ud0 and self.walkable(x + 1, y - 1, z + 1):
            neighbors.append(self.nodes[x + 1][y - 1][z + 1])

        if ud1 and self.walkable(x + 1, y + 1, z + 1):
            neighbors.append(self.nodes[x + 1][y + 1][z + 1])

        if ud2 and self.walkable(x - 1, y + 1, z + 1):
            neighbors.append(self.nodes[x - 1][y + 1][z + 1])

        if ud3 and self.walkable(x - 1, y - 1, z + 1):
            neighbors.append(self.nodes[x - 1][y - 1][z + 1])

        if ld0 and self.walkable(x + 1, y - 1, z - 1):
            neighbors.append(self.nodes[x + 1][y - 1][z - 1])

        if ld1 and self.walkable(x + 1, y + 1, z - 1):
            neighbors.append(self.nodes[x + 1][y + 1][z - 1])

        if ld2 and self.walkable(x - 1, y + 1, z - 1):
            neighbors.append(self.nodes[x - 1][y + 1][z - 1])

        if ld3 and self.walkable(x - 1, y - 1, z - 1):
            neighbors.append(self.nodes[x - 1][y - 1][z - 1])

        return neighbors

    def cleanup(self):
        """
        Reset the state of all nodes in the grid.
        Useful for clearing pathfinding metadata.
        """
        for x_nodes in self.nodes:
            for y_nodes in x_nodes:
                for z_node in y_nodes:
                    z_node.cleanup()

    def visualize(
        self,
        path: Optional[List[Union[GridNode, Tuple]]] = None,
        start: Optional[Union[GridNode, Tuple]] = None,
        end: Optional[Union[GridNode, Tuple]] = None,
        visualize_weight: bool = True,
        save_html: bool = False,
        save_to: str = "./outputs/mplanner.html",
        always_show: bool = False,
    ):
        """
        Visualizes the grid and an optional path using Plotly in 3D.

        Args:
            path (list of Union[GridNode, Tuple], optional): The path to visualize. 
                Can include nodes or coordinate tuples.
            start (Union[GridNode, Tuple], optional): The start position or node. 
                Defaults to the first node in the path if omitted.
            end (Union[GridNode, Tuple], optional): The end position or node. 
                Defaults to the last node in the path if omitted.
            visualize_weight (bool, optional): Whether to include node weights in the visualization. 
                Defaults to True.
            save_html (bool, optional): If True, saves the visualization as an HTML file. 
                Defaults to False.
            save_to (str, optional): Path to save the HTML file, if `save_html` is True. 
                Defaults to "./outputs/mplanner.html".
            always_show (bool, optional): If True, always displays the visualization in a browser 
                even if saved to HTML. Defaults to False.

        Notes:
            - Requires Plotly for visualization. Ensure the `plotly` library is installed.
        """

        if not USE_PLOTLY:
            warnings.warn("Plotly is not installed. Please install it to use this feature.")
            return

        # Extract obstacle and weight information directly from the grid
        X, Y, Z, obstacle_values, weight_values = [], [], [], [], []
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    node = self.node(x, y, z)
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    obstacle_values.append(0 if node.walkable else 1)
                    weight_values.append(node.weight if node.walkable else 0)

        # Create obstacle volume visualization
        obstacle_vol = go.Volume(
            x=np.array(X),
            y=np.array(Y),
            z=np.array(Z),
            value=np.array(obstacle_values),
            isomin=0.1,
            isomax=1.0,
            opacity=0.1,
            surface_count=25,  # Increase for better visibility
            colorscale="Greys",
            showscale=False,
            name="Obstacles",
        )

        # List of items to visualize
        visualizations = [obstacle_vol]

        # Create weight volume visualization
        if visualize_weight:
            weight_vol = go.Volume(
                x=np.array(X),
                y=np.array(Y),
                z=np.array(Z),
                value=np.array(weight_values),
                isomin=1.01,  # Assuming default weight is 1, adjust as needed
                isomax=max(weight_values) * 1.01,
                opacity=0.5,  # Adjust for better visibility
                surface_count=25,
                colorscale="Viridis",  # A different colorscale for distinction
                showscale=True,
                colorbar=dict(title="Weight", ticks="outside"),
            )
            visualizations.append(weight_vol)

        # Add path visualization if path is provided
        if path:
            # Convert path to coordinate tuples
            path = [p.identifier if isinstance(p, GridNode) else p for p in path]

            # Create path visualization
            path_x, path_y, path_z = zip(*path)
            path_trace = go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode="markers+lines",
                marker=dict(size=6, color="red", opacity=0.9),
                line=dict(color="red", width=3),
                name="Path",
                hovertext=[f"Step {i}: ({x}, {y}, {z})" for i, (x, y, z) in enumerate(path)],
                hoverinfo="text",
            )
            visualizations.append(path_trace)

            # Set start and end nodes if not provided
            start = start or path[0]
            end = end or path[-1]

        # Add start and end node visualizations if available
        if start:
            start = start.identifier if isinstance(start, GridNode) else start
            start_trace = go.Scatter3d(
                x=[start[0]],
                y=[start[1]],
                z=[start[2]],
                mode="markers",
                marker=dict(size=8, color="green", symbol="diamond"),
                name="Start",
                hovertext=f"Start: {start}",
                hoverinfo="text",
            )
            visualizations.append(start_trace)

        if end:
            end = end.identifier if isinstance(end, GridNode) else end
            end_trace = go.Scatter3d(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                mode="markers",
                marker=dict(size=8, color="blue", symbol="diamond"),
                name="End",
                hovertext=f"End: {end}",
                hoverinfo="text",
            )
            visualizations.append(end_trace)

        # Camera settings
        # Set camera perpendicular to the z-axis
        camera = dict(eye=dict(x=0.0, y=0.0, z=self.depth / 4))

        # Specify layout
        layout = go.Layout(
            title="Motion Planning Visualization",
            scene=dict(
                xaxis=dict(title="X-axis", showbackground=True),
                yaxis=dict(title="Y-axis", showbackground=True),
                zaxis=dict(title="Z-axis", showbackground=True),
                aspectmode="auto",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            autosize=True,
            scene_camera=camera,
        )

        # Create figure
        fig = go.Figure(data=visualizations, layout=layout)

        # Save visualization to HTML file if specified
        if save_html:
            fig.write_html(save_to, auto_open=False)
            print(f"Visualization saved to: {save_to}")

        if always_show or not save_html:
            fig.show()

class World:
    """
    Represents a world consisting of multiple interconnected grids.

    The world manages navigation and cost calculations across these grids.

    Args:
        grids (Dict[int, Grid]): A dictionary of grids in the world, where the key 
            is the grid ID and the value is the corresponding `Grid` object.
    """
    def __init__(self, grids: Dict[int, Grid]):
        self.grids = grids

    def neighbors(self, node: GridNode, diagonal_movement: int) -> List[GridNode]:
        """
        Retrieves the neighbors of a given node, accounting for its grid.

        Args:
            node (GridNode): The node for which neighbors are retrieved.
            diagonal_movement (int): Specifies if diagonal movement is allowed. 
                Refer to the `DMove` enum for valid options (e.g., never, always).

        Returns:
            List[GridNode]: A list of neighboring nodes for the given node, including potential 
                connections to other grids if supported.
        """
        return self.grids[node.grid_id].neighbors(node, diagonal_movement=diagonal_movement)

    def calc_cost(self, node_a: GridNode, node_b: GridNode, weighted: bool = False) -> float:
        """
        Calculates the movement cost between two nodes, potentially across grids.

        Args:
            node_a (GridNode): The starting node.
            node_b (GridNode): The target node.
            weighted (bool, optional): Whether to factor in node weights for the cost calculation. 
                Defaults to False.

        Returns:
            float: The calculated movement cost between `node_a` and `node_b`.

        Notes:
            - Currently, the method only considers nodes within the same grid. 
            - For nodes in different grids, inter-grid cost calculation is a placeholder.
        """
        # TODO: Handle inter-grid cost calculation when `node_a.grid_id != node_b.grid_id`.
        # For now, only intra-grid costs are considered.
        return self.grids[node_a.grid_id].calc_cost(node_a, node_b, weighted=weighted)

    def cleanup(self):
        """
        Reset all grids in the world by clearing any pathfinding metadata or temporary states.
        """
        for grid in self.grids.values():
            grid.cleanup()


def build_nodes(
    width: int,
    height: int,
    depth: int,
    matrix: MatrixType = None,
    inverse: bool = False,
    grid_id: Optional[int] = None,
) -> List[List[List[GridNode]]]:
    """
    Creates a 3D grid of `GridNode` objects based on specified dimensions and matrix.

    If a matrix is provided, it determines the walkability and weight of the nodes.
    Otherwise, all nodes are initialized as walkable with a default weight of 1.

    Args:
        width (int): The number of nodes along the x-axis.
        height (int): The number of nodes along the y-axis.
        depth (int): The number of nodes along the z-axis.
        matrix (optional, MatrixType): A 3D matrix (list of lists of lists or numpy array) specifying walkability 
            and weights:
            - `0` or values <= 0 indicate non-walkable nodes (obstacles).
            - Positive values indicate walkable nodes, with the value representing the weight.
            If `inverse=True`, the interpretation of values is reversed.
        inverse (bool, optional): If True, reverses the walkability condition:
            - Non-zero or positive values indicate obstacles.
            - `0` or negative values indicate walkable nodes.
            Defaults to False.
        grid_id (int, optional): An identifier for the grid.

    Returns:
        List[List[List[GridNode]]]: A 3D list of `GridNode` objects representing the grid.

    Examples:
        >>> build_nodes(2, 2, 2, matrix=[[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
        [[[GridNode(x=0, y=0, z=0, walkable=True, weight=1), GridNode(x=0, y=0, z=1, walkable=False, weight=0)], 
          [GridNode(x=0, y=1, z=0, walkable=False, weight=0), GridNode(x=0, y=1, z=1, walkable=True, weight=1)]],
         [[GridNode(x=1, y=0, z=0, walkable=True, weight=1), GridNode(x=1, y=0, z=1, walkable=True, weight=1)], 
          [GridNode(x=1, y=1, z=0, walkable=False, weight=0), GridNode(x=1, y=1, z=1, walkable=False, weight=0)]]]

    Notes:
        - If no `matrix` is provided, all nodes will default to walkable with weight = 1.
        - The `grid_id` allows differentiation of nodes across multiple grids.
    """
    nodes: List[List[List[GridNode]]] = []
    use_matrix = matrix is not None

    for x in range(width):
        nodes.append([])
        for y in range(height):
            nodes[x].append([])
            for z in range(depth):
                # Determine weight and walkability based on the matrix or defaults
                weight = int(matrix[x][y][z]) if use_matrix else 1
                walkable = weight <= 0 if inverse else weight >= 1

                # Append a new GridNode to the grid
                nodes[x][y].append(GridNode(x=x, y=y, z=z, walkable=walkable, weight=weight, grid_id=grid_id))
    
    return nodes

