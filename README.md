# MPlanner

*MPlanner* is a compact, concise, and customizable **3D motion planning library**. It is designed to simplify motion planning for everyone—whether you're a researcher, engineer, or enthusiast.

---

## Features

**Customizable 3D Motion Planning**:

  - Supports a variety of heuristics for pathfinding (e.g., Euclidean, Manhattan).
  - Handles multi-grid environments for complex navigation scenarios.
  
**Optimized Data Structures**:

  - Priority queue for efficient pathfinding.
  - Flexible grid and node-based representation.

**Extensible Framework**:

  - Implement your algorithms by extending the base `planner` class.
  - Suitable for algorithms like A*, Dijkstra, and custom planners.

**Pythonic and User-Friendly**:

  - Designed for developers and learners alike.
  - Integrates seamlessly with Python 3.10+.

---

## Installation

Install *MPlanner* using [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/AnugunjNaman/mplanner.git
cd mplanner
poetry install
```

---

## Quickstart

Head over to the [Example](examples/astar.ipynb) guide for step-by-step example instructions on using *MPlanner*.

---

## Contributing

We welcome contributions from the community!  

- Please use [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow):  
  - Create a branch,  
  - Add your commits,  
  - Open a pull request.

For more details, check out our [Contributing Guide](contributing.md). It includes information on our **Code of Conduct** and the process for submitting pull requests.

## License

*MPlanner* is open source and available under the **MIT License**.  
See the [LICENSE](https://github.com/AnugunjNaman/mplanner/blob/main/LICENSE) file for details.
