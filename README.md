Warehouse Optimizer
A visual simulation and analysis tool built with Pygame that helps optimize warehouse storage access using two pathfinding algorithms: A* and Greedy Search. The tool allows users to define obstacles and storage points on a grid, simulate paths, and analyze performance with visual charts.


Features

🧠 Algorithms: Supports A* and Greedy Search for pathfinding.

📊 Analysis Tools: Generates detailed reports with histograms, pie charts, and scatter plots.

🔁 Live Comparison: Compares performance between algorithms in real time.

🔥 Heatmap Visualization: Highlights frequently used paths and bottlenecks.

🖱️ Interactive GUI: Click to add/remove obstacles or storage points.

🗂️ Storage Usage Tracking: Visualizes how often each storage point is accessed.


How to Use

Controls
Left Click:

On grid → Toggle obstacle.

On "Show Analysis" button → Show charts and stats.

On "Compare Algos" button → Compare A* and Greedy.

Right Click: Add a storage point.

Keyboard:

Space → Run the selected algorithm (A* or Greedy).

A → Switch to A*.

G → Switch to Greedy.

R → Reset the grid.

ESC → Exit the program.


Visuals & Analysis

🔷 Start/End Points: Blue and gray circles.

🔲 Obstacles: Dark blue blocks.

🔶 Storage Points: Purple circles (size increases with usage).

🌡️ Heatmap: Redder areas = more path usage.

🚧 Bottlenecks: Highlighted in yellow.

The analysis includes:

Storage accessibility pie chart.

Average path length distribution.

Access time per storage point.

Storage usage frequency.

Correlation between distance and access time.

A full algorithm comparison dashboard.


Gui With A*:

![image](https://github.com/user-attachments/assets/fc1ebf6c-15d5-451e-90bf-1bfef3ad4917)

Analysis With A*:

![image](https://github.com/user-attachments/assets/06d7833f-f609-4cc1-a6f5-44ce492afa53)


Gui With Greedy:

![image](https://github.com/user-attachments/assets/dff6eb6e-fec6-43de-94cb-c4985e033fe1)

Analysis With Greedy:

![image](https://github.com/user-attachments/assets/912f005b-d072-4dd9-93da-7e2d80c2faaf)


Comparison between algorithms:

![image](https://github.com/user-attachments/assets/b181a696-0658-44d1-9144-f3c89f5a0761)

