import pygame
import numpy as np
import heapq
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
purpel = (221,160,221)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
lightc=(224,255,255)
darkc=(0,139,139)
bludark=(95,158,160)
lightblue=(72,209,204)
slategray=(112,128,144)
wheat=(245,222,179)

class WarehouseOptimizer:
    def __init__(self, width=15, height=15, cell_size=40):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = np.zeros((height, width))
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 50
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Warehouse Optimizer")

        self.start = (1, 1)
        self.end = (height - 2, width - 2)
        self.storage_points = []
        self.paths = []
        self.heatmap = np.zeros((height, width))

        self.metrics = {'path_lengths': [], 'timestamps': [], 'algorithm': []}
        self.storage_usage = {}
        self.historical_data = []
        self.storage_times = []
        self.comparison_data = []

        self.algorithm = 'A*'
        self.analysis_button = None
        self.compare_button = None

    def draw_grid(self):
        self.screen.fill(lightc)
        max_heat = np.max(self.heatmap) if np.max(self.heatmap) > 0 else 1
        for y in range(self.height):
            for x in range(self.width):
                if self.heatmap[y, x] > 0:
                    intensity = int(255 * (self.heatmap[y, x] / max_heat))
                    color = (255, 255 - intensity, 255 - intensity)
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    surface = pygame.Surface((self.cell_size, self.cell_size))
                    surface.set_alpha(128)
                    surface.fill(color)
                    self.screen.blit(surface, rect)
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, darkc, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height - 50, self.cell_size):
            pygame.draw.line(self.screen, darkc, (0, y), (self.screen_width, y))
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, bludark, rect)

        threshold = np.percentile(self.heatmap[self.heatmap > 0], 90) if np.any(self.heatmap > 0) else 0
        bottlenecks = np.where(self.heatmap > threshold)
        for y, x in zip(*bottlenecks):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, YELLOW, rect, 3)

        pygame.draw.circle(self.screen, lightblue, (self.start[1] * self.cell_size + self.cell_size // 2,
                                                self.start[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3)
        pygame.draw.circle(self.screen, slategray, (self.end[1] * self.cell_size + self.cell_size // 2,
                                              self.end[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3)

        for point in self.storage_points:
            pos = (point[1] * self.cell_size + self.cell_size // 2, point[0] * self.cell_size + self.cell_size // 2)
            usage = self.storage_usage.get(point, 0)
            size = int(self.cell_size // 3 * (1 + usage / 10))
            pygame.draw.circle(self.screen, purpel, pos, size)

        for path in self.paths:
            if len(path) > 1:
                points = [(p[1] * self.cell_size + self.cell_size // 2, p[0] * self.cell_size + self.cell_size // 2) for p in path]
                pygame.draw.lines(self.screen, BLACK, False, points, 2)

        font = pygame.font.SysFont(None, 25)
        algo_text = font.render(f"Algorithm: {self.algorithm}", True, BLACK)
        self.screen.blit(algo_text, (10, self.screen_height - 40))
    
        button_rect = pygame.Rect(self.screen_width - 320, self.screen_height - 40, 150, 30)
        pygame.draw.rect(self.screen, purpel, button_rect)
        pygame.draw.rect(self.screen, BLACK, button_rect, 2)
        text = font.render("Show Analysis", True, BLACK)
        self.screen.blit(text, (self.screen_width - 310, self.screen_height - 35))
        self.analysis_button = button_rect
        
        # Compare button
        compare_rect = pygame.Rect(self.screen_width - 160, self.screen_height - 40, 150, 30)
        pygame.draw.rect(self.screen, purpel, compare_rect)
        pygame.draw.rect(self.screen, BLACK, compare_rect, 2)
        text = font.render("Compare Algos", True, BLACK)
        self.screen.blit(text, (self.screen_width - 150, self.screen_height - 35))
        self.compare_button = compare_rect

        pygame.display.flip()

    def get_cell_from_pos(self, pos):
        x, y = pos
        return (y // self.cell_size, x // self.cell_size)

    def add_storage(self, pos):
        y, x = pos
        if 0 <= y < self.height and 0 <= x < self.width:
            if (y, x) not in self.storage_points:
                self.storage_points.append((y, x))
                self.storage_usage[(y, x)] = 1
            else:
                self.storage_usage[(y, x)] += 1

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        y, x = pos
        neighbors = []
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != 1:
                neighbors.append((ny, nx))
        return neighbors

    def a_star(self, start, goal):
        start_time = datetime.now()
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
                    self.draw_grid()
                    pygame.time.delay(10)

        if goal not in came_from:
            return None, 0
        path, current = [], goal
        while current:
            path.append(current)
            current = came_from[current]
        elapsed = (datetime.now() - start_time).total_seconds()
        return path[::-1], elapsed

    def greedy_search(self, start, goal):
        start_time = datetime.now()
        frontier = [(self.heuristic(start, goal), start)]
        came_from = {start: None}
        visited = set()
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            if current in visited:
                continue
            visited.add(current)
            
            for next_pos in self.get_neighbors(current):
                if next_pos not in came_from:
                    priority = self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
                    self.draw_grid()
                    pygame.time.delay(10)

        if goal not in came_from:
            return None, 0
        path, current = [], goal
        while current:
            path.append(current)
            current = came_from[current]
        elapsed = (datetime.now() - start_time).total_seconds()
        return path[::-1], elapsed

    def optimize(self):
        self.paths = []
        self.heatmap = np.zeros((self.height, self.width))
        targets = self.storage_points if self.storage_points else [self.end]
        
        for target in targets:
            if self.algorithm == 'A*':
                path, elapsed = self.a_star(self.start, target)
            else: 
                path, elapsed = self.greedy_search(self.start, target)
                
            if path:
                self.paths.append(path)
                self.storage_times.append({'target': target, 'time': elapsed, 'algorithm': self.algorithm})
                for y, x in path:
                    self.heatmap[y, x] += 1
        self.analyze_results()

    def analyze_results(self):
        if not self.paths:
            return
        timestamp = datetime.now()
        total_distance = sum(len(p) for p in self.paths)
        avg_distance = total_distance / len(self.paths)

        reachable_storage = 0
        unreachable_storage = 0
        for storage_point in self.storage_points:
            reachable = False
            for path in self.paths:
                if storage_point in path:
                    reachable = True
                    break
            if reachable:
                reachable_storage += 1
            else:
                unreachable_storage += 1

        self.metrics['path_lengths'].append(avg_distance)
        self.metrics['timestamps'].append(timestamp)
        self.metrics['algorithm'].append(self.algorithm)
        
        self.historical_data.append({
            'timestamp': timestamp,
            'total_distance': total_distance,
            'avg_distance': avg_distance,
            'reachable_storage': reachable_storage,
            'unreachable_storage': unreachable_storage,
            'algorithm': self.algorithm
        })

    def generate_report(self):
        if not self.historical_data:
            return
            
        current_algo_data = [d for d in self.historical_data if d['algorithm'] == self.algorithm]
        if not current_algo_data:
            return
            
        df = pd.DataFrame(current_algo_data)
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"Algorithm Analysis: {self.algorithm}", fontsize=16)

        plt.subplot(3, 2, 1)
        reachable_storage = df['reachable_storage'].sum()
        unreachable_storage = df['unreachable_storage'].sum()
        plt.pie([reachable_storage, unreachable_storage], 
                labels=['Reachable Storage', 'Unreachable Storage'], 
                autopct='%1.1f%%', colors=['lightblue','cyan'])
        plt.title('Storage Accessibility')

        plt.subplot(3, 2, 2)
        plt.hist(df['avg_distance'], bins=10, color='lightblue')
        plt.title('Avg Path Length Distribution')
        

        current_algo_times = [s for s in self.storage_times if s['algorithm'] == self.algorithm]
        
        plt.subplot(3, 2, 3)
        plt.bar([str(s['target']) for s in current_algo_times], 
                [s['time'] for s in current_algo_times], color='lightblue')
        plt.title("Storage Access Times")
        plt.ylabel("Time (s)")
        plt.xlabel("Storage Point")
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 4)
        if self.storage_usage:
            usage_values = list(self.storage_usage.values())
            plt.pie(usage_values, labels=[f'Storage {i+1}' for i in range(len(usage_values))],
                    colors=plt.cm.Pastel1.colors)
            plt.title('Storage Usage')
            
        plt.subplot(3, 2, 5)
        df_time = pd.DataFrame(current_algo_times)
        if not df_time.empty:
            df_time['distance'] = df_time['target'].apply(lambda t: abs(t[0] - self.start[0]) + abs(t[1] - self.start[1]))
            plt.scatter(df_time['distance'], df_time['time'], color='lightblue')
            plt.title("Access Time vs Distance")
            plt.xlabel("Manhattan Distance from Start")
            plt.ylabel("Time (s)")
            

        plt.tight_layout()
        plt.show()
        
    def compare_algorithms(self):
        a_star_data = [d for d in self.historical_data if d['algorithm'] == 'A*']
        greedy_data = [d for d in self.historical_data if d['algorithm'] == 'Greedy']
        
        if not a_star_data or not greedy_data:
            print("Need data for both algorithms to compare")
            return
    
        a_star_times = [s['time'] for s in self.storage_times if s['algorithm'] == 'A*']
        greedy_times = [s['time'] for s in self.storage_times if s['algorithm'] == 'Greedy']
    
        a_star_total_time = sum(a_star_times)
        greedy_total_time = sum(greedy_times)
        a_star_avg_time = a_star_total_time / len(a_star_times) if a_star_times else 0
        greedy_avg_time = greedy_total_time / len(greedy_times) if greedy_times else 0
        a_star_total_distance = sum(d['total_distance'] for d in a_star_data)
        greedy_total_distance = sum(d['total_distance'] for d in greedy_data)
        a_star_avg_distance = sum(d['avg_distance'] for d in a_star_data) / len(a_star_data)
        greedy_avg_distance = sum(d['avg_distance'] for d in greedy_data) / len(greedy_data)
    
        plt.figure(figsize=(12, 10))
        plt.suptitle("Algorithm Comparison: A* vs Greedy", fontsize=18)
    
        plt.subplot(2, 2, 1)
        plt.bar(['A*', 'Greedy'], [a_star_total_time, greedy_total_time], color=['lightblue', 'cyan'])
        plt.title('Total Execution Time')
        plt.ylabel('Time (s)')
        plt.subplot(2, 2, 2)
        plt.bar(['A*', 'Greedy'], [a_star_avg_time, greedy_avg_time], color=['lightblue', 'cyan'])
        plt.title('Average Execution Time')
        plt.ylabel('Time (s)')
        plt.subplot(2, 2, 3)
        plt.bar(['A*', 'Greedy'], [a_star_total_distance, greedy_total_distance], color=['lightblue', 'cyan'])
        plt.title('Total Path Distance')
        plt.ylabel('Cells')
        plt.subplot(2, 2, 4)
        plt.bar(['A*', 'Greedy'], [a_star_avg_distance, greedy_avg_distance], color=['lightblue', 'cyan'])
        plt.title('Average Path Distance')
        plt.ylabel('Cells')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def main():
    warehouse = WarehouseOptimizer()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                cell = warehouse.get_cell_from_pos(event.pos)
                if event.button == 1:
                    if hasattr(warehouse, 'analysis_button') and warehouse.analysis_button.collidepoint(event.pos):
                        warehouse.generate_report()
                    elif hasattr(warehouse, 'compare_button') and warehouse.compare_button.collidepoint(event.pos):
                        warehouse.compare_algorithms()
                    else:
                        warehouse.grid[cell] = 0 if warehouse.grid[cell] == 1 else 1
                elif event.button == 3:
                    warehouse.add_storage(cell)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    warehouse.optimize()
                elif event.key == pygame.K_r:
                    warehouse = WarehouseOptimizer()
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    warehouse.algorithm = 'A*'
                elif event.key == pygame.K_g:
                    warehouse.algorithm = 'Greedy'
        warehouse.draw_grid()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()