import numpy as np
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, points, n_ants, n_iterations, decay=0.1, alpha=1, beta=1):
        self.points = points
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.n_cities = len(points)
        self.pheromone = np.ones((self.n_cities, self.n_cities))

    def plot_path(self, path):
        plt.figure(figsize=(10, 10))
        x_coords = [self.points[i][1][0] for i in path]
        y_coords = [self.points[i][1][1] for i in path]

        # Plot cities with labels
        plt.scatter(x_coords, y_coords, color='red')
        for i, city in enumerate(path):
            if i == 0:
                plt.text(x=self.points[city][1][0], y=self.points[city][1][1], 
                        s="Start  ", fontsize=8, ha='right')
            elif i == len(path) - 1:
                plt.text(x=self.points[city][1][0], y=self.points[city][1][1], 
                        s="  Koniec", fontsize=8, ha='left')            
            else:
                plt.text(x=self.points[city][1][0], y=self.points[city][1][1], 
                        s=str(i) + "-" + self.points[city][0], fontsize=10, ha='right')

        # Plot path
        plt.plot(x_coords, y_coords, color='blue')
        
        plt.title("Best Path")
        plt.show()

    def run(self):
        shortest_distance = np.inf
        shortest_path = None
        best_iteration = None
        for iteration in range(self.n_iterations):
            all_paths = []
            ant_pheromones = []
            ant_probabilities = []
            for ant in range(self.n_ants):
                path, distance, pheromones, probabilities = self.generate_path(iteration, ant)
                all_paths.append((path, distance))
                ant_pheromones.append(pheromones)
                ant_probabilities.append(probabilities)
            self.update_pheromone(all_paths)
            best_path, best_distance = min(all_paths, key=lambda x: x[1])
            if best_distance < shortest_distance:
                shortest_distance = best_distance
                shortest_path = best_path
                best_pheromones = ant_pheromones[all_paths.index((best_path, best_distance))]
                best_probabilities = ant_probabilities[all_paths.index((best_path, best_distance))]
                best_iteration = iteration + 1
                
        return best_iteration, shortest_path, shortest_distance, best_pheromones, best_probabilities

    def generate_path(self, iteration, ant):
        start_index = self.points.index(('g', (4, 2)))
        path = [start_index]
        distance = 0
        pheromones = []
        probabilities = []
        visited = set([start_index])
        while len(path) < self.n_cities:
            current_city = path[-1]
            next_city, pheromone, probability = self.choose_next_city(path, visited)
            path.append(next_city)
            distance += self.calculate_distance([path[-2], path[-1]])
            pheromones.append(pheromone)
            probabilities.append(probability)
            visited.add(next_city)
            print(f"Iteration {iteration + 1}, Ant {ant + 1}, Current City: {self.points[current_city][0]}, Next City: {self.points[next_city][0]}, Pheromone: {pheromone}, Probability: {probability}")
        path.append(start_index)
        distance += self.calculate_distance([path[-2], path[-1]])
        return path, distance, pheromones, probabilities

    def choose_next_city(self, path, visited):
        pheromone_values = self.pheromone[path[-1]] ** self.alpha
        visibility_values = self.calculate_visibility(path[-1], visited)
        probabilities = pheromone_values * visibility_values
        for i in path:
            probabilities[i] = 0
        probabilities /= probabilities.sum()
        chosen_city = np.random.choice(range(self.n_cities), p=probabilities)
        return chosen_city, pheromone_values[chosen_city], probabilities[chosen_city]


    def calculate_visibility(self, current_city, visited):
        visibility = []
        for i in range(self.n_cities):
            if i not in visited:
                distance = np.linalg.norm(np.array(self.points[current_city][1]) - np.array(self.points[i][1]))
                visibility.append(1 / distance)
            else:
                visibility.append(0)
        return visibility

    def calculate_distance(self, path):
        return np.linalg.norm(np.array(self.points[path[0]][1]) - np.array(self.points[path[1]][1]))

    def update_pheromone(self, all_paths):
        evaporation = 1 - self.decay
        self.pheromone *= evaporation
        best_path, best_distance = min(all_paths, key=lambda x: x[1])
        
        for i in range(len(best_path) - 1):   
            city_from = best_path[i]
            city_to = best_path[i + 1]
            self.pheromone[city_from, city_to] += 1.0 / best_distance
            self.pheromone[city_to, city_from] += 1.0 / best_distance  # Update in both directions        

        print("Updated pheromones:")
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j and self.pheromone[i, j] != 0:  # Filter out self-transitions
                    print(f"From {self.points[i][0]} to {self.points[j][0]}: {self.pheromone[i, j]}")


    def path_to_names(self, path):
        return [self.points[i][0] for i in path]
    
    def print_summary(self, best_iteration, path, total_distance, pheromones, probabilities):
        print("\nNajlepsza iteracja:", best_iteration)
        
        print("\nPunkty na mapie:")
        for point in self.points:
            print(point[1])

        print(f"\nPunkt startowy:\n{self.points[path[0]][1]}")

        print("\nKolejność odwiedzonych punktów:")
        for i in path:
            print(self.points[i][1])
            
        print(f"\nKoszt przebycia całej drogi: \n{total_distance}")

        print("\nKoszty przejścia między punktami:")
        for i in range(len(path)-1):
            dist = self.calculate_distance([path[i], path[i+1]])
            print(f"Między punktami {self.points[path[i]][1]} i {self.points[path[i+1]][1]}: {dist}")

        print("\nFeromony w najlepszej iteracji:")
        for i in range(len(pheromones)):
            print(f"Feromon na trasie {self.points[path[i]][1]} i {self.points[path[i+1]][1]}: {pheromones[i]}")

        print("\nPrawdopodobieństwo pójścia do danego punktu:")
        for i in range(len(probabilities)):
            print(f"Prawdopodobieństwo pójścia z {self.points[path[i]][1]} do {self.points[path[i+1]][1]}: {probabilities[i]}")


points = [('a', (1, 1)), ('b', (5, 8)), ('c', (7, 12)), ('d', (2, 9)), ('e', (7, 2)), ('f', (1, 12)), ('g', (4, 2))]

n_ants = 7
n_iterations = 15

ant_colony = AntColony(points, n_ants, n_iterations)
best_iteration, shortest_path, shortest_distance, best_pheromones, best_probabilities = ant_colony.run()

ant_colony.print_summary(best_iteration, shortest_path, shortest_distance, best_pheromones, best_probabilities)

ant_colony.plot_path(shortest_path)