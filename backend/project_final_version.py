import pandas as pd
import time
import heapq
from collections import Counter
from constraint import Problem
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from constraint import Problem

# Load dataset
try:
    df = pd.read_csv("Crop_recommendationV2.csv")
except FileNotFoundError:
    raise FileNotFoundError("Dataset 'Crop_recommendationV2.csv' not found. Please ensure the file exists in the correct directory.")

# Verify required columns
required_columns = ['label', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'growth_stage', 'soil_type', 'water_source_type',
                    'fertilizer_usage', 'co2_concentration', 'irrigation_frequency', 'pest_pressure', 'soil_moisture', 'organic_matter',
                    'sunlight_exposure', 'wind_speed', 'urban_area_proximity']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Configuration
interval_sizes = {
    'N': 10, 'P': 10, 'K': 10, 'temperature': 1, 'humidity': 2, 'ph': 0.2, 'rainfall': 20,
    'soil_moisture': 5, 'organic_matter': 1, 'sunlight_exposure': 1, 'wind_speed': 5, 'urban_area_proximity': 5
}
gift_weights = {
    'fertilizer_usage': 0.2319, 'co2_concentration': 0.2029, 'irrigation_frequency': 0.1594,
    'pest_pressure': 0.1449, 'water_usage_efficiency': 0.2609
}
penalty_weights = {
    'N': 0.18, 'P': 0.14, 'K': 0.13, 'rainfall': 0.15, 'temperature': 0.16, 'ph': 0.12, 'humidity': 0.12,
    'soil_moisture': 0.1, 'organic_matter': 0.1, 'sunlight_exposure': 0.1, 'wind_speed': 0.05, 'urban_area_proximity': 0.05
}
feature_order = ['N', 'P', 'K', 'humidity', 'temperature', 'rainfall', 'ph', 'soil_moisture', 'organic_matter', 'sunlight_exposure', 'wind_speed', 'urban_area_proximity']
numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in ['soil_type', 'water_source_type']]
feature_ranges = {feature: (df[feature].min(), df[feature].max()) for feature in numerical_features}

# Helper Functions
def compute_gift(gift_values, user_inputs, gift_ranges):
    gift_score = 0
    def get_interval_gift(diff, weight):
        return 0.02 * weight if diff > 0.6 else 0.01 * weight if diff > 0.3 else 0.005 * weight if diff > 0.1 else 0
    def compute_feature_gift(user_val, optimal_val, range_min, range_max, weight):
        if range_max == range_min: return 0
        diff = abs(user_val - optimal_val) / (range_max - range_min)
        return get_interval_gift(diff, weight)
    def compute_water_efficiency_gift(user_inputs, eff_min, eff_max, weight):
        rainfall = user_inputs.get('rainfall', 1)
        irrigation = user_inputs.get('irrigation_frequency', 1)
        source = user_inputs.get('water_source_type', 1)
        if irrigation == 0: irrigation = 1
        source_score = 1.0 if source == 1 else 1.5 if source == 2 else 2.0
        efficiency = rainfall / (irrigation * source_score)
        if eff_max == eff_min: return 0
        diff = abs(eff_max - efficiency) / (eff_max - eff_min)
        return get_interval_gift(diff, weight)
    for feature, weight in gift_weights.items():
        user_val = user_inputs.get(feature, 0)
        range_min, range_max = gift_ranges.get(feature, (0, 1))
        if feature == 'water_usage_efficiency':
            gift_score += compute_water_efficiency_gift(user_inputs, range_min, range_max, weight)
        else:
            gift_score += compute_feature_gift(user_val, range_min, range_min, range_max, weight)
    return gift_score

def compute_heuristic(gift_score, penalty):
    return penalty + gift_score

def normalize_values(values, feature_ranges, features_to_normalize):
    normalized = {}
    for k, v in values.items():
        if k not in features_to_normalize:
            continue
        min_val, max_val = feature_ranges[k]
        if max_val == min_val:
            normalized[k] = 0
        else:
            normalized[k] = (v - min_val) / (max_val - min_val)
    return normalized

def compute_penalty_by_depth(user_inputs, node_avg_values, feature_ranges, current_depth, feature_order, penalty_weights):
    penalty = 0
    active_features = feature_order[:current_depth] if current_depth > 0 else feature_order
    normalized_user = normalize_values(user_inputs, feature_ranges, active_features)
    normalized_node = normalize_values(node_avg_values, feature_ranges, active_features)
    for feature in active_features:
        user_val = normalized_user.get(feature, 0)
        node_val = normalized_node.get(feature, 0)
        weight = penalty_weights.get(feature, 1)
        diff = abs(user_val - node_val)
        penalty += (1.0 if diff >= 0.6 else 0.8 if diff >= 0.35 else 0.6 if diff >= 0.2 else 0.4 if diff >= 0.1 else 0.2 if diff >= 0.05 else 0.1) * weight
    return penalty

def filter_rows_by_path(df, path):
    filtered_df = df.copy()
    for feature, (min_val, max_val) in path:
        filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]
    return filtered_df

# Greedy Search
class CropNode:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1
    def __hash__(self): return hash(tuple(self.state['path']))
    def __eq__(self, other): return isinstance(other, CropNode) and self.state['path'] == other.state['path']
    def __lt__(self, other): return self.state['heuristic_score'] < other.state['heuristic_score']

class CropPredictionProblemSearch:
    def __init__(self, initial_state, user_inputs, feature_order, df):
        self.initial_state = initial_state
        self.user_inputs = user_inputs
        self.feature_order = feature_order
        self.df = df
    def is_goal(self, state): return state['depth'] == len(self.feature_order)
    def get_valid_actions(self, state):
        if state['depth'] >= len(self.feature_order): return []
        feature = self.feature_order[state['depth']]
        return [(feature, interval) for interval in self.get_intervals(feature)]
    def get_intervals(self, feature):
        min_val, max_val = self.df[feature].min(), self.df[feature].max()
        step = interval_sizes[feature]
        if step <= 0: step = 1.0
        num_intervals = max(1, int((max_val - min_val) / step) + 1)
        intervals = []
        for i in range(num_intervals):
            start = min_val + i * step
            end = min(start + step, max_val)
            intervals.append((start, end))
            if end == max_val: break
        if intervals and intervals[-1][1] < max_val:
            intervals.append((intervals[-1][1], max_val))
        return intervals
    def apply_action(self, state, action):
        feature, interval = action
        new_path = state['path'] + [(feature, interval)]
        filtered = filter_rows_by_path(self.df, new_path)
        if len(filtered) == 0: return None
        avg_vals = filtered.mean(numeric_only=True).to_dict()
        crops = filtered['label'].tolist()
        crop_dist = dict(Counter(crops))
        dominant_crop = max(crop_dist, key=crop_dist.get, default=None) if crop_dist else None
        dominant_crop_ratio = crop_dist[dominant_crop] / len(crops) if crops and dominant_crop in crop_dist else 0
        gift_values = {k: avg_vals.get(k, 0) for k in gift_weights.keys()}
        gift_ranges = {k: (filtered[k].min(), filtered[k].max()) for k in gift_values.keys() if k in filtered.columns and k != 'water_usage_efficiency'}
        efficiencies = []
        for _, row in filtered.iterrows():
            if row['irrigation_frequency'] > 0:
                source_score = 1.0 if row['water_source_type'] == 1 else 1.5 if row['water_source_type'] == 2 else 2.0
                efficiency = row['rainfall'] / (row['irrigation_frequency'] * source_score)
                efficiencies.append(efficiency)
        gift_ranges['water_usage_efficiency'] = (min(efficiencies), max(efficiencies)) if efficiencies else (0.1, 1.0)
        gift_values['water_usage_efficiency'] = sum(efficiencies) / len(efficiencies) if efficiencies else 0.5
        penalty = compute_penalty_by_depth(self.user_inputs, avg_vals, feature_ranges, len(new_path), self.feature_order, penalty_weights)
        gift_score = compute_gift(gift_values, self.user_inputs, gift_ranges)
        heuristic = compute_heuristic(gift_score, penalty)
        return {
            'path': new_path,
            'path_values': {k: (a + b) / 2 for k, (a, b) in new_path},
            'valid_rows': filtered.index.tolist(),
            'crop_distribution': crop_dist,
            'dominant_crop': dominant_crop,
            'dominant_crop_ratio': dominant_crop_ratio,
            'average_values': avg_vals,
            'gift_values': gift_values,
            'heuristic_score': heuristic,
            'depth': len(new_path)
        }

def greedy_search_k_best(problem, k=3, max_iterations=10000):
    root_node = CropNode(problem.initial_state)
    frontier = [(root_node.state['heuristic_score'], root_node)]
    heapq.heapify(frontier)
    goal_nodes = []
    visited = set()
    seen_crops = set()
    iteration = 0
    while frontier and iteration < max_iterations and len(goal_nodes) < k:
        print(iteration)
        iteration += 1
        _, node = heapq.heappop(frontier)
        node_hash = hash(node)
        if node_hash in visited: continue
        visited.add(node_hash)
        if problem.is_goal(node.state):
            crop = node.state['dominant_crop']
            if crop and crop not in seen_crops:
                goal_nodes.append(node)
                seen_crops.add(crop)
            continue
        for action in problem.get_valid_actions(node.state):
            new_state = problem.apply_action(node.state, action)
            if new_state:
                new_node = CropNode(new_state, node, action)
                heapq.heappush(frontier, (new_state['heuristic_score'], new_node))
    return [{
        'crop': node.state['dominant_crop'],
        'heuristic_score': node.state['heuristic_score'],
        'path_values': node.state['path_values'],
        'path': node.state['path'],
        'average_values': node.state['average_values'],
        'gift_values': node.state['gift_values']
    } for node in goal_nodes[:k]] if goal_nodes else [{'crop': None, 'heuristic_score': 0, 'path_values': {}, 'path': [], 'average_values': {}, 'gift_values': {}}] * k

def a_star_search_k_best(problem, k=3, max_iterations=10000):
    root_node = CropNode(problem.initial_state)
    frontier = [(root_node.path_cost + root_node.state['heuristic_score'], root_node)]
    heapq.heapify(frontier)
    goal_nodes = []
    visited = set()
    seen_crops = set()
    iteration = 0
    while frontier and iteration < max_iterations and len(goal_nodes) < k:
        iteration += 1
        _, node = heapq.heappop(frontier)
        node_hash = hash(node)
        if node_hash in visited: continue
        visited.add(node_hash)
        if problem.is_goal(node.state):
            crop = node.state['dominant_crop']
            if crop and crop not in seen_crops:
                goal_nodes.append(node)
                seen_crops.add(crop)
            continue
        for action in problem.get_valid_actions(node.state):
            new_state = problem.apply_action(node.state, action)
            if new_state:
                new_path_cost = node.path_cost + compute_cost(problem.user_inputs, new_state['average_values'], feature_ranges, node.depth + 1, penalty_weights)
                new_node = CropNode(new_state, node, action, path_cost=new_path_cost)
                heapq.heappush(frontier, (new_node.path_cost + new_node.state['heuristic_score'], new_node))
    return [{
        'crop': node.state['dominant_crop'],
        'heuristic_score': node.state['heuristic_score'],
        'path_values': node.state['path_values'],
        'path': node.state['path'],
        'average_values': node.state['average_values'],
        'gift_values': node.state['gift_values']
    } for node in goal_nodes[:k]] if goal_nodes else [{'crop': None, 'heuristic_score': 0, 'path_values': {}, 'path': [], 'average_values': {}, 'gift_values': {}}] * k

def compute_cost(user_inputs, node_avg_values, feature_ranges, current_depth, penalty_weights):
    cost = 0
    active_features = feature_order[:current_depth] if current_depth > 0 else feature_order
    normalized_user = normalize_values(user_inputs, feature_ranges, active_features)
    normalized_node = normalize_values(node_avg_values, feature_ranges, active_features)
    for feature in active_features:
        user_val = normalized_user.get(feature, 0)
        node_val = normalized_node.get(feature, 0)
        penalty = penalty_weights.get(feature, 1)
        cost += penalty * abs(user_val - node_val)
    return cost

class Individual:
    def __init__(self, crop_type, growth_stage, env_conditions, crop_info):
        self.crop_type = crop_type
        self.growth_stage = growth_stage
        self.env_conditions = env_conditions
        self.crop_info = crop_info
    def __str__(self):
        return f"Crop Type: {self.crop_type}\nGrowth Stage: {self.growth_stage}"

class CropPredictionProblemGA:
    def __init__(self, data_file, user_input, excluded_crops=None):
        self.crop_data = pd.read_csv(data_file)
        self.user_env = user_input
        self.excluded_crops = excluded_crops if excluded_crops else []
        self.crops_dictionary = self._create_crop_dictionary()
        self._precompute_column_ranges()
    def _precompute_column_ranges(self):
        self.column_ranges = {}
        for col in self.crop_data.columns:
            if col not in ['label', 'growth_stage', 'soil_type', 'water_source_type']:
                min_val = self.crop_data[col].min()
                max_val = self.crop_data[col].max()
                self.column_ranges[col] = {'min': min_val, 'max': max_val if max_val > min_val else min_val + 1}
    def _create_crop_dictionary(self):
        crop_dict = {}
        for _, row in self.crop_data.iterrows():
            crop_type = row['label']
            if crop_type not in self.excluded_crops:
                entry = {
                    "growth_stage": row['growth_stage'],
                    "env_conditions": {
                        k: row[k] for k in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type',
                                            'water_source_type', 'co2_concentration', 'soil_moisture', 'organic_matter',
                                            'sunlight_exposure', 'wind_speed', 'urban_area_proximity']
                    },
                    "crop_info": {
                        k: row[k] for k in ['irrigation_frequency', 'fertilizer_usage', 'pest_pressure', 'frost_risk',
                                            'crop_density', 'water_usage_efficiency']
                    }
                }
                crop_dict.setdefault(crop_type, []).append(entry)
        return crop_dict
    def generate_random_individual(self):
        if not self.crops_dictionary:
            return Individual("unknown", "unknown", {}, {})
        crop_type = random.choice(list(self.crops_dictionary.keys()))
        growth_data = random.choice(self.crops_dictionary[crop_type])
        return Individual(crop_type, growth_data['growth_stage'], growth_data['env_conditions'], growth_data['crop_info'])
    def normalize_value(self, column, value):
        if column not in self.column_ranges: return 0
        col_range = self.column_ranges[column]
        denominator = col_range['max'] - col_range['min']
        return (value - col_range['min']) / denominator if denominator != 0 else 0
    def fitness(self, crop):
        if not crop.env_conditions or not self.user_env: return 0
        return 0.4 * self._match_score(crop.env_conditions, self.user_env)
    def _match_score(self, env_conditions, user_env):
        score = 0
        counter = 0
        for condition in env_conditions:
            if condition in ['soil_type', 'water_source_type']:
                if env_conditions[condition] != user_env.get(condition, env_conditions[condition]):
                    score += 1
            else:
                score += abs(self.normalize_value(condition, env_conditions[condition]) - self.normalize_value(condition, user_env.get(condition, 0)))
            counter += 1
        return 1 - (score / counter) if counter > 0 else 0
    def find_closest_crop(self, target_env):
        if not self.crops_dictionary:
            return Individual("unknown", "unknown", target_env, {})
        best_score, best_crop = -float('inf'), None
        for crop_type in self.crops_dictionary:
            for crop_data in self.crops_dictionary[crop_type]:
                score = self._match_score(crop_data['env_conditions'], target_env)
                if score > best_score:
                    best_score = score
                    best_crop = (crop_type, crop_data)
        return Individual(best_crop[0], best_crop[1]['growth_stage'], target_env, best_crop[1]['crop_info']) if best_crop else Individual("unknown", "unknown", target_env, {})

class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, generations=80, mutation_rate=0.2, tournament_size=3):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
    def solve(self):
        population = [self.problem.generate_random_individual() for _ in range(self.population_size)]
        if not population or all(ind.crop_type == "unknown" for ind in population):
            return Individual("unknown", "unknown", {}, {})
        best_solution = max(population, key=lambda x: self.problem.fitness(x))
        for _ in range(self.generations):
            population = self.evolve_population(population)
            current_best = max(population, key=lambda x: self.problem.fitness(x))
            if self.problem.fitness(current_best) > self.problem.fitness(best_solution):
                best_solution = copy.deepcopy(current_best)
        return best_solution
    def evolve_population(self, population):
        new_pop = [max(population, key=lambda x: self.problem.fitness(x))]
        while len(new_pop) < self.population_size:
            p1, p2 = [self._tournament_selection(population) for _ in range(2)]
            child = self._crossover(p1, p2)
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            new_pop.append(child)
        return new_pop
    def _tournament_selection(self, population):
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: self.problem.fitness(x))
    def _crossover(self, p1, p2):
        if not p1.env_conditions or not p2.env_conditions:
            return self.problem.generate_random_individual()
        child_env = {}
        for key in p1.env_conditions:
            if key in ['soil_type', 'water_source_type']:
                child_env[key] = random.choice([p1.env_conditions[key], p2.env_conditions[key]])
            else:
                child_env[key] = (p1.env_conditions[key] + p2.env_conditions[key]) / 2
        return self.problem.find_closest_crop(child_env)
    def _mutate(self, individual):
        if not individual.env_conditions:
            return self.problem.generate_random_individual()
        mutated_env = copy.deepcopy(individual.env_conditions)
        for key in mutated_env:
            if key not in ['soil_type', 'water_source_type']:
                col_range = self.problem.column_ranges.get(key, {'min': 0, 'max': 1})
                current_val = mutated_env[key]
                mutation_range = (col_range['max'] - col_range['min']) * 0.1
                mutated_val = current_val + random.uniform(-mutation_range, mutation_range)
                mutated_env[key] = max(col_range['min'], min(col_range['max'], mutated_val))
        return self.problem.find_closest_crop(mutated_env)

def get_k_distinct_crops(data_file, user_input, population_size=50, generations=120, mutation_rate=0.2, tournament_size=3, k=3):
    top_k_solutions = []
    excluded_crops = []
    for i in range(k):
        problem = CropPredictionProblemGA(data_file, user_input, excluded_crops)
        if not problem.crops_dictionary:
            break
        ga = GeneticAlgorithm(problem, population_size, generations, mutation_rate, tournament_size)
        best_solution = ga.solve()
        if best_solution.crop_type == "unknown":
            break
        top_k_solutions.append(best_solution)
        excluded_crops.append(best_solution.crop_type)
    return top_k_solutions

def csp_search_k_crops(df, user_inputs, k=3, tolerance=0.1):
    specific_features = ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'ph', 'soil_moisture', 'organic_matter', 'sunlight_exposure', 'wind_speed', 'urban_area_proximity']
    problem = Problem()
    specific_ranges = {}

    for crop in df['label'].unique():
        crop_data = df[df['label'].str.lower() == crop.lower()]
        specific_ranges[crop] = {
            feature: {
                'min': crop_data[feature].min(),
                'max': crop_data[feature].max()
            } for feature in specific_features if feature in crop_data
        }
        for feature, range_ in specific_ranges[crop].items():
            min_val, max_val = range_['min'], range_['max']
            range_width = max_val - min_val
            tol_min = min_val - range_width * tolerance
            tol_max = max_val + range_width * tolerance
            specific_ranges[crop][feature]['tol_min'] = tol_min
            specific_ranges[crop][feature]['tol_max'] = tol_max

    crop_list = list(specific_ranges.keys())
    problem.addVariable("crop", crop_list)

    def feature_constraint(crop_value):
        for feature, value in user_inputs.items():
            if feature in specific_features and feature in specific_ranges[crop_value]:
                range_ = specific_ranges[crop_value][feature]
                tol_min = range_['tol_min']
                tol_max = range_['tol_max']
                if not (tol_min <= value <= tol_max):
                    return False
        return True

    problem.addConstraint(feature_constraint, ["crop"])
    solutions = problem.getSolutions()

    if not solutions:
        compatibility_scores = []
        for crop in crop_list:
            score = 0
            count = 0
            for feature, value in user_inputs.items():
                if feature in specific_features and feature in specific_ranges[crop]:
                    min_val = specific_ranges[crop][feature]['min']
                    max_val = specific_ranges[crop][feature]['max']
                    range_width = max_val - min_val if max_val != min_val else 1
                    diff = abs(value - (min_val + max_val) / 2) / range_width
                    compatibility = max(0, 1 - diff)
                    score += compatibility
                    count += 1
            avg_score = score / count if count > 0 else 0
            compatibility_scores.append((crop, avg_score))

        compatibility_scores.sort(key=lambda x: x[1], reverse=True)
        solutions = [{'crop': crop} for crop, _ in compatibility_scores[:k]]

    results = []
    for sol in solutions:
        crop = sol['crop']
        crop_data = df[df['label'] == crop]
        avg_vals = crop_data.mean(numeric_only=True).to_dict()
        gift_values = {k: avg_vals.get(k, 0) for k in gift_weights.keys()}
        gift_ranges = {k: (crop_data[k].min(), crop_data[k].max()) for k in gift_values.keys() if k in crop_data.columns and k != 'water_usage_efficiency'}
        efficiencies = []
        for _, row in crop_data.iterrows():
            irrigation = row['irrigation_frequency'] if row['irrigation_frequency'] > 0 else 1
            source_score = 1.0 if row['water_source_type'] == 1 else 1.5 if row['water_source_type'] == 2 else 2.0
            efficiency = row['rainfall'] / (irrigation * source_score)
            efficiencies.append(efficiency)
        gift_ranges['water_usage_efficiency'] = (min(efficiencies), max(efficiencies)) if efficiencies else (0.1, 1.0)
        gift_values['water_usage_efficiency'] = sum(efficiencies) / len(efficiencies) if efficiencies else 0.5
        penalty = compute_penalty_by_depth(user_inputs, avg_vals, feature_ranges, len(feature_order), feature_order, penalty_weights)
        gift_score = compute_gift(gift_values, user_inputs, gift_ranges)
        heuristic = compute_heuristic(gift_score, penalty)
        path = [(f, (specific_ranges[crop][f]['tol_min'], specific_ranges[crop][f]['tol_max'])) for f in specific_features]
        results.append({
            'crop': crop,
            'heuristic_score': heuristic,
            'path_values': {f: avg_vals.get(f, 0) for f in specific_features},
            'path': path,
            'average_values': avg_vals,
            'gift_values': gift_values
        })
    results.sort(key=lambda x: x['heuristic_score'])
    return results[:k] if results else [{'crop': None, 'heuristic_score': 0, 'path_values': {}, 'path': [], 'average_values': {}, 'gift_values': {}}] * k

def predict_crop(method, user_inputs, k=3):
    initial_state = {
        'path': [],
        'path_values': {},
        'valid_rows': df.index.tolist(),
        'crop_distribution': {},
        'dominant_crop': None,
        'dominant_crop_ratio': 0,
        'average_values': {},
        'gift_values': {k: 0 for k in gift_weights.keys()},
        'heuristic_score': 0,
        'depth': 0
    }

    if method in ['astar', 'greedy']:
        problem = CropPredictionProblemSearch(initial_state, user_inputs, feature_order, df)
        results = a_star_search_k_best(problem, k=k) if method == 'astar' else greedy_search_k_best(problem, k=k)
    elif method == 'genetic':
        results = []
        top_solutions = get_k_distinct_crops("Crop_recommendationV2.csv", user_inputs, k=k)
        for sol in top_solutions:
            crop_data = df[df['label'] == sol.crop_type]
            avg_vals = crop_data.mean(numeric_only=True).to_dict()
            gift_values = {k: avg_vals.get(k, 0) for k in gift_weights.keys()}
            gift_ranges = {k: (crop_data[k].min(), crop_data[k].max()) for k in gift_values.keys() if k in crop_data.columns and k != 'water_usage_efficiency'}
            efficiencies = []
            for _, row in crop_data.iterrows():
                irrigation = row['irrigation_frequency'] if row['irrigation_frequency'] > 0 else 1
                source_score = 1.0 if row['water_source_type'] == 1 else 1.5 if row['water_source_type'] == 2 else 2.0
                efficiency = row['rainfall'] / (irrigation * source_score)
                efficiencies.append(efficiency)
            gift_ranges['water_usage_efficiency'] = (min(efficiencies), max(efficiencies)) if efficiencies else (0.1, 1.0)
            gift_values['water_usage_efficiency'] = sum(efficiencies) / len(efficiencies) if efficiencies else 0.5
            penalty = compute_penalty_by_depth(user_inputs, avg_vals, feature_ranges, len(feature_order), feature_order, penalty_weights)
            gift_score = compute_gift(gift_values, user_inputs, gift_ranges)
            heuristic = compute_heuristic(gift_score, penalty)
            results.append({
                'crop': sol.crop_type,
                'heuristic_score': heuristic,
                'path_values': sol.env_conditions,
                'path': [(k, (v-0.1, v+0.1)) for k, v in sol.env_conditions.items() if k in feature_order],
                'average_values': avg_vals,
                'gift_values': gift_values,
                'growth_stage': sol.growth_stage
            })
    elif method == 'csp':
        results = csp_search_k_crops(df, user_inputs, k=k)
    else:
        raise ValueError("Invalid method specified")

    response = {
        'bestCrop': None,
        'otherCrops': [],
        'environmentCondition': {},
        'userCondition': user_inputs,
        'recommendedCropManagement': {}
    }

    if results and results[0]['crop']:
        best_result = results[0]
        crop_data = df[df['label'] == best_result['crop']].iloc[0]
        response['bestCrop'] = {
            'name': best_result['crop'],
            'growthStage': crop_data.get('growth_stage', 'Unknown'),
            'matchingPercentage': round((1 - best_result['heuristic_score']) * 100, 2),
            'imageUrl': f"/static/image/{best_result['crop'].lower()}.jpg"
        }
        response['environmentCondition'] = best_result['average_values']
        response['recommendedCropManagement'] = best_result['gift_values']
        response['recommendedCropManagement']['frost_risk'] = crop_data.get('frost_risk', 0)
        response['recommendedCropManagement']['crop_density'] = crop_data.get('crop_density', 0)

        for result in results[1:]:
            if result['crop']:
                response['otherCrops'].append({
                    'name': result['crop'],
                    'matchingPercentage': round((1 - result['heuristic_score']) * 100, 2),
                    'imageUrl': f"/static/image/{result['crop'].lower()}.jpg"
                })

    return response

def main():
    user_inputs = {
        'N': 90, 'P': 42, 'K': 43,
        'temperature': 23.5, 'humidity': 60,
        'ph': 6.5, 'rainfall': 120,
        'soil_type': 2,
        'water_source_type': 1,
        'fertilizer_usage': 0.5,
        'co2_concentration': 410,
        'irrigation_frequency': 3,
        'pest_pressure': 0.2,
        'soil_moisture': 25,
        'organic_matter': 2.5,
        'sunlight_exposure': 8,
        'wind_speed': 10,
        'urban_area_proximity': 15
    }

    method = 'greedy'  # Try also 'astar', 'genetic', or 'csp'
    k = 3

    results = predict_crop(method, user_inputs, k)
    print("Raw results:", results)  # <--- Add this line to debug

    if not results:
        print("No crops predicted.")
        return

    print("\nTop Predicted Crops:")
    for i, crop in enumerate(results, 1):
        print(f"\nChoice {i}:")
        print(f"Crop: {crop}")

if __name__ == "__main__":
    main()

