import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Optional


class AlgorithmManager:
    def __init__(self, algorithms):
        """
        Initialize the manager with a list of algorithm names.

        Args:
            algorithms (list): List of algorithm names as strings
        """
        self.algorithms = algorithms
        self._create_mappings()

    def _create_mappings(self):
        """Create the index-to-name, predictions, and durations dictionaries."""
        self.algo_dict = {i: name for i, name in enumerate(self.algorithms)}
        self.predictions_dict = {name: [] for name in self.algorithms}
        self.durations_dict = {name: [] for name in self.algorithms}

    def add_algorithm(self, name):
        """
        Add a new algorithm to the manager.

        Args:
            name (str): Name of the algorithm to add
        """
        if name not in self.algorithms:
            self.algorithms.append(name)
            self._create_mappings()

    def remove_algorithm(self, name: str):
        """
        Remove an algorithm from the manager.

        Args:
            name (str): Name of the algorithm to remove
        """
        if name in self.algorithms:
            self.algorithms.remove(name)
            self._create_mappings()

    def get_algo_by_index(self, index):
        """Get algorithm name by index."""
        return self.algo_dict.get(index)

    def get_predictions(self, algo_name: Optional[str]):
        """Get predictions for a specific algorithm."""
        if algo_name:
            return self.predictions_dict.get(algo_name, [])
        return self.predictions_dict

    def add_prediction(self, algo_name: str, prediction: float):
        """Add a prediction for a specific algorithm."""
        if algo_name in self.predictions_dict:
            self.predictions_dict[algo_name].append(prediction)

    def set_predictions(self, algo_name: str, predictions: list[float]):
        """Add a prediction for a specific algorithm."""
        if algo_name in self.predictions_dict:
            self.predictions_dict[algo_name] = predictions


    def add_duration(self, algo_name, duration):
        """Add an execution duration for a specific algorithm."""
        if algo_name in self.durations_dict:
            self.durations_dict[algo_name].append(duration)

    def get_durations(self, algo_name: Optional[str]):
        """Get all durations for a specific algorithm."""
        if algo_name:
            return self.durations_dict.get(algo_name, [])
        return self.durations_dict

    def get_mean_duration(self, algo_name: str):
        """Get mean duration for a specific algorithm."""
        durations = self.durations_dict.get(algo_name, [])
        return sum(durations) / len(durations) if durations else 0

    def clear_predictions(self, algo_name=None):
        """
        Clear predictions for one or all algorithms.

        Args:
            algo_name (str, optional): If provided, clear only this algorithm's predictions
        """
        if algo_name:
            if algo_name in self.predictions_dict:
                self.predictions_dict[algo_name] = []
        else:
            for name in self.predictions_dict:
                self.predictions_dict[name] = []

    def clear_durations(self, algo_name=None):
        """
        Clear durations for one or all algorithms.

        Args:
            algo_name (str, optional): If provided, clear only this algorithm's durations
        """
        if algo_name:
            if algo_name in self.durations_dict:
                self.durations_dict[algo_name] = []
        else:
            for name in self.durations_dict:
                self.durations_dict[name] = []

    def time_algorithm_parallel(self, algo_name: str):
        """
        Decorator that returns both result and duration for parallel processing.

        Args:
            algo_name (str): Name of the algorithm being timed
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                # Return both the result and the duration
                return {"result": result, "duration": duration, "algo_name": algo_name}

            return wrapper

        return decorator

    def collect_parallel_results(self, parallel_results):
        """
        Collect results from parallel processing and update the manager.

        Args:
            parallel_results (list): List of dictionaries containing results and durations
        """
        for result_dict in parallel_results:
            algo_name = result_dict["algo_name"]
            self.durations_dict[algo_name].append(result_dict["duration"])

    def save(self, filepath):
        """
        Save the AlgorithmManager instance to a file.

        Args:
            filepath (str): Path where to save the manager
        """
        filepath = Path(filepath)
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load an AlgorithmManager instance from a file.

        Args:
            filepath (str): Path to the saved manager

        Returns:
            AlgorithmManager: Loaded instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        """String representation of the AlgorithmManager."""
        return (f"AlgorithmManager(created_at={self.created_at}, "
                f"algorithms={self.algorithms}, "
                f"total_predictions={sum(len(preds) for preds in self.predictions_dict.values())}, "
                f"total_durations={sum(len(durs) for durs in self.durations_dict.values())})")
