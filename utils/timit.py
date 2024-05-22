import time
from functools import wraps
import json
from datetime import datetime
log_path = "C:/Users/doron/OneDrive/Desktop/thesis/TSC/timer.log"


def record_duration(func):
    """
    A decorator that records the duration of the execution of the function 'func'.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record end time
        duration = end_time - start_time
        print(f"Function '{func.__name__}' took {duration} seconds to execute.")
        with open(log_path, 'a') as file:
            file.write(json.dumps({"timestamp": datetime.now().isoformat(), "function_nam": func.__name__, "duration_seconds": duration}) + '\n')
        return result
    return wrapper

# Example of using the decorator
# @record_duration
# def example_function(n):
#     """
#     Example function that sleeps for 'n' seconds.
#     """
#     time.sleep(n)
#     return n

# # Calling the decorated function
# result = example_function(2)
