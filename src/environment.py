import gym
from gym import spaces
import numpy as np
import json

class CustomEnv(gym.Env):
    """A custom environment for reinforcement learning."""
    def __init__(self, config_path):
        super(CustomEnv, self).__init__()
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        # Modifications to incorporate task types, characteristics, and execution locations
        # self.task_type = self.config["task_type"]  # Independent or sequence
        # self.task_size = self.config["task_size"]  # e.g., in MB
        # self.computing_complexity = self.config["computing_complexity"]
        # self.execution_location = self.config["execution_location"]  # Local, edge, or cloud
        # self.execution_status = self.config["execution_status"]
        # self.sequence_order = None if self.task_type == "independent" else self.config["sequence_order"]
        
        # # Define action and observation space according to the new parameters
        # self.action_space = spaces.Discrete(len(self.config["execution_location_options"]))
        self.observation_space = self.define_observation_space()

        
        #Current exectuing task
        self.current_executing_task = 0
        
        
    def can_execute_task(self, current_task):
        if current_task['task_type'][0] == 1:  # It's a sequential task
            sequence_group = current_task['task_type'][1]
            for task in self.config['tasks']:
                if (task['sequence_order'] < current_task['sequence_order'] and 
                    task['execution_status'] != 2):  # Predecessor not completed
                        return False  # Current task cannot execute yet
        return True  # Task can execute

    def define_observation_space(self):
        # Assume maximum task size and computing complexity are predefined or calculated similarly
        max_task_size = max([task["task_size"] for task in self.config["tasks"]])
        max_computing_complexity = max([task["computing_complexity"] for task in self.config["tasks"]])
        
        # Dynamically find the highest sequence identifier
        max_sequence_id = max(task["task_type"][1] for task in self.config["tasks"] if task["task_type"][0] == 1)
        
        low = np.array([0, 0, 0, 0])  # Assuming four elements: task_type[0], task_type[1], task_size, computing_complexity
        high = np.array([1, max_sequence_id, max_task_size, max_computing_complexity])
        
        return spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        # Reset the environment state
        self.state = self.initialize_state()
        
        # Reset execution status for all tasks
        for task in self.config['tasks']:
            task['execution_status'] = 1  # Set to 'to be executed'
        
        # Optionally, reset other environment variables
        
        # Return the initial environment observation
        return self.state
        
    def step(self, action):
        # Initialize cumulative reward if not already done
        if not hasattr(self, 'cumulative_reward'):
            self.cumulative_reward = 0

        current_task_index = self.current_executing_task
        current_task = self.config['tasks'][current_task_index]
        current_task_type = current_task['task_type']
        print(current_task_type)
        # Check if the task can be executed
        if self.can_execute_task(current_task):
            # Simulate task execution and update task status
            current_task['execution_status'] = 2  # Mark as executed
            current_task['execution_location'] = self.determine_execution_location(action)
            
            # Calculate the reward for this task and add to cumulative reward
            # task_reward = self.calculate_reward(current_task)
            task_reward = 1
            self.cumulative_reward += task_reward
            
            # Move to the next task
            self.current_executing_task += 1

            # Check if all tasks are completed
            done = self.check_all_tasks_completed()
            
            if done:
                # Optionally reset cumulative reward for the next episode
                final_reward = self.cumulative_reward
                self.cumulative_reward = 0  # Reset for the next episode
                return self.state, final_reward, done, {}
        else:
            # If the current task cannot be executed, you might want to handle this differently
            task_reward = 0  # No reward if task cannot be executed

        # Not done yet, or task couldn't be executed
        return self.state, task_reward, False, {"message": "Proceeding to next task or unable to execute current task"}

    def determine_execution_location(self, action):
        # Map action to execution location
        if action == 0:
            return "local"
        elif action == 1:
            return "edge"
        else:  # action == 2
            return "cloud"

    def calculate_reward(self, task):
        # Example placeholders for latency and expected completion time
        latency = self.get_task_latency(task)  # Implement this based on your latency measurements
        expected_time = 10  # Define expected times for tasks
        
        # Calculate reward components
        latency_reward = 1.0 / (latency + 1)  # Inverse of latency, +1 to avoid division by zero
        delay_penalty = -1 if latency > expected_time else 0  # Penalize if task takes longer than expected
        completion_reward = 1 if task['execution_status'] == 2 else 0  # Reward for completing a task
        
        # Combine rewards
        reward = latency_reward + delay_penalty + completion_reward
        return reward
    
    def check_all_tasks_completed(self):
    # Check if all tasks have been executed (status 2)
        all_completed = all(task['execution_status'] == 2 for task in self.config['tasks'])
        return all_completed

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current State: {self.state}")
    # Add more detailed visualization logic here if necessary
    # For more complex visualizations, you might implement GUI rendering or use libraries like Matplotlib or Pygame

    # Implementation of render logic (optional for visualization)

    def initialize_state(self):
        """Initializes the state of the environment."""
        # Implementation based on your environment
        return np.zeros(self.observation_space.shape)

    def update_state(self, action):
        """Updates the environment's state based on an action."""
        # Implementation based on your environment
        # Example: Update self.state, calculate reward, check if done
        return self.state, reward, done, info

# if __name__ == "__main__":
#     # Example usage
#     env = CustomEnv(config_path="config/environment.json")
#     state = env.reset()
#     for _ in range(100):
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         if done:
#             break

# Test on reset
# # Create an instance of the environment
# env = CustomEnv(config_path="config/environment.json")

# # Call the reset method to initialize the environment
# state = env.reset()

# # The 'state' variable now contains the initial state of the environment
# print("Initial State:", state)

# Test on step
if __name__ == "__main__":
    env = CustomEnv(config_path="config/environment.json")
    state = env.reset()
    print("Initial State:", state)

    done = False
    while not done:
        # Example of manually selecting an action or using a simple policy
        # This is where you'd implement your action selection logic
        action = 1  # Randomly sample an action

        state, reward, done, info = env.step(action)
        print(f"Action: {action}, New State: {state}, Reward: {reward}, Done: {done}")
        print()

    print("Episode finished")