import numpy as np
from mesa import Model
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.datacollection import DataCollector
from agents import UnicellularAgent


def calculate_mean_distance_to_resources(model):
    """Calculate the mean distance from each agent to its nearest resource"""
    if not model.agents:
        return 0
    
    # Get active (non-depleted) resources
    active_resources = [r for r in model.resources if not r.is_depleted()]
    
    if not active_resources:
        return np.nan  # No resources available
    
    min_distances = []
    for agent in model.agents:
        distances_to_resources = []
        for resource in active_resources:
            dists, _ = model.space.calculate_distances(
                point=resource.pos,
                agents=[agent]
            )
            distances_to_resources.append(dists[0])
        
        # Get distance to nearest resource
        min_distance = min(distances_to_resources)
        min_distances.append(min_distance)
    
    return np.mean(min_distances)

class UnicellularModel(Model):
    def __init__(self,                          #Model parameters
                 population_size = 50,
                 width = 100,
                 height = 100,
                 num_resources = 30,
                 num_hazards = 5,
                 sensing_radius = 15,           # Agents parameters (integer from slider)
                 movement_speed = 10,           # Integer from slider (x0.01)
                 avoidance_threshold = 30,      # Integer from slider (x0.1)
                 seed=None):
        super().__init__(seed=seed)
        self.num_agents = population_size
        
        # Convert integer slider values to floats (SolaraViz slider cannot do floats)
        movement_speed = movement_speed * 0.01
        avoidance_threshold = avoidance_threshold * 0.1
        
        # Define continuous space
        self.space = ContinuousSpace(
            dimensions = [[0, width], [0, height]],
            torus=True,
            random=self.random,
            n_agents=self.num_agents
        )
        
        # Create agents 
        positions = self.rng.random(size=(population_size, 2)) * self.space.size
        UnicellularAgent.create_agents(
            self,
            population_size,
            self.space,
            pos=positions,
            sensing_radius=sensing_radius,
            movement_speed=movement_speed,
            avoidance_threshold=avoidance_threshold
        )
        
        # Create resources (using seeded random)
        self.resources = []
        for i in range(num_resources):
            resource_position = (
                self.random.uniform(0, self.space.width), 
                self.random.uniform(0, self.space.height)
            )
            amount = self.random.randint(20, 100)
            resource = Resource(
                unique_id=i, 
                model=self, 
                pos=resource_position, 
                amount=amount
            )
            self.resources.append(resource)
        
        # Create hazards (using seeded random)
        self.hazards = []
        for i in range(num_hazards):
            hazard_position = (
                self.random.uniform(0, self.space.width), 
                self.random.uniform(0, self.space.height)
            )
            radius = self.random.uniform(2, 5)  
            damage = self.random.randint(2, 10)
            hazard = Hazard(
                unique_id=i, 
                model=self, 
                pos=hazard_position, 
                radius=radius, 
                damage=damage
            )
            self.hazards.append(hazard)
            
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: len(m.agents),
                "Mean Energy": lambda m: np.mean([a.energy for a in m.agents]) if m.agents else 0,
                "Total Resources Collected": lambda m: sum(a.resources_collected for a in m.agents),
                "Active Resources": lambda m: sum(1 for r in m.resources if not r.is_depleted()),
                "Survival Rate": lambda m: len(m.agents) / self.num_agents,
                "Mean Distance to Resources": lambda m: calculate_mean_distance_to_resources(m),
            },
            agent_reporters={
                "Energy": "energy",
                "Resources Collected": "resources_collected",
            }
        )
    
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


class Resource:
    """Resource patch that agents can collect from"""
    def __init__(self, unique_id, model, pos, amount=100):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos
        self.amount = amount
        self.initial_amount = amount
    
    def collect(self, amount=1):
        """Agent collects from this resource"""
        if self.amount > 0:
            collected = min(amount, self.amount)
            self.amount -= collected
            return collected
        return 0
    
    def is_depleted(self):
        return self.amount <= 0


class Hazard:
    """Hazardous area that damages agents"""
    def __init__(self, unique_id, model, pos, radius=5.0, damage=10):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos
        self.radius = radius
        self.damage = damage