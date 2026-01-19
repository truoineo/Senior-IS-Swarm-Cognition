import numpy as np
from mesa import Model
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.datacollection import DataCollector
from agents import UnicellularAgent, Resource, Hazard


def calculate_mean_distance_to_resources(model):
    """Calculate the mean distance from each agent to its nearest resource (Chebyshev)"""
    unicellular_agents = model.agents_by_type.get(UnicellularAgent, [])
    
    if not unicellular_agents:
        return 0
    
    resources = model.agents_by_type.get(Resource, [])
    active_resources = [r for r in resources if not r.is_depleted()]
    
    if not active_resources:
        return np.nan
    
    min_distances = []
    for agent in unicellular_agents:
        distances_to_resources = []
        for resource in active_resources:
            dx = abs(agent.cell.coordinate[0] - resource.cell.coordinate[0])
            dy = abs(agent.cell.coordinate[1] - resource.cell.coordinate[1])
            
            if model.space.torus:
                dx = min(dx, model.space.width - dx)
                dy = min(dy, model.space.height - dy)
            
            distance = max(dx, dy)
            distances_to_resources.append(distance)
        
        min_distance = min(distances_to_resources)
        min_distances.append(min_distance)
    
    return np.mean(min_distances)


class UnicellularModel(Model):
    def __init__(self,
                 population_size=50,
                 width=100,
                 height=100,
                 num_resources=30,
                 num_hazards=5,
                 sensing_radius=15,
                 movement_speed=1,
                 avoidance_threshold=3,
                 seed=None):
        super().__init__(seed=seed)
        self.num_agents = population_size
        
        # Define discrete space
        self.space = OrthogonalMooreGrid(
            dimensions=(width, height),
            torus=True,
            random=self.random
        )
        
        # Create UnicellularAgents
        agents = UnicellularAgent.create_agents(
            model=self,
            n=population_size,
            sensing_radius=sensing_radius,
            movement_speed=movement_speed,
            avoidance_threshold=avoidance_threshold
        )
        for agent in agents:
            x = self.random.randint(0, width - 1)
            y = self.random.randint(0, height - 1)
            agent.cell = self.space[x, y]
        
        # Create Resources
        amounts = [self.random.randint(20, 100) for _ in range(num_resources)]
        resources = Resource.create_agents(
            model=self,
            n=num_resources,
            amount=amounts
        )
        for resource in resources:
            x = self.random.randint(0, width - 1)
            y = self.random.randint(0, height - 1)
            resource.cell = self.space[x, y]
        
        # Create Hazards with varying damage and radius
        damages = [self.random.randint(2, 10) for _ in range(num_hazards)]
        radii = [self.random.randint(1, 4) for _ in range(num_hazards)]
        hazards = Hazard.create_agents(
            model=self,
            n=num_hazards,
            damage=damages,
            radius=radii
        )
        for hazard in hazards:
            x = self.random.randint(0, width - 1)
            y = self.random.randint(0, height - 1)
            hazard.cell = self.space[x, y]
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: len(m.agents_by_type.get(UnicellularAgent, [])),
                "Mean Energy": lambda m: np.mean([a.energy for a in m.agents_by_type.get(UnicellularAgent, [])]) if m.agents_by_type.get(UnicellularAgent) else 0,
                "Total Resources Collected": lambda m: sum(a.resources_collected for a in m.agents_by_type.get(UnicellularAgent, [])),
                "Active Resources": lambda m: len(m.agents_by_type.get(Resource, [])),
                "Survival Rate": lambda m: len(m.agents_by_type.get(UnicellularAgent, [])) / m.num_agents,
                "Mean Distance to Resources": calculate_mean_distance_to_resources,
            },
            agenttype_reporters={
                UnicellularAgent: {
                    "Energy": "energy",
                    "Resources Collected": "resources_collected",
                }
            }
        )
    
    def step(self):
        self.agents_by_type[UnicellularAgent].shuffle_do("step")
        self.datacollector.collect(self)