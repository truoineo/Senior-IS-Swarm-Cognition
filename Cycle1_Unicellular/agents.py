from mesa.discrete_space import CellAgent, FixedAgent
import numpy as np
import random


class UnicellularAgent(CellAgent):
    """
    Agent with local sensing for resources and hazards.
    NO NEIGHBOR AVOIDANCE - for testing purposes.
    """
    
    # ==================== INITIALIZATION ====================
    
    def __init__(self,
                 model,
                 sensing_radius,
                 movement_speed,
                 avoidance_threshold):
        
        super().__init__(model)
        
        self.sensing_radius = int(sensing_radius)
        self.movement_speed = int(movement_speed)
        self.avoidance_threshold = int(avoidance_threshold)
        
        self.energy = 100
        self.resources_collected = 0
        
        self.direction = random.choice([
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ])
    
    # ==================== MAIN STEP ====================
    
    def step(self):
        """Execute one step of behavior"""
        # 1. Sense environment (for movement decisions)
        perceived_resources = self.sense_resources()
        perceived_hazards = self.sense_hazards()
        
        # 2. Decide movement
        direction = self.decide_movement(perceived_resources, perceived_hazards)
        
        # 3. Move
        self.move(direction)
        
        # 4. Interact with environment AT NEW POSITION
        #    Check current cell directly, not old sensed data
        self.collect_resources_at_current_cell()
        self.check_hazards_at_current_position()
        
        # 5. Update state
        self.energy -= 1
        if self.energy <= 0:
            self.remove()
    
    # ==================== 1. SENSING ====================
    
    def sense_resources(self):
        """Detect resources within sensing radius"""
        detected = []
        
        neighborhood = self.cell.get_neighborhood(
            radius=self.sensing_radius, 
            include_center=True
        )
        
        for agent in neighborhood.agents:
            if isinstance(agent, Resource) and not agent.is_depleted():
                detected.append({
                    'resource': agent,
                    'distance': self.cell_distance(agent.cell.coordinate),
                })
        
        return detected
    
    def sense_hazards(self):
        """Detect hazards within sensing radius"""
        detected = []
        
        max_hazard_radius = 5
        neighborhood = self.cell.get_neighborhood(
            radius=self.sensing_radius + max_hazard_radius, 
            include_center=True
        )
        
        for agent in neighborhood.agents:
            if isinstance(agent, Hazard):
                distance = self.cell_distance(agent.cell.coordinate)
                if distance <= self.sensing_radius + agent.radius:
                    detected.append({
                        'hazard': agent,
                        'distance': distance,
                    })
        
        return detected
    
    def cell_distance(self, other_coord):
        """Chebyshev distance for Moore grid"""
        my_coord = self.cell.coordinate
        
        dx = abs(my_coord[0] - other_coord[0])
        dy = abs(my_coord[1] - other_coord[1])
        
        if self.model.space.torus:
            dx = min(dx, self.model.space.width - dx)
            dy = min(dy, self.model.space.height - dy)
        
        return max(dx, dy)
    
    # ==================== 2. DECISION ====================
    
    def decide_movement(self, resources, hazards):
        """
        SIMPLIFIED RULE-BASED LOGIC (NO NEIGHBOR AVOIDANCE)
        Priority: Hazards > Resources > Random Walk
        """
        
        # Priority 1: AVOID HAZARDS
        if hazards:
            weighted_dx = 0
            weighted_dy = 0
            
            for hazard_info in hazards:
                away = self.get_direction_away(hazard_info['hazard'].cell.coordinate)
                effective_distance = max(0.1, hazard_info['distance'] - hazard_info['hazard'].radius)
                weight = 1.0 / (effective_distance + 0.1)
                weighted_dx += away[0] * weight
                weighted_dy += away[1] * weight
            
            if weighted_dx != 0 or weighted_dy != 0:
                return (int(np.sign(weighted_dx)), int(np.sign(weighted_dy)))
        
        # Priority 2: MOVE TOWARD RESOURCES
        if resources:
            weighted_dx = 0
            weighted_dy = 0
            
            for resource_info in resources:
                toward = self.get_direction_to(resource_info['resource'].cell.coordinate)
                weight = 1.0 / (resource_info['distance'] + 1.0)
                weighted_dx += toward[0] * weight
                weighted_dy += toward[1] * weight
            
            if weighted_dx != 0 or weighted_dy != 0:
                return (int(np.sign(weighted_dx)), int(np.sign(weighted_dy)))
        
        # Priority 3: RANDOM WALK
        if random.random() < 0.2:
            self.direction = random.choice([
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ])
        
        return self.direction
    
    def get_direction_to(self, target_coord):
        """Get direction toward target"""
        my_coord = self.cell.coordinate
        
        dx = target_coord[0] - my_coord[0]
        dy = target_coord[1] - my_coord[1]
        
        if self.model.space.torus:
            if abs(dx) > self.model.space.width / 2:
                dx = -np.sign(dx) * (self.model.space.width - abs(dx))
            if abs(dy) > self.model.space.height / 2:
                dy = -np.sign(dy) * (self.model.space.height - abs(dy))
        
        return (int(np.sign(dx)), int(np.sign(dy)))
    
    def get_direction_away(self, target_coord):
        """Get direction away from target"""
        toward = self.get_direction_to(target_coord)
        return (-toward[0], -toward[1])
    
    # ==================== 3. MOVEMENT ====================
    
    def move(self, direction):
        """Move agent by cell offset direction"""
        self.direction = direction
        
        if self.movement_speed == 1:
            next_cell = self.cell.connections.get(direction)
            if next_cell is not None:
                self.cell = next_cell
        else:
            current = self.cell.coordinate
            new_x = current[0] + direction[0] * self.movement_speed
            new_y = current[1] + direction[1] * self.movement_speed
            
            if self.model.space.torus:
                new_x = new_x % self.model.space.width
                new_y = new_y % self.model.space.height
            else:
                new_x = max(0, min(self.model.space.width - 1, new_x))
                new_y = max(0, min(self.model.space.height - 1, new_y))
            
            target_cell = self.model.space[int(new_x), int(new_y)]
            self.cell = target_cell
    
    # ==================== 4. INTERACTION ====================
    
    def collect_resources_at_current_cell(self):
        """Collect from resources in current cell (after moving)"""
        # Check agents in current cell directly
        for agent in self.cell.agents:
            if isinstance(agent, Resource) and not agent.is_depleted():
                collected = agent.collect(amount=5)
                if collected > 0:
                    self.energy += collected
                    self.resources_collected += collected
    
    def check_hazards_at_current_position(self):
        """Take damage from hazards if within their radius"""
        # Check nearby cells for hazards
        neighborhood = self.cell.get_neighborhood(
            radius=5,  # Max hazard radius
            include_center=True
        )
        
        for agent in neighborhood.agents:
            if isinstance(agent, Hazard):
                distance = self.cell_distance(agent.cell.coordinate)
                if distance <= agent.radius:
                    self.energy -= agent.damage
                    if self.energy <= 0:
                        self.remove()
                        return


# ==================== ENVIRONMENT ENTITIES ====================

class Resource(FixedAgent):
    """Resource patch - auto-removes when depleted"""
    
    def __init__(self, model, amount=100):
        super().__init__(model)
        self.amount = amount
        self.initial_amount = amount
    
    def collect(self, amount=1):
        """Agent collects from this resource"""
        if self.amount > 0:
            collected = min(amount, self.amount)
            self.amount -= collected
            
            if self.amount <= 0:
                self.remove()
            
            return collected
        return 0
    
    def is_depleted(self):
        return self.amount <= 0


class Hazard(FixedAgent):
    """Hazardous area with radius of effect"""
    
    def __init__(self, model, damage=10, radius=2):
        super().__init__(model)
        self.damage = damage
        self.radius = radius
