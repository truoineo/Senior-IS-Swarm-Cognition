from mesa.experimental.continuous_space import ContinuousSpaceAgent

import random
import numpy as np

class UnicellularAgent(ContinuousSpaceAgent):
    """
    Agent with local sensing for resources and hazards
    """
    def __init__(self,
                 model,
                 space,
                 pos,
                 sensing_radius,
                 movement_speed,
                 avoidance_threshold):
        
        super().__init__(space=space, model=model)
        self.position = np.array(pos, dtype=float)
        
        # Evolution parameters
        self.sensing_radius = sensing_radius
        self.movement_speed = movement_speed
        self.avoidance_threshold = avoidance_threshold
        
        # State
        self.energy = 100
        self.resources_collected = 0
        
        # Direction (random initial heading as unit vector)
        angle = random.uniform(0, 2 * np.pi)
        self.heading = angle
        self.direction = np.array([np.cos(angle), np.sin(angle)])
    
    def step(self):
        """Execute one step of behavior"""
        # 1. Sense environment
        perceived_resources = self.sense_resources()
        perceived_hazards = self.sense_hazards()
        perceived_neighbors = self.get_neighbors_in_radius(
            radius=self.avoidance_threshold
        )
        
        # 2. Decide movement based on rules (returns direction vector)
        direction_vector = self.decide_movement(
            perceived_resources, 
            perceived_hazards, 
            perceived_neighbors
        )
        
        # 3. Move
        self.move(direction_vector)
        
        # 4. Interact with environment (reuse distances from sensing!)
        interaction_radius = 0.5
        nearby_resources = [
            r for r in perceived_resources 
            if r['distance'] < interaction_radius
        ]
        nearby_hazards = [
            h for h in perceived_hazards 
            if h['distance'] < interaction_radius
        ]
        
        self.collect_resources(nearby_resources)
        self.check_hazards(nearby_hazards)
        
        # 5. Update state
        self.energy -= 1
        if self.energy <= 0:
            self.remove()  # Agent dies if out of energy
    
    def sense_resources(self):
        """Detect resources within sensing radius"""
        detected = []
        for resource in self.model.resources: 
            if resource.is_depleted():
                continue
            
            distances, _ = self.space.calculate_distances(
                point=resource.pos,
                agents=[self]
            )
            distance = distances[0]
            
            if distance <= self.sensing_radius:
                detected.append({
                    'resource': resource,
                    'distance': distance,
                })
        
        return detected
    
    def sense_hazards(self):
        """Detect hazards within sensing radius"""
        detected = []
        for hazard in self.model.hazards:
            distances, _ = self.space.calculate_distances(
                point=hazard.pos,
                agents=[self]
            )
            distance = distances[0]
            
            if distance <= self.sensing_radius:
                detected.append({
                    'hazard': hazard,
                    'distance': distance,
                })
        
        return detected
    
    def decide_movement(self, resources, hazards, neighbors):
        """
        CORE RULE-BASED LOGIC
        Priority: Hazards > Neighbors > Resources > Random Walk
        Returns direction as a unit vector [dx, dy]
        """
        
        # Priority 1: AVOID HAZARDS
        if hazards:
            avoidance_vector = np.zeros(2)
            for hazard_info in hazards:
                diff = self.position - np.array(hazard_info['hazard'].pos)
                
                if self.space.torus:
                    inverse_diff = diff - np.sign(diff) * self.space.size
                    diff = np.where(
                        np.abs(diff) < np.abs(inverse_diff),
                        diff,
                        inverse_diff
                    )
                
                weight = 1.0 / (hazard_info['distance'] + 0.1)
                avoidance_vector += diff * weight
            
            if np.linalg.norm(avoidance_vector) > 0:
                return avoidance_vector / np.linalg.norm(avoidance_vector)
        
        # Priority 2: AVOID CROWDING
        neighbor_agents, distances = neighbors
        if len(neighbor_agents) > 0:
            diff_vectors = self.space.calculate_difference_vector(
                self.position,
                agents=neighbor_agents
            )
            
            repulsion_strengths = 1.0 / (distances + 0.1)
            separation_vector = np.sum(
                diff_vectors * repulsion_strengths[:, np.newaxis],
                axis=0
            )
            
            if np.linalg.norm(separation_vector) > 0.1:
                return separation_vector / np.linalg.norm(separation_vector)
        
        # Priority 3: MOVE TOWARD RESOURCES
        if resources:
            attraction_vector = np.zeros(2)
            for resource_info in resources:
                diff = np.array(resource_info['resource'].pos) - self.position
                
                if self.space.torus:
                    inverse_diff = diff - np.sign(diff) * self.space.size
                    diff = np.where(
                        np.abs(diff) < np.abs(inverse_diff),
                        diff,
                        inverse_diff
                    )
                
                weight = 1.0 / np.sqrt(resource_info['distance'] + 1.0)
                attraction_vector += diff * weight
            
            if np.linalg.norm(attraction_vector) > 0:
                return attraction_vector / np.linalg.norm(attraction_vector)
        
        # Priority 4: RANDOM WALK
        turn_angle = random.uniform(-0.3, 0.3)
        rotation_matrix = np.array([
            [np.cos(turn_angle), -np.sin(turn_angle)],
            [np.sin(turn_angle), np.cos(turn_angle)]
        ])
        
        new_direction = rotation_matrix @ self.direction
        return new_direction / np.linalg.norm(new_direction)
    
    def move(self, direction_vector):
        """Move agent in direction specified by unit vector"""
        # Store direction for next step
        self.direction = direction_vector
        self.heading = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Calculate and apply movement
        new_position = self.position + direction_vector * self.movement_speed
        self.position = new_position
    
    def collect_resources(self, nearby_resources):
        """
        Collect from ALL nearby resources (standard BFO behavior).
        Multiple resources can be collected simultaneously.
        """
        for resource_info in nearby_resources:
            resource = resource_info['resource']
            
            collected = resource.collect(amount=5)
            if collected > 0:
                self.energy += collected
                self.resources_collected += collected
                #if hasattr(self.model, 'resources_collected'):
                    #self.model.resources_collected += collected
    
    def check_hazards(self, nearby_hazards):
        """
        Take damage from ONE nearby hazard (standard BFO behavior).
        Only the first hazard affects the agent per step.
        """
        if nearby_hazards:
            # Take damage from first hazard only
            hazard = nearby_hazards[0]['hazard']
            self.energy -= hazard.damage
            
            if self.energy <= 0:
                self.remove()