from mesa.visualization import Slider, SolaraViz, make_plot_component, SpaceRenderer, make_space_component
from mesa.visualization.components import AgentPortrayalStyle
from models import UnicellularModel

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "population_size": Slider(
        label="Population Size",
        value=50,
        min=10,
        max=200,
        step=10,
    ),
    "width": 100,
    "height": 100,
    "num_resources": Slider(
        label="Number of Resources",
        value=30,
        min=5,
        max=100,
        step=5,
    ),
    "num_hazards": Slider(
        label="Number of Hazards",
        value=5,
        min=0,
        max=50,
        step=5,
    ),
    "sensing_radius": Slider(
        label="Sensing Radius",
        value=15,
        min=5,
        max=50,
        step=5,
    ),
    "movement_speed": Slider(
        label="Movement Speed (x0.01)",
        value=10,
        min=5,
        max=200,
        step=5,
    ),
    "avoidance_threshold": Slider(
        label="Avoidance Threshold (x0.1)",
        value=30,
        min=10,
        max=100,
        step=5,
    ),
}

def agent_portrayal(agent):
    """Color agents based on energy level"""
    if agent.energy > 75:
        color = "green"
    elif agent.energy > 50:
        color = "yellow"
    elif agent.energy > 25:
        color = "orange"
    else:
        color = "red"
    
    return AgentPortrayalStyle(
        color=color,
        size=50,
        marker="o",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5
    )

def post_process_population(ax):
    """Customize the matplotlib axes for the population line plot."""
    ax.set_title("Population Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Population")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("auto")
    
population_plot = make_plot_component("Population", post_process = post_process_population)

# Simple approach: Just show agents
# Resources and hazards are background elements

unicellular_model = UnicellularModel()
renderer = SpaceRenderer(model = unicellular_model, backend = "matplotlib")
renderer.draw_agents(agent_portrayal)

def post_process(ax, model):
    """Customize the matplotlib axes after rendering, and draw resources and hazards."""
    from matplotlib.patches import Circle, Rectangle
    
    
    # Draw hazards as transparent red circles
    for hazard in model.hazards:
        circle = Circle(
            hazard.pos,
            radius=hazard.radius,
            color='red',
            alpha=0.3,
            zorder=1,  # Behind agents
            linewidth=1,
            edgecolor='darkred'
        )
        ax.add_patch(circle)
        
        # Add damage label
        ax.text(
            hazard.pos[0],
            hazard.pos[1],
            f'-{int(hazard.damage)}',
            ha='center',
            va='center',
            fontsize=8,
            color='darkred',
            fontweight='bold',
            zorder=2
        )
    
    # Draw resources as green squares
    for resource in model.resources:
        if not resource.is_depleted():
            # Size based on remaining amount (0.5 to 2.0)
            size = 0.5 + (resource.amount / resource.initial_amount) * 1.5
            square = Rectangle(
                (resource.pos[0] - size/2, resource.pos[1] - size/2),  # Bottom-left corner
                width=size,
                height=size,
                color='lightgreen',
                alpha=0.7,
                zorder=1,  # Behind agents
                linewidth=1,
                edgecolor='darkgreen'
            )
            ax.add_patch(square)
            
            # Add amount label
            ax.text(
                resource.pos[0],
                resource.pos[1],
                f'{int(resource.amount)}',
                ha='center',
                va='center',
                fontsize=6,
                color='darkgreen',
                fontweight='bold',
                zorder=2
            )
    
    # Customize axes
    ax.set_title("Unicellular Organism Simulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")



page = SolaraViz(
    model = unicellular_model, 
    renderer = renderer,
    components = [population_plot],
    models_params = model_params,
    name = "Unicellular Organism Simulation",
)
#need conversion to discrete space instead
