from mesa.visualization import Slider, SolaraViz, make_plot_component, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle
from models import UnicellularModel
from agents import UnicellularAgent, Resource, Hazard


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
        label="Sensing Radius (cells)",
        value=15,
        min=1,
        max=30,
        step=1,
    ),
    "movement_speed": Slider(
        label="Movement Speed (cells/step)",
        value=1,
        min=1,
        max=5,
        step=1,
    ),
    "avoidance_threshold": Slider(
        label="Avoidance Threshold (cells)",
        value=3,
        min=1,
        max=10,
        step=1,
    ),
}


def agent_portrayal(agent):
    """Portrayal function for all agent types."""
    if isinstance(agent, UnicellularAgent):
        if agent.energy > 75:
            color = "tab:blue"
        elif agent.energy > 50:
            color = "tab:cyan"
        elif agent.energy > 25:
            color = "tab:orange"
        else:
            color = "tab:red"
        size = 50
        zorder = 3
        alpha = 0.9
        
    elif isinstance(agent, Resource):
        color = "tab:green"
        # Size scales with amount
        size = 30 + (agent.amount / agent.initial_amount) * 70
        zorder = 1
        # Alpha scales with amount: 0.2 (nearly depleted) to 0.9 (full)
        alpha = 0.2 + (agent.amount / agent.initial_amount) * 0.7
        
    elif isinstance(agent, Hazard):
        color = "tab:red"
        cells_covered = (2 * agent.radius + 1)
        size = 50 * cells_covered
        zorder = 0
        alpha = 0.3
        
    else:
        color = "gray"
        size = 25
        zorder = 0
        alpha = 0.5
    
    return AgentPortrayalStyle(
        color=color,
        size=size,
        marker="s",
        zorder=zorder,
        alpha=alpha,
    )


def post_process_space(ax):
    """Customize the space visualization."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def post_process_lines(ax):
    """Customize the line plot."""
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


# Create line plot components
population_plot = make_plot_component(
    {
        "Population": "tab:blue",
        "Active Resources": "tab:green",
    },
    post_process=post_process_lines,
)

energy_plot = make_plot_component(
    {"Mean Energy": "tab:orange"},
)

survival_plot = make_plot_component(
    {"Survival Rate": "tab:purple"},
)


# Initialize model
unicellular_model = UnicellularModel()

# Create renderer and draw agents
renderer = SpaceRenderer(
    unicellular_model,
    backend="matplotlib",
)
renderer.draw_agents(agent_portrayal)
renderer.post_process = post_process_space

# Create the visualization page
page = SolaraViz(
    unicellular_model,
    renderer,
    components=[population_plot, energy_plot, survival_plot],
    model_params=model_params,
    name="Unicellular Organism Simulation",
)

page  # noqa