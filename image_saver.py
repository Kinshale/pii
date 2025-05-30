import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, FloatSlider, IntSlider, Layout, Dropdown
from IPython.display import display
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def A(x, d):
    return d * (1 - x)


def B(x, a, h, g):
    return (a * x * (1 - x) ** g) / (1 + a * h * x)


def C(x, r, s):
    if x == 0:  # Avoid 0 division error
        return 0

    term1 = r * (1 - x)
    term2 = (1 - r) * x
    return 1 / (1 + (term1 / term2) ** s)


def delta(x, d, a, h, g, r, s):
    return A(x, d) - B(x, a, h, g) * C(x, r, s)


def simulate(x0, d, a, h, g, r, s, steps=50):
    x = np.zeros(steps)
    x[0] = x0

    for t in range(steps - 1):
        x[t + 1] = x[t] + delta(x[t], d, a, h, g, r, s)
        x[t + 1] = np.clip(x[t + 1], 0, 1)

    return x


PARAMETERS = {
    "x0": {
        "default": 0.3,
        "description": "Initial value",
        "range": {"min": 0.01, "max": 0.99, "step": 0.01}
    },
    "steps": {
        "default": 100,
        "description": "Number of simulation steps",
        "range": {"min": 10, "max": 300, "step": 10}
    },
    "d": {
        "default": 0.5,
        "description": "Environmental dynamicity",
        "range": {"min": 0.01, "max": 5, "step": 0.1}
    },
    "a": {
        "default": 2,
        "description": "IT department efficacy",
        "range": {"min": 0.1, "max": 10, "step": 0.1}
    },
    "h": {
        "default": 1,
        "description": "IT system rigidity",
        "range": {"min": 0.1, "max": 5, "step": 0.1}
    },
    "g": {
        "default": 1,
        "description": "IT investment propensity",
        "range": {"min": 0.1, "max": 5, "step": 0.1}
    },
    "r": {
        "default": 0.3,
        "description": "Action threshold",
        "range": {"min": 0.01, "max": 0.99, "step": 0.01}
    },
    "s": {
        "default": 3,
        "description": "Organizational flexibility",
        "range": {"min": 1, "max": 10, "step": 0.1}
    }
}

def cobweb_plot(x0, d, a, h, g, r, s, steps):
    fig, ax = plt.subplots(figsize=(7, 7))
    x_values = np.linspace(0, 1, 1000)
    delta_values = np.array([delta(x, d, a, h, g, r, s) for x in x_values])
    
    next_x_values = x_values + delta_values
    
    next_x_values = np.clip(next_x_values, 0, 1)
    
    ax.plot(x_values, x_values, 'k-', label='y = x')
    
    ax.plot(x_values, next_x_values, 'r-', label='y = x + Δ(x)')
    
    x_sim = np.zeros(steps)
    x_sim[0] = x0
    
    for t in range(steps - 1):
        delta_val = delta(x_sim[t], d, a, h, g, r, s)
        x_sim[t + 1] = x_sim[t] + delta_val
        x_sim[t + 1] = np.clip(x_sim[t + 1], 0, 1)
    
    for i in range(steps - 1):
        ax.plot([x_sim[i], x_sim[i]], 
                [x_sim[i], x_sim[i+1]], 'b-', alpha=0.5)
        
        ax.plot([x_sim[i], x_sim[i+1]], 
                [x_sim[i+1], x_sim[i+1]], 'b-', alpha=0.5)
    
    # Mark the initial point
    ax.plot(x_sim[0], 0, 'go', markersize=8, label='Start')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x(t)', fontsize=12)
    ax.set_ylabel('x(t+1)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(
        "images/int-cases/chaotic-cobweb.pdf",
        dpi=400,  # High resolution
        bbox_inches="tight",  # Avoid cropping
        transparent=False,  # White background
    )
    plt.show()
    
def bifurcation_diagram(b_param, b_min, b_max, params, x0, n_points=700):
    b_values = np.linspace(b_min, b_max, n_points)
    n_last = 100
    n_transient = 100

    plt.figure(figsize=(12, 7))

    for b in b_values:
        params[b_param] = b
        
        x = x0
        
        for _ in range(n_transient):
            x = np.clip(x + delta(x, **params), 0, 1)
        
        x_vals = []
        for _ in range(n_last):
            x = np.clip(x + delta(x, **params), 0, 1)
            x_vals.append(x)
        
        plt.plot([b] * len(x_vals), x_vals, 'k.', markersize=0.5, alpha=0.6)

    plt.xlabel(f"{b_param} ({PARAMETERS[b_param]['description'] if b_param in PARAMETERS else ''})", fontsize=12)
    plt.ylabel("Long-term x value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "images/int-cases/chaotic-bifurcation.pdf",
        dpi=400,  # High resolution
        bbox_inches="tight",  # Avoid cropping
        transparent=False,  # White background
    )
    plt.show()
    
def compute_lyapunov_exponent(x0=0.3, epsilon=1e-8, steps=50, **kwargs):

    x = simulate(x0=x0, steps=steps, **kwargs)
    x_perturbed = simulate(x0=x0 + epsilon, steps=steps, **kwargs)
    

    separation = np.abs(x_perturbed - x)
    
    valid_indices = separation > 0
    if np.sum(valid_indices) < 2:
        return 0
    
    t = np.arange(steps)[valid_indices]
    log_sep = np.log(separation[valid_indices])
    
    # Perform linear regression
    coeffs = np.polyfit(t, log_sep, 1)
    lyapunov_exponent = coeffs[0]
    
    return lyapunov_exponent

def lyapunov_diagram(b_param, b_min, b_max, params, x0, n_points=700):
    b_values = np.linspace(b_min, b_max, n_points)


    lambda_vals = []
    
    for b in b_values:
        params[b_param] = b
        lambda_vals.append(compute_lyapunov_exponent(x0, **params))
               

    plt.figure(figsize=(12, 6))

    plt.plot(b_values, lambda_vals, 
            color='black',    
            linewidth=1.5,
            linestyle='-',
            alpha=1)

    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    plt.xlabel(f"{b_param} ({PARAMETERS[b_param]['description'] if b_param in PARAMETERS else ''})", fontsize=12)
    plt.ylabel("λ (Lyapunov exponent)", fontsize=12)
    plt.grid(True, color='lightgray', linestyle=':', alpha=0.7)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        "images/int-cases/chaotic-lyapunov.pdf",
        dpi=400,  # High resolution
        bbox_inches="tight",  # Avoid cropping
        transparent=False,  # White background
    )
    plt.show()
    
chaotic_params = {"a": 5.0, "h": 0.3, "g": 1, "r": 0.25, "s": 3.0}
x0 = 0.3

bifurcation_diagram("d", 0.1, 2, x0 = 0.3, params=chaotic_params)
lyapunov_diagram("d", 0.1, 2, x0 = 0.3, params=chaotic_params)

chaotic_params.pop("d") #We add d whenn calling the above two functions
cobweb_plot(x0 = x0, d = 0.75, steps = 50, **chaotic_params)