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


def run_simulation(steps, x0, d, a, h, g, r, s):
    b_values = simulate(x0, d, a, h, g, r, s, steps)

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    plt.plot(
        b_values,
        "b-o",
        markersize=6,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    plt.xlabel("Iteration (t)", fontsize=12, labelpad=10)
    plt.ylabel("Dissatisfaction (x)", fontsize=12, labelpad=10)
    plt.title("Business-IT Alignment Dynamics", fontsize=14, pad=20)

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.ylim(0, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        "images/results/res-int-1.pdf",
        dpi=400,  # High resolution
        bbox_inches="tight",  # Avoid cropping
        transparent=False,  # White background
    )

    plt.show()


# run_simulation(
#     60,
#     0.2,
#     0.5,
#     6, 
#     0.40,
#     2.00, 
#     0.25,
#     5.00
# )

def phase_portait(d, a, h, g, r, s, x_range=(0, 1), n_points=100):
    x = np.linspace(x_range[0], x_range[1], n_points)
    dx = np.array([delta(xi, d, a, h, g, r, s) for xi in x])
    
    plt.figure(figsize=(14, 6))
        
    # Plot 20 arrows evenly spaced for the derivative, letting color represent the intensity. 
    arrow_indices = np.linspace(0, len(x)-1, 20, dtype=int)
    arrow_x = x[arrow_indices]
    arrow_dx = dx[arrow_indices]
    
    norm = Normalize(vmin=0, vmax=np.max(np.abs(dx)))
    cmap = plt.cm.RdYlBu_r
    
    arrow_length = 0.02 
    arrow_head_width = 0.02
    arrow_head_length = 0.008
    arrow_y_pos = 0
    
    for xi, dxi in zip(arrow_x, arrow_dx):
        color = cmap(norm(np.abs(dxi)))
        if dxi > 0:
            plt.arrow(xi, arrow_y_pos, arrow_length, 0, 
                     head_width=arrow_head_width, 
                     head_length=arrow_head_length,
                     fc=color, ec=color, 
                     width=0.01,
                     length_includes_head=True,
                     alpha=0.9)
        elif dxi < 0:
            plt.arrow(xi, arrow_y_pos, -arrow_length, 0,
                     head_width=arrow_head_width,
                     head_length=arrow_head_length,
                     fc=color, ec=color,
                     width=0.01,
                     length_includes_head=True,
                     alpha=0.9)
    
    plt.plot(x, dx, 'k-', linewidth=1.5, alpha=0.7, label='dx/dt')
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical',
                        pad=0.05, shrink=0.8)
    cbar.set_label('Speed of Change (|dx/dt|)', labelpad=15)
    
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.xlabel('x (Misalignement)', fontsize=11, labelpad=12)
    plt.ylabel('dx/dt', fontsize=11, labelpad=12)
    plt.title(f'ODE Phase Portrait', fontsize=13, pad=15)
    
    y_max = np.max(np.abs(dx)) * 1.1
    plt.ylim([-y_max, y_max])
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(
        "images/results/res-int-2.pdf",
        dpi=400,  # High resolution
        bbox_inches="tight",  # Avoid cropping
        transparent=False,  # White background
    )
    
    plt.show()

# phase_portait(
#     0.5,
#     2.00,
#     1.00,
#     0.5, 
#     0.3,
#     3.00, 
# ) 

# Define the parameters
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

def bifurcation_diagram(b_param, b_min, b_max, fixed_params, n_points=700):
    b_values = np.linspace(b_min, b_max, n_points)
    n_last = 100
    n_transient = 100

    plt.figure(figsize=(12, 7))

    for b in b_values:
        params = fixed_params.copy()
        params[b_param] = b
        
        x = 0.3
        
        for _ in range(n_transient):
            x = np.clip(x + delta(x, **params), 0, 1)
        
        x_vals = []
        for _ in range(n_last):
            x = np.clip(x + delta(x, **params), 0, 1)
            x_vals.append(x)
        
        plt.plot([b] * len(x_vals), x_vals, 'k.', markersize=0.3, alpha=0.5)

    plt.title(f"Bifurcation Diagram (varying {b_param})", fontsize=14)
    plt.xlabel(f"{b_param} ({PARAMETERS[b_param]['description'] if b_param in PARAMETERS else ''})", fontsize=12)
    plt.ylabel("Long-term x value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "images/results/res-int-3.pdf",
        dpi=400,  # High resolution
        bbox_inches="tight",  # Avoid cropping
        transparent=False,  # White background
    )
    plt.show()

# bifurcation_diagram(
#     b_param='g',
#     b_min=1.20,
#     b_max=2.5,
#     fixed_params={'d': 0.5, 'a': 7, 'h': 0.3, 'r': 0.3, 's': 5.1},  # Must match delta()'s expected keys
#     n_points=700
# )

# =================== A(x) ===================
x = np.linspace(0, 1, 100)
d_values = [0.5, 1, 1.5]
plt.figure(figsize=(8, 6))

for d in d_values:
    plt.plot(x, A(x, d), label=f'd = {d}')
    
plt.xlabel('x', fontsize=12)
plt.ylabel('A(x) - Environmental pressure', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# plt.savefig(
#     "images/params/A(x).pdf",
#     dpi=400,  # High resolution
#     bbox_inches="tight",  # Avoid cropping
#     transparent=False,  # White background
# )

plt.gca().set_aspect('equal', adjustable='box')  # Preserves scaling

plt.xlim(-0.5, 1.5)
plt.ylim(0, 1.8)

# plt.show()

# =================== B(x) ===================
x = np.linspace(0, 1, 500)

parameter_sets = [
    (1, 1, 3),
    (5, 0.5, 2),      
    (5, 0.5, 0.3),    
    (3, 2, 0.3)     
]

plt.figure(figsize=(8, 6))

for a, h, g in parameter_sets:
    y = B(x, a, h, g)
    plt.plot(x, y, label=f'a={a}, h={h}, g={g}')

plt.xlabel('x', fontsize=12)
plt.ylabel('B(x) - IT Department Efficacy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(-0.25, 1.25)
plt.ylim(0, 1.0) 

plt.tight_layout()

# plt.savefig(
#     "images/params/B(x).pdf",
#     dpi=400,  # High resolution
#     bbox_inches="tight",  # Avoid cropping
#     transparent=False,  # White background
# )

# plt.show()

# =================== C(x) ===================
def C(x, r, s):
    # Vectorized computation: avoid division by zero where x=0
    with np.errstate(divide='ignore', invalid='ignore'):  # Temporarily suppress divide-by-zero warnings
        term1 = r * (1 - x)
        term2 = (1 - r) * x
        ratio = np.divide(term1, term2, out=np.zeros_like(term1), where=(term2 != 0))  # Safe division
        result = 1 / (1 + ratio**s)
    return np.where(x == 0, 0, result)  # Explicitly set C(0) = 0

x = np.linspace(0, 1, 500)

parameter_sets = [
    (0.5, 1),
    (0.5, 5),   
    (0.25, 2), 
]

plt.figure(figsize=(8, 6))

for r, s in parameter_sets:
    y = C(x, r, s)
    plt.plot(x, y, label=f'r={r}, s={s}')

plt.xlabel('x', fontsize=12)
plt.ylabel('C(x) - Organizational Adaptability', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0, 1.0)
plt.ylim(0, 1.0) 

plt.tight_layout()

plt.savefig(
    "images/params/C(x).pdf",
    dpi=400,  # High resolution
    bbox_inches="tight",  # Avoid cropping
    transparent=False,  # White background
)

plt.show()