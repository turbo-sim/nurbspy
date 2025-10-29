import jax.numpy as jnp
import optimistix as optx
import matplotlib.pyplot as plt


# Function to maximize
def f(x):
    lower, upper = 0.0, 2.0
    x_clamped = jnp.clip(x, lower, upper)
    return x_clamped**3

# Define soft-bounded loss (we minimize this)
def loss(x):
    lower, upper = 0.0, 2.0
    lam = 1e2  # penalty strength
    penalty = jnp.where(x < lower, (lower - x)**2, 0.0) + jnp.where(x > upper, (x - upper)**2, 0.0)
    return -f(x) + lam * penalty  # maximize f(x) by minimizing -f(x)

# Define objective for optimistix
def objective(x, args):
    return loss(x)

# Initial guess
x0 = jnp.array(0.5)

# Use a stable optimizer
solver = optx.BFGS(rtol=1e-6, atol=1e-6)

# Run optimization
sol = optx.minimise(objective, solver, y0=x0)

# Extract optimum
x_opt = sol.value
f_opt = f(x_opt)
print(f"Optimal x = {x_opt:.6f}")
print(f"Maximum f(x) = {f_opt:.6f}")

# ---- Plot section ----
x_vals = jnp.linspace(-1.0, 3.0, 10000)
f_vals = f(x_vals)
loss_vals = loss(x_vals)

# Plot f(x) and penalized loss
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(x_vals, f_vals, label="f(x) = x³", linewidth=2)
ax1.plot(x_vals, -loss_vals, "--", label="effective objective (-loss)", linewidth=2)
ax1.axvline(0, color="k", linestyle=":", linewidth=1)
ax1.axvline(2, color="k", linestyle=":", linewidth=1)
ax1.axvspan(-1, 0, color="gray", alpha=0.1)
ax1.axvspan(2, 3, color="gray", alpha=0.1)

# Mark the optimal point
ax1.plot(x_opt, f_opt, "ro", label=f"optimum x = {x_opt:.3f}")
ax1.set_xlabel("x")
ax1.set_ylabel("Value")
ax1.set_title("Maximization of f(x) = x³ with penalized bounds [0, 2]")
ax1.legend()
ax1.set_ylim([-5, 15])
ax1.grid(True)
plt.tight_layout(pad=1)
plt.show()
