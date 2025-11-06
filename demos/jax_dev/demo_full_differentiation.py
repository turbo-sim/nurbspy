"""
Simple test of JAX differentiability and JIT compilation for NurbsCurve
"""
import os
import time
import jax
import jax.numpy as jnp
import jax.profiler
import nurbspy.jax as nrb

# Define a log directory for the trace
logdir = "jax_traces"
os.makedirs(logdir, exist_ok=True)

# -------------------------------------------------------------------------------------------------------------------- #
# Utility: timed evaluation with individual run times
# -------------------------------------------------------------------------------------------------------------------- #
def timed_eval(fn, *args, repeat=5):
    """Run fn(*args) multiple times, report all and average runtimes."""
    times = []
    result = None
    for i in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"  Run {i+1}: {dt*1e3:.2f} ms")
    avg_time = sum(times) / repeat
    print(f"  Average time: {avg_time*1e3:.2f} ms\n")
    return result, times


# -------------------------------------------------------------------------------------------------------------------- #
# Define objective function
# -------------------------------------------------------------------------------------------------------------------- #
def objective(P_flat):
    """
    Compute arclength of a NURBS curve given flattened control point coordinates.
    
    Parameters
    ----------
    P_flat : array, shape (10,) for 5 control points in 2D
        Flattened control point coordinates [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]
        
    Returns
    -------
    arclength : float
        Total arclength of the curve
    """
    # Reshape to (2, 5)
    P = P_flat.reshape(2, 5)
    
    # Define curve parameters
    p = 3  # degree
    
    # Create curve with explicit knots
    curve = nrb.NurbsCurve(control_points=P, degree=p)
    
    # Compute arclength
    arclength = curve.get_arclength(u1=0.0, u2=1.0, tol=1e-6)
    
    return arclength




# -------------------------------------------------------------------------------------------------------------------- #
# Test 1: Direct evaluation
# -------------------------------------------------------------------------------------------------------------------- #
print("Test 1: Direct evaluation")
print("-" * 40)

P_init = jnp.array([
    [0.20, 0.40, 0.80, 0.60, 0.40],
    [0.50, 0.70, 0.60, 0.20, 0.20]
])
P_flat = P_init.flatten()

try:
    result, times_eval = timed_eval(objective, P_flat)
    print(f"Direct call successful")
    print(f"Arclength: {result:.6f}\n")
except Exception as e:
    print(f"Direct call failed: {e}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Test 2: JIT compilation
# -------------------------------------------------------------------------------------------------------------------- #
print("Test 2: JIT compilation")
print("-" * 40)

try:
    objective_jit = jax.jit(objective)
    # Warm-up (compilation)
    _ = objective_jit(P_flat).block_until_ready()
    print("JIT warm-up complete\n")

    result_jit, times_jit = timed_eval(objective_jit, P_flat)
    print(f"JIT call successful")
    print(f"Arclength: {result_jit:.6f}\n")
except Exception as e:
    print(f"JIT compilation or call failed: {e}\n")



# -------------------------------------------------------------------------------------------------------------------- #
# Test 3: Gradient computation (forward & reverse on compiled objective)
# -------------------------------------------------------------------------------------------------------------------- #
print("Test 3: Gradient (autodiff)")
print("-" * 40)

def try_grad(method_name, grad_fn):
    print(f"{method_name}:")
    try:
        gradient, _ = timed_eval(grad_fn, P_flat)
        print(f"  Gradient shape: {gradient.shape}")
        print(f"  Gradient norm: {jnp.linalg.norm(gradient):.6f}")
        print(f"  First 5 elements: {gradient[:5]}\n")
    except Exception as e:
        print(f"  {method_name} failed: {e}\n")


# Build gradient functions from the *compiled* objective
grad_fwd = jax.jit(jax.jacfwd(objective_jit))
grad_rev = jax.jit(jax.jacrev(objective_jit))

# Forward-mode
try_grad("Forward-mode (jacfwd + jit)", grad_fwd)

# Reverse-mode
try_grad("Reverse-mode (jacrev + jit)", grad_rev)
