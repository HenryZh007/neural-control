import os, time, copy, random, sys
import numpy as np
import torch

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model
from trajectory import generate_trajectory, get_trajectory_description


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # Trajectory types to test: 'sin', 'cos', 'triangle', 'semicircle', 'square'
    "trajectory_types": ["square", "cos", "triangle", "semicircle"],
    
    # Trajectory-specific parameters
    "trajectory_params": {
        # For sin/cos trajectories
        "amplitude": 0.05,          # Wave amplitude
        "frequency": 3.0,           # Wave frequency (number of cycles)
        
        # For triangle wave
        "period": 0.5,              # Period of triangle wave
        
        # For semicircle
        "radius": 0.25,             # Radius of semicircle
        "direction": "down",        # 'up' or 'down'
        
        # For square wave
        "square_amplitude": 0.12,   # Amplitude of square wave
        "num_segments": 10,         # Number of segments
    },
    
    # Target node index (which node to track)
    "target_index": 50,
    
    # Optimization parameters
    "T": 101,                       # Number of time steps
    "iteration_number": 2,          # CEM iterations (same as epoch count in noMPC)
    "loss_threshold": 1e-7,
    
    # CEM parameters
    "popsize": 10,                 # Population size
    "elite_frac": 0.3,              # Fraction of elite samples
    "alpha": 0.25,                  # Smoothing for mean/std updates
    "min_std": 1e-2,                # Minimum standard deviation
}


# =============================================================================
# Thread safety / stability
# =============================================================================
def configure_threads(num_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def parameters_to_vector(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


@torch.no_grad()
def set_params_from_vector_(params, vec):
    offset = 0
    for p in params:
        n = p.numel()
        p.copy_(vec[offset:offset + n].view_as(p))
        offset += n


# =============================================================================
# Simulator helper functions
# =============================================================================
def resetSim(sim_manager):
    sim_manager.resetSim()


# =============================================================================
# Black-box rollout loss for trajectory tracking
# =============================================================================
def rollout_loss_trajectory(
    sim_manager,
    u_seq: np.ndarray,        # (T, 2)
    target: np.ndarray,       # (T, 2) - target trajectory
    target_index: int,
    dlam: float,
    fail_loss: float = 1e6,
):
    """
    Compute trajectory tracking loss.
    
    Parameters
    ----------
    u_seq : np.ndarray
        Control sequence of shape (T, 2)
    target : np.ndarray
        Target trajectory of shape (T, 2)
    target_index : int
        Index of the node to track
    dlam : float
        Time step size
    fail_loss : float
        Loss value to return if simulation fails
        
    Returns
    -------
    L_total : float
        Trajectory tracking loss
    """
    sim_manager.resetSim()
    verts0 = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()  # (8,)
    
    T = u_seq.shape[0]
    L_total = 0.0
    
    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam
        
        v0 = xb_k[0:2].copy()
        v1 = xb_k[2:4].copy()
        v2 = xb_k[4:6].copy()
        v3 = xb_k[6:8].copy()
        
        # Only move x
        v0[0] += dx1
        v1[0] += dx1
        v2[0] += dx2
        v3[0] += dx2
        
        xb_k = np.hstack((v0, v1, v2, v3))
        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        
        try:
            sim_manager.step()
        except Exception:
            return float(fail_loss)
        
        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        
        # Compute tracking loss at this time step
        v_i = verts_xy[target_index]
        dv = v_i - target[i]
        L_total += 0.5 * float(dv @ dv) * dlam
    
    return L_total


def rollout_collect_vertices(
    sim_manager,
    u_seq: np.ndarray,        # (T, 2)
    dlam: float,
):
    """Roll out and return vertices over time for visualization: list of (N*2,) arrays."""
    sim_manager.resetSim()
    verts0 = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()
    
    vertices_list = []
    T = u_seq.shape[0]
    
    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam
        
        v0 = xb_k[0:2].copy()
        v1 = xb_k[2:4].copy()
        v2 = xb_k[4:6].copy()
        v3 = xb_k[6:8].copy()
        
        v0[0] += dx1
        v1[0] += dx1
        v2[0] += dx2
        v3[0] += dx2
        
        xb_k = np.hstack((v0, v1, v2, v3))
        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        sim_manager.step()
        
        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        vertices_list.append(verts_xy.reshape(-1).copy())
    
    return vertices_list


@torch.no_grad()
def loss_only_forward(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,
    sim_manager,
    target: np.ndarray,
    target_index: int,
    dlam: float,
    fail_loss: float = 1e6,
):
    policy_model.eval()
    T = int(lams.numel())
    u_seq = policy_model(lams.view(T, 1)).cpu().numpy()
    return rollout_loss_trajectory(
        sim_manager, u_seq, target, target_index, dlam, fail_loss=fail_loss
    )


# =============================================================================
# CEM optimizer for trajectory tracking
# =============================================================================
def cem_optimize_trajectory(
    sim_manager,
    T: int,
    target: np.ndarray,       # (T, 2) - target trajectory
    target_index: int,
    dlam: float,
    u_max: np.ndarray,        # (2,) or scalar
    popsize: int = 128,
    elite_frac: float = 0.1,
    cem_iters: int = 20,
    alpha: float = 0.25,      # smoothing for mean/std updates
    init_std: float | None = None,
    min_std: float = 1e-2,
    seed: int = 42,
    fail_loss: float = 1e6,
    warm_start_mu: np.ndarray | None = None,   # (T, 2)
    warm_start_std: np.ndarray | None = None,  # (T, 2)
    loss_threshold: float = 1e-7,
):
    """
    Diagonal-Gaussian CEM over policy parameters for trajectory tracking.
    
    Returns
    -------
    best_u : np.ndarray
        Best control sequence (T, 2)
    best_L : float
        Best loss value
    mu : np.ndarray
        Final mean of distribution (T, 2)
    std : np.ndarray
        Final std of distribution (T, 2)
    hist : dict
        Per-iteration logs
    """
    rng = np.random.default_rng(seed)

    u_max = np.asarray(u_max, dtype=np.float64)
    if u_max.size == 1:
        u_max = np.array([float(u_max), float(u_max)], dtype=np.float64)

    seq_shape = (T, int(u_max.size))
    seq_dim = int(np.prod(seq_shape))
    u_max_flat = np.tile(u_max, T)

    if warm_start_mu is not None and warm_start_mu.size == seq_dim:
        mu = warm_start_mu.astype(np.float64).reshape(-1).copy()
    else:
        mu = np.zeros((seq_dim,), dtype=np.float64)

    if warm_start_std is not None and warm_start_std.size == seq_dim:
        std = warm_start_std.astype(np.float64).reshape(-1).copy()
    else:
        if init_std is None:
            init_std = 0.1
        std = np.ones((seq_dim,), dtype=np.float64) * float(init_std)

    K = max(1, int(popsize * elite_frac))

    best_u = None
    best_theta = None
    best_L = float("inf")
    best_positions = None  # Store best positions at each time step
    position_history = []  # Store best-so-far positions at each epoch
    
    best_iter_hist = []
    best_so_far_hist = []
    std_mean_hist = []
    epoch_dt_hist = []
    
    for it in range(cem_iters):
        t0 = time.perf_counter()

        eps = rng.standard_normal(size=(popsize, seq_dim))
        Theta = mu[None, :] + std[None, :] * eps
        Theta = np.clip(Theta, -u_max_flat[None, :], u_max_flat[None, :])

        losses = np.empty((popsize,), dtype=np.float64)
        for j in range(popsize):
            losses[j] = rollout_loss_trajectory(
                sim_manager,
                Theta[j].reshape(seq_shape),
                target,
                target_index,
                dlam,
                fail_loss=fail_loss,
            )

        j_best = int(np.argmin(losses))
        best_iter = float(losses[j_best])

        if best_iter < best_L:
            best_L = best_iter
            best_theta = Theta[j_best].copy()
            best_u = best_theta.reshape(seq_shape).copy()
            # Collect middle point positions for the best control sequence
            vertices_list = rollout_collect_vertices(sim_manager, best_u, dlam)
            best_positions = []
            for v_flat in vertices_list:
                v_xy = v_flat.reshape(-1, 2)
                best_positions.append(v_xy[target_index].tolist())

        if best_positions is not None:
            position_history.append([pos.copy() for pos in best_positions])
        
        elite_idx = np.argsort(losses)[:K]
        elites = Theta[elite_idx]
        elite_mu = elites.mean(axis=0)
        elite_std = elites.std(axis=0)

        mu = (1 - alpha) * mu + alpha * elite_mu
        std = (1 - alpha) * std + alpha * elite_std
        std = np.maximum(std, min_std)
        
        epoch_dt = time.perf_counter() - t0
        
        best_iter_hist.append(best_iter)
        best_so_far_hist.append(best_L)
        std_mean_hist.append(float(std.mean()))
        epoch_dt_hist.append(epoch_dt)
        
        print(f"Epoch {it:03d} | Loss {best_L:.6e} | best_iter {best_iter:.3e} | std_mean {std.mean():.3e} | dt {epoch_dt*1e3:.1f} ms")
    
    hist = {
        "best_iter": np.array(best_iter_hist),
        "best_so_far": np.array(best_so_far_hist),
        "std_mean": np.array(std_mean_hist),
        "epoch_dt": np.array(epoch_dt_hist),
    }
    return best_u, best_L, mu.reshape(seq_shape), std.reshape(seq_shape), hist, position_history


if __name__ == "__main__":
    configure_threads(1)
    set_seed(42)
    device = torch.device("cpu")

    # Simulator setup
    sim_manager = py_der.SimulationManager()
    sim_manager.configure({
        "youngM": 1e5,
        "Poisson": 0.5,
        "density": 1000,
        "deltaTime": 0.01,
        "totalTime": 10.0,
        "gVector": np.array([0, 0, -0.0]),
        "viscosity": 0.000,
        "tol": 1e-4,
        "maxIter": 10000,
        "stol": 1e-4,
        "rodRadius": 1e-3,
        "geometry_file": "vertices.txt",
        "d_h": 0.001,
        "col_limit": 0.01,
        "k_scaler": 1.0,
    })

    controller_type = [0, 0, 0, 0]
    control_dofs = [0, 1, 99, 100]
    control_info = np.array([controller_type, control_dofs]).T
    sim_manager.defineController(control_info)
    resetSim(sim_manager)

    verts_init = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts_init.shape[0]

    # Load configuration parameters
    trajectory_types = CONFIG["trajectory_types"]
    trajectory_params = CONFIG["trajectory_params"]
    target_index = CONFIG["target_index"]
    T = CONFIG["T"]
    iteration_number = CONFIG["iteration_number"]
    loss_threshold = CONFIG["loss_threshold"]
    popsize = CONFIG["popsize"]
    elite_frac = CONFIG["elite_frac"]
    alpha = CONFIG["alpha"]
    min_std = CONFIG["min_std"]

    # Time discretization
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    dlam = float(lams_np[1] - lams_np[0])

    # Bounds on u (same as in noMPC version)
    u_max = np.array([0.1 / dlam, 0.1 / dlam], dtype=np.float64)

    # Store results for all trajectories
    all_results = []

    print(f"\n{'='*60}")
    print(f"Testing {len(trajectory_types)} trajectory types with CEM")
    print(f"{'='*60}\n")

    for traj_idx, trajectory_type in enumerate(trajectory_types):
        # Reset seed for each trajectory to ensure fair comparison
        set_seed(42)
        
        # Generate target trajectory
        middle_node = verts_init[target_index, :].copy()
        target = generate_trajectory(trajectory_type, middle_node, T, trajectory_params)
        traj_desc = get_trajectory_description(trajectory_type, trajectory_params)

        # Print configuration
        print(f"\n{'='*60}")
        print(f"[{traj_idx+1}/{len(trajectory_types)}] Trajectory: {traj_desc}")
        print(f"  Target node index: {target_index}")
        print(f"  Number of time steps: {T}")
        print(f"  Number of CEM iterations: {iteration_number}")
        print(f"  Population size: {popsize}")
        print(f"  Elite fraction: {elite_frac}")
        print(f"{'='*60}\n")

        # Run CEM optimization
        total_start_time = time.perf_counter()

        best_u, best_loss, mu, std, hist, position_history = cem_optimize_trajectory(
            sim_manager=sim_manager,
            T=T,
            target=target,
            target_index=target_index,
            dlam=dlam,
            u_max=u_max,
            popsize=popsize,
            elite_frac=elite_frac,
            cem_iters=iteration_number,
            alpha=alpha,
            init_std=None,
            min_std=min_std,
            seed=42,
            fail_loss=1e6,
            loss_threshold=loss_threshold,
        )

        total_time = time.perf_counter() - total_start_time
        avg_epoch_time = np.mean(hist["epoch_dt"])

        print(f"\n{'='*60}")
        print(f"[{trajectory_type}] CEM optimization completed!")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Best loss: {best_loss:.6e}")
        print(f"  Average epoch time: {avg_epoch_time*1e3:.3f} ms")
        print(f"{'='*60}\n")

        # Store results
        best_epoch = int(np.argmin(hist["best_so_far"]))
        all_results.append({
            "trajectory_type": trajectory_type,
            "trajectory_desc": traj_desc,
            "total_time": total_time,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "avg_epoch_time": avg_epoch_time,
            "total_epochs": len(hist["best_iter"]),
            "position_history": position_history,
        })

    # Save all results to current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "middle_tracking_cem.txt")

    # Save summary to txt file
    with open(output_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Summary Table (CEM):\n")
        f.write("="*60 + "\n")
        f.write(f"CEM population size: {popsize}\n")
        f.write(f"CEM elite fraction: {elite_frac}\n")
        f.write("-"*65 + "\n")
        f.write(f"{'Trajectory':<20} {'Total Time (s)':<15} {'Best Loss':<15} {'Avg Epoch (ms)':<15}\n")
        f.write("-"*65 + "\n")
        for result in all_results:
            f.write(f"{result['trajectory_type']:<20} {result['total_time']:<15.4f} {result['best_loss']:<15.6e} {result['avg_epoch_time']*1e3:<15.3f}\n")
        f.write("-"*65 + "\n")
        
        # Compute averages
        avg_total_time = np.mean([r['total_time'] for r in all_results])
        avg_best_loss = np.mean([r['best_loss'] for r in all_results])
        avg_epoch_time_all = np.mean([r['avg_epoch_time'] for r in all_results])
        
        f.write(f"{'AVERAGE':<20} {avg_total_time:<15.4f} {avg_best_loss:<15.6e} {avg_epoch_time_all*1e3:<15.3f}\n")
        f.write("="*60 + "\n")

    print(f"\n{'='*60}")
    print(f"Overall Averages (CEM):")
    print(f"  Average total time: {avg_total_time:.4f} s")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average epoch time: {avg_epoch_time_all*1e3:.3f} ms")
    print(f"{'='*60}")
    print(f"\nAll results saved to {output_file}")
    
    # Save per-step middle point positions for each case to txt files
    for traj_idx, result in enumerate(all_results):
        traj_type = result['trajectory_type']
        position_hist = result['position_history']
        pos_file = os.path.join(script_dir, f"middle_tracking_cem_case{traj_idx}_{traj_type}_positions.txt")
        with open(pos_file, "w") as f:
            f.write("# Per-step middle point position history for middle_tracking_cem\n")
            f.write(f"# Case {traj_idx}: Trajectory type = {traj_type}\n")
            f.write(f"# Target node index: {target_index}\n")
            f.write("# Format: Epoch, TimeStep, X, Y\n")
            for epoch_idx, epoch_positions in enumerate(position_hist):
                for step_idx, (x, y) in enumerate(epoch_positions):
                    f.write(f"{epoch_idx}, {step_idx}, {x:.10e}, {y:.10e}\n")
        print(f"Position history saved to: {pos_file}")
