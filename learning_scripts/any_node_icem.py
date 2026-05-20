import os, time, sys
import numpy as np
import torch

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # List of experiment configurations to run
    # Each entry: (controlled_node_index, target_position)
    "experiments": [
        {"target_index": 20, "target_position": [0.2, 0.2]},
        {"target_index": 40, "target_position": [0.2, 0.2]},
        {"target_index": 60, "target_position": [-0.05, 0.1]},
        {"target_index": 80, "target_position": [-0.05, 0.1]},
    ],
    
    # Optimization parameters
    "T": 101,                       # Number of time steps
    "iteration_number": 1,         # ICEM iterations
    "loss_threshold": 1e-20,
    
    # ICEM parameters
    "popsize": 10,                 # Population size
    "elite_frac": 0.3,              # Fraction of elites
    "alpha": 0.25,                  # Smoothing for mean/std updates
    "init_std": None,               # Initial std (None = 0.5 * u_max)
    "min_std": 1e-2,                # Minimum std floor
    "noise_beta": 2.0,              # Temporal correlation of ICEM samples
    "elite_keep_frac": 1.0,         # Reuse fraction of current elites
    "population_decay": 1.0,        # Population decay across iterations
    "keep_best": True,              # Reinsert current global best sequence
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


def parameters_to_vector(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


@torch.no_grad()
def set_params_from_vector_(params, vec):
    offset = 0
    for p in params:
        n = p.numel()
        p.copy_(vec[offset:offset + n].view_as(p))
        offset += n


def sample_colored_noise(
    rng,
    batch_size: int,
    dim: int,
    beta: float = 3.0,
    eps: float = 1e-8,
):
    noise = rng.standard_normal((batch_size, dim))
    if dim <= 1 or beta <= 0.0:
        return noise

    freqs = np.fft.rfftfreq(dim)
    spectral_scale = np.ones_like(freqs)
    nonzero = freqs > 0
    spectral_scale[nonzero] = 1.0 / np.power(freqs[nonzero], beta / 2.0)

    spectrum = np.fft.rfft(noise, axis=1)
    colored = np.fft.irfft(spectrum * spectral_scale[None, :], n=dim, axis=1)
    colored -= colored.mean(axis=1, keepdims=True)
    colored_std = colored.std(axis=1, keepdims=True)
    colored /= np.maximum(colored_std, eps)
    return colored


def build_icem_population(
    rng,
    mu: np.ndarray,
    std: np.ndarray,
    base_popsize: int,
    iteration: int,
    noise_beta: float = 3.0,
    population_decay: float = 1.0,
    prev_elites: np.ndarray | None = None,
    best_theta: np.ndarray | None = None,
    keep_best: bool = True,
):
    dim = mu.shape[0]
    curr_popsize = max(1, int(np.ceil(base_popsize * (population_decay ** iteration))))

    parts = []
    remaining = curr_popsize
    reused_count = 0

    if keep_best and best_theta is not None and remaining > 0:
        parts.append(best_theta[None, :].copy())
        remaining -= 1
        reused_count += 1

    if prev_elites is not None and prev_elites.size > 0 and remaining > 0:
        elite_count = min(prev_elites.shape[0], remaining)
        parts.append(prev_elites[:elite_count].copy())
        remaining -= elite_count
        reused_count += elite_count

    if remaining > 0:
        eps = sample_colored_noise(rng, remaining, dim, beta=noise_beta)
        fresh = mu[None, :] + std[None, :] * eps
        parts.insert(0, fresh)

    population = np.concatenate(parts, axis=0)
    return population, reused_count


# =============================================================================
# Black-box rollout loss for a given OPEN-LOOP u_seq (T,2)
# =============================================================================
def rollout_loss_u_seq(
    sim_manager,
    u_seq: np.ndarray,        # (T,2)
    targets: np.ndarray,      # (num_targets, 2)
    target_indices: list,     # list of node indices
    dlam: float,
    fail_loss: float = 1e6,
):
    """
    Dynamics:
      dx1 = u[k,0]*dlam applied to x of boundary nodes [0,1]
      dx2 = u[k,1]*dlam applied to x of boundary nodes [-2,-1]
    Loss:
      sum over all targets: 0.5 * ||x_node(target_index) - target||^2
    """
    sim_manager.resetSim()
    verts0 = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()  # (8,)
    final_vertices_flat = None
    T = u_seq.shape[0]
    
    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam
        v0 = xb_k[0:2].copy()
        v1 = xb_k[2:4].copy()
        v2 = xb_k[4:6].copy()
        v3 = xb_k[6:8].copy()
        # only move x
        v0[0] += dx1
        v1[0] += dx1
        v2[0] += dx2
        v3[0] += dx2
        xb_k = np.hstack((v0, v1, v2, v3))
        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        try:
            sim_manager.step()
        except Exception:
            return float(fail_loss), None
        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        final_vertices_flat = verts_xy.reshape(-1)
    
    # Compute loss - sum over all controlled nodes
    L_total = 0.0
    for idx, target in zip(target_indices, targets):
        v_f = final_vertices_flat.reshape(-1, 2)[idx]
        dv = v_f - target
        L_total += 0.5 * float(dv @ dv)
    
    return L_total, final_vertices_flat


def rollout_collect_vertices_u_seq(
    sim_manager,
    u_seq: np.ndarray,        # (T,2)
    dlam: float,
):
    """Roll out and return vertices over time for visualization: (T, N, 2)."""
    sim_manager.resetSim()
    verts0 = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()
    traj = []
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
        traj.append(verts_xy)
    
    return np.stack(traj, axis=0)  # (T,N,2)


@torch.no_grad()
def loss_only_forward(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,
    sim_manager,
    targets: np.ndarray,
    target_indices: list,
    dlam: float,
    fail_loss: float = 1e6,
):
    policy_model.eval()
    T = int(lams.numel())
    u_seq = policy_model(lams.view(T, 1)).cpu().numpy()
    loss, _ = rollout_loss_u_seq(
        sim_manager, u_seq, targets, target_indices, dlam, fail_loss=fail_loss
    )
    return float(loss)


# =============================================================================
# ICEM optimizer for open-loop u_seq in R^{T x 2}
# =============================================================================
def icem_optimize_u_seq(
    sim_manager,
    T: int,
    targets: np.ndarray,          # (num_targets, 2)
    target_indices: list,         # list of node indices
    dlam: float,
    u_max: np.ndarray,            # (2,) or scalar
    popsize: int = 128,
    elite_frac: float = 0.3,
    icem_iters: int = 20,
    alpha: float = 0.25,          # smoothing for mean/std updates
    init_std: float | None = None,
    min_std: float = 1e-2,
    noise_beta: float = 2.0,
    elite_keep_frac: float = 1.0,
    population_decay: float = 1.0,
    keep_best: bool = True,
    seed: int = 42,
    fail_loss: float = 1e6,
    loss_threshold: float = 1e-7,
    warm_start_mu: np.ndarray | None = None,   # (T,2)
    warm_start_std: np.ndarray | None = None,  # (T,2)
):
    """
    ICEM over policy parameters using colored-noise sampling and elite reuse.
    Returns:
      best_u  : (T,2) generated by the best policy parameters
      best_L  : float
      mu, std : final distribution over flattened params
      hist    : dict with per-iter logs
      epoch_dt_hist : list of epoch times
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

    best_u = None
    best_theta = None
    best_L = float("inf")
    best_iter_hist = []
    best_so_far_hist = []
    std_mean_hist = []
    epoch_dt_hist = []
    popsize_hist = []
    reused_hist = []
    prev_elites = np.empty((0, seq_dim), dtype=np.float64)
    
    for it in range(icem_iters):
        t0 = time.perf_counter()
        Theta, reused_count = build_icem_population(
            rng=rng,
            mu=mu,
            std=std,
            base_popsize=popsize,
            iteration=it,
            noise_beta=noise_beta,
            population_decay=population_decay,
            prev_elites=prev_elites,
            best_theta=best_theta,
            keep_best=keep_best,
        )
        Theta = np.clip(Theta, -u_max_flat[None, :], u_max_flat[None, :])
        curr_popsize = Theta.shape[0]
        K = max(1, int(np.ceil(curr_popsize * elite_frac)))
        
        # evaluate
        losses = np.empty((curr_popsize,), dtype=np.float64)
        for j in range(curr_popsize):
            losses[j], _ = rollout_loss_u_seq(
                sim_manager,
                Theta[j].reshape(seq_shape),
                targets,
                target_indices,
                dlam,
                fail_loss=fail_loss,
            )
        
        j_best = int(np.argmin(losses))
        best_iter = float(losses[j_best])
        if best_iter < best_L:
            best_L = best_iter
            best_theta = Theta[j_best].copy()
            best_u = best_theta.reshape(seq_shape).copy()
        
        # elites
        elite_idx = np.argsort(losses)[:K]
        elites = Theta[elite_idx]
        elite_mu = elites.mean(axis=0)
        elite_std = elites.std(axis=0)
        elite_reuse_count = max(1, int(np.ceil(K * elite_keep_frac)))
        prev_elites = elites[:elite_reuse_count].copy()
        
        # smooth update + floor on std
        mu = (1 - alpha) * mu + alpha * elite_mu
        std = (1 - alpha) * std + alpha * elite_std
        std = np.maximum(std, min_std)
        
        epoch_dt = time.perf_counter() - t0
        epoch_dt_hist.append(epoch_dt)
        
        best_iter_hist.append(best_iter)
        best_so_far_hist.append(best_L)
        std_mean_hist.append(float(std.mean()))
        popsize_hist.append(curr_popsize)
        reused_hist.append(reused_count)
        
        print(
            f"Epoch {it:03d} | Loss {best_L:.6e} | best_iter {best_iter:.3e} "
            f"| std_mean {std.mean():.3e} | pop {curr_popsize:03d} | reuse {reused_count:03d} "
            f"| dt {epoch_dt*1e3:.1f} ms"
        )
        
        # Early stopping
        if best_L < loss_threshold:
            print(f"\nReached loss threshold at epoch {it}")
            break
    
    hist = {
        "best_iter": np.array(best_iter_hist),
        "best_so_far": np.array(best_so_far_hist),
        "std_mean": np.array(std_mean_hist),
        "popsize": np.array(popsize_hist),
        "reused": np.array(reused_hist),
    }
    return best_u, best_L, mu.reshape(seq_shape), std.reshape(seq_shape), hist, epoch_dt_hist


if __name__ == "__main__":
    configure_threads(1)

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
    sim_manager.resetSim()

    verts_init = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts_init.shape[0]

    # Load configuration parameters
    experiments = CONFIG["experiments"]
    
    T = CONFIG["T"]
    iteration_number = CONFIG["iteration_number"]
    loss_threshold = CONFIG["loss_threshold"]
    popsize = CONFIG["popsize"]
    elite_frac = CONFIG["elite_frac"]
    alpha = CONFIG["alpha"]
    init_std = CONFIG["init_std"]
    min_std = CONFIG["min_std"]
    noise_beta = CONFIG["noise_beta"]
    elite_keep_frac = CONFIG["elite_keep_frac"]
    population_decay = CONFIG["population_decay"]
    keep_best = CONFIG["keep_best"]
    
    # Time discretization
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    dlam = float(lams_np[1] - lams_np[0])
    
    # Bounds on u
    u_max = np.array([0.05 / dlam, 0.05 / dlam], dtype=np.float64)
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Multi-Experiment Point-to-Point Control (ICEM)")
    print(f"  Number of experiments: {len(experiments)}")
    print(f"  Number of time steps: {T}")
    print(f"  ICEM iterations per run: {iteration_number}")
    print(f"  Population size: {popsize}")
    print(f"{'='*60}\n")

    # Storage for all results
    all_results = []
    all_loss_histories = []  # Store loss history for each experiment
    
    total_start_time = time.perf_counter()
    
    # Loop over experiments
    for exp_idx, exp_config in enumerate(experiments):
        node_index = exp_config["target_index"]
        target_position = np.array(exp_config["target_position"], dtype=np.float64)
        
        print(f"\n{'='*60}")
        print(f"Experiment {exp_idx+1}/{len(experiments)}: Node {node_index} -> {target_position.tolist()}")
        print(f"{'='*60}")
        
        run_start_time = time.perf_counter()
        
        best_u, best_loss, mu, std, hist, epoch_dt_hist = icem_optimize_u_seq(
            sim_manager=sim_manager,
            T=T,
            targets=target_position.reshape(1, 2),  # Single target
            target_indices=[node_index],            # Single node
            dlam=dlam,
            u_max=u_max,
            popsize=popsize,
            elite_frac=elite_frac,
            icem_iters=iteration_number,
            alpha=alpha,
            init_std=init_std,
            min_std=min_std,
            noise_beta=noise_beta,
            elite_keep_frac=elite_keep_frac,
            population_decay=population_decay,
            keep_best=keep_best,
            seed=42,
            fail_loss=1e6,
            loss_threshold=loss_threshold,
        )
        
        run_time = time.perf_counter() - run_start_time
        avg_epoch_time = np.mean(epoch_dt_hist)
        
        exp_result = {
            "node_index": node_index,
            "target_position": target_position.tolist(),
            "loss": best_loss,
            "time": run_time,
            "epoch_time": avg_epoch_time,
        }
        all_results.append(exp_result)
        all_loss_histories.append(hist["best_so_far"].tolist())  # Store best_so_far loss at each iteration
        
        print(f"\nExperiment {exp_idx+1} Summary:")
        print(f"  Loss: {best_loss:.6e}")
        print(f"  Time: {run_time:.3f}s")
    
    total_time = time.perf_counter() - total_start_time
    
    # Compute overall statistics
    all_losses = [r["loss"] for r in all_results]
    mean_loss = np.mean(all_losses)
    all_times = [r["time"] for r in all_results]
    mean_time = np.mean(all_times)
    
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"\nOverall Statistics:")
    print(f"  Mean loss: {mean_loss:.6e}")
    print(f"  Mean time per experiment: {mean_time:.3f}s")
    print(f"{'='*60}\n")

    # Save results to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "any_node_icem.txt")
    
    with open(output_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("ICEM Multi-Experiment Results Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Number of experiments: {len(experiments)}\n")
        f.write(f"  Number of time steps: {T}\n")
        f.write(f"  ICEM iterations per run: {iteration_number}\n")
        f.write(f"  Population size: {popsize}\n")
        f.write(f"  Elite fraction: {elite_frac}\n")
        f.write(f"  Alpha: {alpha}\n")
        f.write(f"  Noise beta: {noise_beta}\n")
        f.write(f"  Elite keep frac: {elite_keep_frac}\n")
        f.write(f"  Population decay: {population_decay}\n")
        f.write(f"  Keep best: {keep_best}\n")
        f.write(f"  Min std: {min_std}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Per-Experiment Results:\n")
        f.write("-"*60 + "\n")
        for i, r in enumerate(all_results):
            f.write(f"\nExperiment {i+1}:\n")
            f.write(f"  Node index: {r['node_index']}\n")
            f.write(f"  Target position: {r['target_position']}\n")
            f.write(f"  Loss: {r['loss']:.10e}\n")
            f.write(f"  Time: {r['time']:.6f}s\n")
            f.write(f"  Epoch time: {r['epoch_time']*1e3:.3f}ms\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Overall Statistics:\n")
        f.write("="*60 + "\n")
        f.write(f"  Total time: {total_time:.6f}s\n")
        f.write(f"  Mean loss: {mean_loss:.10e}\n")
        f.write(f"  Mean time per experiment: {mean_time:.6f}s\n")
        f.write("="*60 + "\n")
    
    print(f"Results saved to: {output_file}")
    
    # Save per-step loss for each case to txt files
    for exp_idx, (exp_config, loss_hist) in enumerate(zip(experiments, all_loss_histories)):
        node_index = exp_config["target_index"]
        loss_file = os.path.join(script_dir, f"any_node_icem_case{exp_idx}_node{node_index}_loss.txt")
        with open(loss_file, "w") as f:
            f.write("# Per-step loss history for any_node_icem\n")
            f.write(f"# Case {exp_idx}: Node {node_index} -> {exp_config['target_position']}\n")
            f.write("# Step, Loss\n")
            for step, loss_val in enumerate(loss_hist):
                f.write(f"{step}, {loss_val:.10e}\n")
        print(f"Loss history saved to: {loss_file}")
