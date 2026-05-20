import os, time, copy, sys
import numpy as np
import torch

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model, to_3d, translate_and_rotate_segment


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # List of cases: each case is a dict with "initial" and "target" file paths
    "cases": [
        {"initial": "C_initial.txt", "target": "targets/target_C.txt"},
        {"initial": "U_initial.txt", "target": "targets/target_U.txt"},
        {"initial": "M_initial.txt", "target": "targets/target_M.txt"},
    ],
    
    # CEM parameters
    "popsize": 10,                 # Population size
    "elite_frac": 0.3,              # Fraction of elites
    "cem_iters": 2,              # CEM iterations (equivalent to max_epochs)
    "alpha": 0.25,                  # Smoothing for mean/std updates
    "min_std": 1e-2,                # Minimum standard deviation
    
    # Time discretization
    "T": 21,                        # Number of time steps
    
    # Control bounds (will be divided by dlam)
    "bounds_xy": 0.05,
    "bounds_a": 0.3,
}

# =============================================================================
# Thread safety / stability (avoid BLAS oversubscription + nondeterminism)
# =============================================================================
def configure_threads(num_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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
# Black-box rollout loss for a given OPEN-LOOP u_seq (T, 3)
# =============================================================================
def rollout_loss_u_seq(
    sim_manager,
    u_seq: np.ndarray,        # (T, 3) - dx, dy, da
    target: np.ndarray,       # target vertices
    dlam: float,
    fail_loss: float = 1e6,
):
    """
    Dynamics:
      Apply translate_and_rotate_segment to boundary nodes [-2, -1]
    Loss:
      Curvature loss + stretch loss (same as letter_curve_noMPC)
    """
    sim_manager.resetSim()
    
    # Compute target curvature
    kap_target = sim_manager.compute_curvature(to_3d(target))
    
    verts0 = np.asarray(sim_manager.getAllVertices()).copy()
    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()
    
    v0_fixed = xb_k[0:2].copy()
    v1_fixed = xb_k[2:4].copy()
    
    T = u_seq.shape[0]
    for i in range(T):
        uk = u_seq[i]
        dx, dy, da = uk * dlam
        
        v2 = xb_k[4:6]
        v3 = xb_k[6:8]
        v2_1, v3_1 = translate_and_rotate_segment(v2, v3, dx, dy, da)
        xb_k = np.hstack([v0_fixed, v1_fixed, v2_1, v3_1])
        
        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        try:
            sim_manager.step()
        except Exception:
            return float(fail_loss)
    
    # Compute loss (same as letter_curve_noMPC)
    coeff_b = np.array([[1e-3, 0.0], [0.0, 1e-3]])
    L_kap = sim_manager.compute_curvature_loss(kap_target, coeff_b)
    L_stretch = sim_manager.compute_stretch_loss(1.0)
    L_total = L_kap + L_stretch
    
    return float(L_total)


@torch.no_grad()
def loss_only_forward(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,
    sim_manager,
    target: np.ndarray,
    dlam: float,
    fail_loss: float = 1e6,
):
    policy_model.eval()
    T = int(lams.numel())
    u_seq = policy_model(lams.view(T, 1)).cpu().numpy()
    return rollout_loss_u_seq(
        sim_manager=sim_manager,
        u_seq=u_seq,
        target=target,
        dlam=dlam,
        fail_loss=fail_loss,
    )


# =============================================================================
# CEM optimizer for open-loop u_seq in R^{T x 3}
# =============================================================================
def cem_optimize_u_seq(
    sim_manager,
    T: int,
    target: np.ndarray,
    dlam: float,
    u_max: np.ndarray,            # (3,)
    popsize: int = 128,
    elite_frac: float = 0.1,
    cem_iters: int = 1000,
    alpha: float = 0.25,
    init_std: float | None = None,
    min_std: float = 1e-2,
    seed: int = 42,
    fail_loss: float = 1e6,
    case_idx: int = 0,
):
    """
    Diagonal-Gaussian CEM over policy parameters.

    Returns:
      best_u  : (T, 3) generated by the best policy parameters
      best_L  : float
      hist    : dict with per-iter logs
    """
    rng = np.random.default_rng(seed)
    u_max = np.asarray(u_max, dtype=np.float64)

    seq_shape = (T, int(u_max.size))
    seq_dim = int(np.prod(seq_shape))
    u_max_flat = np.tile(u_max, T)

    mu = np.zeros((seq_dim,), dtype=np.float64)
    if init_std is None:
        init_std = 0.1
    std = np.ones((seq_dim,), dtype=np.float64) * float(init_std)

    K = max(1, int(popsize * elite_frac))

    best_u = None
    best_theta = None
    best_L = float("inf")
    best_iter_hist = []
    best_so_far_hist = []
    std_mean_hist = []
    
    start_time = time.perf_counter()
    
    for it in range(cem_iters):
        t0 = time.perf_counter()

        eps = rng.standard_normal(size=(popsize, seq_dim))
        Theta = mu[None, :] + std[None, :] * eps
        Theta = np.clip(Theta, -u_max_flat[None, :], u_max_flat[None, :])

        losses = np.empty((popsize,), dtype=np.float64)
        for j in range(popsize):
            losses[j] = rollout_loss_u_seq(
                sim_manager=sim_manager,
                u_seq=Theta[j].reshape(seq_shape),
                target=target,
                dlam=dlam,
                fail_loss=fail_loss,
            )

        j_best = int(np.argmin(losses))
        best_iter = float(losses[j_best])

        if best_iter < best_L:
            best_L = best_iter
            best_theta = Theta[j_best].copy()
            best_u = best_theta.reshape(seq_shape).copy()

        elite_idx = np.argsort(losses)[:K]
        elites = Theta[elite_idx]
        elite_mu = elites.mean(axis=0)
        elite_std = elites.std(axis=0)

        mu = (1 - alpha) * mu + alpha * elite_mu
        std = (1 - alpha) * std + alpha * elite_std
        std = np.maximum(std, min_std)
        
        best_iter_hist.append(best_iter)
        best_so_far_hist.append(best_L)
        std_mean_hist.append(float(std.mean()))
        
        epoch_dt = time.perf_counter() - t0
        print(f"Case {case_idx:02d} | Iter {it:04d} | Loss: {best_L:.6f} | Iter Best: {best_iter:.6e} | Std: {std.mean():.6e} | Time: {epoch_dt:.3f}s")
    
    total_time = time.perf_counter() - start_time
    
    hist = {
        "best_iter": np.array(best_iter_hist),
        "best_so_far": np.array(best_so_far_hist),
        "std_mean": np.array(std_mean_hist),
        "total_time": total_time,
    }
    return best_u, best_L, mu.reshape(seq_shape), std.reshape(seq_shape), hist


def run_single_case(
    sim_manager,
    initial_file: str,
    target_file: str,
    config: dict,
    case_idx: int,
):
    """
    Run CEM optimization for a single case.
    
    Returns
    -------
    result : dict
        Contains total_time, best_loss, total_iters, etc.
    """
    # Reconfigure simulator with new geometry file
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
        "geometry_file": initial_file,
        "d_h": 0.001,
        "col_limit": 0.01,
        "k_scaler": 1.0,
    })

    # Controller : BC controller
    controller_type = [0, 0, 0, 0]
    control_dofs = [0, 1, 99, 100]
    control_info = np.array([controller_type, control_dofs]).T
    sim_manager.defineController(control_info)
    sim_manager.resetSim()

    # Load target
    target = np.loadtxt(target_file)
    target = target.reshape(-1)

    # CEM setup
    T = config["T"]
    cem_iters = config["cem_iters"]
    popsize = config["popsize"]
    elite_frac = config["elite_frac"]
    alpha = config["alpha"]
    min_std = config["min_std"]
    bounds_xy = config["bounds_xy"]
    bounds_a = config["bounds_a"]

    lams_np = np.linspace(0.0, 1.0, T).astype(np.float32)
    dlam = float(lams_np[1] - lams_np[0])

    u_max = np.array([bounds_xy/dlam, bounds_xy/dlam, bounds_a/dlam], dtype=np.float64)

    # Run CEM optimization
    start_time = time.perf_counter()
    
    best_u, best_L, mu, std, hist = cem_optimize_u_seq(
        sim_manager=sim_manager,
        T=T,
        target=target,
        dlam=dlam,
        u_max=u_max,
        popsize=popsize,
        elite_frac=elite_frac,
        cem_iters=cem_iters,
        alpha=alpha,
        init_std=None,
        min_std=min_std,
        seed=1234,
        fail_loss=1e6,
        case_idx=case_idx,
    )
    
    total_time = time.perf_counter() - start_time

    return {
        "initial_file": initial_file,
        "target_file": target_file,
        "total_time": total_time,
        "best_loss": best_L,
        "total_iters": cem_iters,
        "avg_iter_time": total_time / cem_iters if cem_iters > 0 else 0.0,
        "best_u": best_u,
        "loss_history": hist["best_so_far"].tolist(),
    }


if __name__ == "__main__":
    configure_threads(num_threads=1)
    set_seed(1234, deterministic=True)

    # Load configuration
    cases = CONFIG["cases"]
    
    # Create simulator
    sim_manager = py_der.SimulationManager()

    # Store results for all cases
    all_results = []

    print(f"\n{'='*70}")
    print(f"Running {len(cases)} cases with CEM")
    print(f"CEM iterations per case: {CONFIG['cem_iters']}")
    print(f"Population size: {CONFIG['popsize']}")
    print(f"{'='*70}\n")

    for case_idx, case in enumerate(cases):
        # Reset seed for each case to ensure fair comparison
        set_seed(1234)
        
        initial_file = case["initial"]
        target_file = case["target"]

        print(f"\n{'='*70}")
        print(f"[{case_idx+1}/{len(cases)}] Case: {os.path.basename(initial_file)} -> {os.path.basename(target_file)}")
        print(f"{'='*70}\n")

        result = run_single_case(
            sim_manager,
            initial_file,
            target_file,
            CONFIG,
            case_idx,
        )

        all_results.append(result)

        print(f"\n[Case {case_idx}] Completed!")
        print(f"  Total time: {result['total_time']:.3f} s")
        print(f"  Best loss: {result['best_loss']:.6e}")
        print(f"  Avg iter time: {result['avg_iter_time']:.3f} s")

    # Compute averages
    avg_total_time = np.mean([r['total_time'] for r in all_results])
    avg_best_loss = np.mean([r['best_loss'] for r in all_results])
    avg_iter_time = np.mean([r['avg_iter_time'] for r in all_results])

    # Save results to txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "letter_curve_cem.txt")

    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Summary Table (Letter Curve CEM Method):\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration: CEM iters={CONFIG['cem_iters']}, T={CONFIG['T']}, Popsize={CONFIG['popsize']}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Case':<6} {'Initial':<25} {'Target':<25} {'Time (s)':<12} {'Best Loss':<15}\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(all_results):
            init_name = os.path.basename(result['initial_file'])
            target_name = os.path.basename(result['target_file'])
            f.write(f"{i:<6} {init_name:<25} {target_name:<25} {result['total_time']:<12.4f} {result['best_loss']:<15.6e}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'AVG':<6} {'':<25} {'':<25} {avg_total_time:<12.4f} {avg_best_loss:<15.6e}\n")
        f.write("="*80 + "\n")
        f.write(f"\nAverage iteration time: {avg_iter_time:.4f} s\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"All Cases Completed!")
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Total cases: {len(all_results)}")
    print(f"  Average total time: {avg_total_time:.4f} s")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average iteration time: {avg_iter_time:.4f} s")
    print(f"{'='*70}")
    print(f"\nResults saved to {output_file}")
    
    # Save per-step loss for each case to txt files
    for case_idx, result in enumerate(all_results):
        init_name = os.path.basename(result['initial_file']).replace('.txt', '')
        loss_hist = result['loss_history']
        loss_file = os.path.join(script_dir, f"letter_curve_cem_case{case_idx}_{init_name}_loss.txt")
        with open(loss_file, "w") as f:
            f.write("# Per-step loss history for letter_curve_cem\n")
            f.write(f"# Case {case_idx}: {result['initial_file']} -> {result['target_file']}\n")
            f.write("# Step, Loss\n")
            for step, loss_val in enumerate(loss_hist):
                f.write(f"{step}, {loss_val:.10e}\n")
        print(f"Loss history saved to: {loss_file}")
