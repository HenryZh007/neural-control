import os, math, copy, time, random, sys
from typing import Optional
import numpy as np
import torch

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # Output directory for saving results
    
    "controlled_percentages": [20, 40, 60, 80],
    
    "target_positions": [[0.2, 0.2], [0.2, 0.2], [-0.05, 0.1], [-0.05, 0.1]],
    
    # Optimization parameters
    "T": 101,                       # Number of time steps
    "learning_rate": 0.01,
    "iteration_number": 100,        # Stop after this many iterations
    "loss_threshold": 1e-7,
    
    # Network parameters
    "hidden_sizes": [64, 64],
    
    # SPSA parameters
    "spsa_c": 5e-3,                 # Perturbation magnitude
    "spsa_m": 2,                    # Number of SPSA pairs to average
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


# =============================================================================
# Simulator helper functions
# =============================================================================
def resetSim(sim_manager):
    sim_manager.resetSim()


def get_sim_states(sim_manager):
    return {
        "vertices": np.asarray(sim_manager.getAllVertices()).copy(),
        "frames":   np.asarray(sim_manager.get_all_frames()).copy(),
    }


def set_sim_states(sim_manager, state):
    sim_manager.set_all_vertices(np.ascontiguousarray(state["vertices"], dtype=np.float64).reshape(-1))
    sim_manager.set_all_frames(np.ascontiguousarray(state["frames"], dtype=np.float64))


# =============================================================================
# Parameter vector helpers for SPSA
# =============================================================================
def parameters_to_vector(params):
    """Flatten a list of parameters into one 1D tensor (on same device)."""
    return torch.cat([p.detach().reshape(-1) for p in params])


@torch.no_grad()
def set_params_from_vector_(params, vec):
    """In-place: params <- vec (flat)."""
    offset = 0
    for p in params:
        n = p.numel()
        p.copy_(vec[offset:offset + n].view_as(p))
        offset += n


def vector_to_grads_list(params, gvec):
    """Split a flat vector into a list of tensors matching params."""
    grads = []
    offset = 0
    for p in params:
        n = p.numel()
        grads.append(gvec[offset:offset + n].view_as(p).clone())
        offset += n
    return grads


# =============================================================================
# Black-box rollout loss (no Jacobians, no adjoint)
# =============================================================================
@torch.no_grad()
def loss_only_forward(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,       # (T,)
    sim_manager,
    targets: np.ndarray,      # (num_targets, 2)
    target_indices: list,     # list of node indices to control
    dlam: float,
    fail_loss: float = 1e6,   # penalty if sim fails
):
    """
    Returns terminal loss 0.5*||x_target - x_node||^2 using simulator rollouts only.
    Supports multiple target nodes.
    """
    policy_model.eval()
    T = int(lams.numel())

    # (T, 2)
    u_seq = policy_model(lams.view(T, 1)).cpu().numpy()

    sim_manager.resetSim()
    verts0 = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()  # (8,)

    final_vertices_flat = None

    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam

        v0 = xb_k[0:2].copy()
        v1 = xb_k[2:4].copy()
        v2 = xb_k[4:6].copy()
        v3 = xb_k[6:8].copy()

        # only move x
        v0[0] += dx1; v1[0] += dx1
        v2[0] += dx2; v3[0] += dx2

        xb_k = np.hstack((v0, v1, v2, v3))

        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        try:
            sim_manager.step()
        except Exception:
            return float(fail_loss)

        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        final_vertices_flat = verts_xy.reshape(-1)  # (2N,)

    # Compute loss - sum over all controlled nodes
    L_total = 0.0
    for idx, target in zip(target_indices, targets):
        v_f = final_vertices_flat.reshape(-1, 2)[idx]
        dv = v_f - target
        L_total += 0.5 * float(dv @ dv)
    
    return L_total


# =============================================================================
# SPSA gradient estimator
# =============================================================================
def compute_dL_dtheta_spsa(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,                 # (T,)
    sim_manager,
    targets: np.ndarray,                # (num_targets, 2)
    target_indices: list,               # list of node indices to control
    dlam: float,
    spsa_c: float = 5e-3,               # perturbation magnitude
    spsa_m: int = 2,                    # number of SPSA pairs to average
    generator: Optional[torch.Generator] = None,  # for reproducible deltas
    fail_loss: float = 1e6,
):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimator.
    
    Returns:
      grads_list : list[Tensor] grads wrt policy_model params (same order as params)
      L_total    : float loss at current (unperturbed) theta
    """
    params = [p for p in policy_model.parameters() if p.requires_grad]
    theta0 = parameters_to_vector(params)

    # Loss at current theta (for logging)
    with torch.no_grad():
        set_params_from_vector_(params, theta0)
        L_total = loss_only_forward(policy_model, lams, sim_manager, targets, target_indices, dlam, fail_loss=fail_loss)

    # SPSA gradient estimate
    ghat = torch.zeros_like(theta0)
    eps = 1e-12
    c = float(spsa_c)
    if c <= 0:
        raise ValueError("spsa_c must be > 0")

    for _ in range(int(spsa_m)):
        # Rademacher Δ ∈ {±1}^d (same dtype/device as theta0)
        delta = torch.empty_like(theta0)
        if generator is None:
            delta.bernoulli_(0.5)
        else:
            delta.bernoulli_(0.5, generator=generator)
        delta.mul_(2).sub_(1)  # {0,1} -> {-1,+1}

        with torch.no_grad():
            # θ + cΔ
            set_params_from_vector_(params, theta0 + c * delta)
            Lp = loss_only_forward(policy_model, lams, sim_manager, targets, target_indices, dlam, fail_loss=fail_loss)

            # θ - cΔ
            set_params_from_vector_(params, theta0 - c * delta)
            Lm = loss_only_forward(policy_model, lams, sim_manager, targets, target_indices, dlam, fail_loss=fail_loss)

        ghat.add_(((Lp - Lm) / (2.0 * c + eps)) * delta)

    ghat.div_(float(spsa_m))

    # Restore original params
    with torch.no_grad():
        set_params_from_vector_(params, theta0)

    grads_list = vector_to_grads_list(params, ghat)
    return grads_list, float(L_total)


import torch.nn as nn

def reinit_net_(net: nn.Module):
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
    net.apply(_init)

    with torch.no_grad():
        for name in ["log_mag", "log_mag_xy", "log_mag_a", "rho_xy", "rho_a", "log_metric"]:
            if hasattr(net, name):
                getattr(net, name).zero_()


def run_single_case(
    sim_manager,
    target_index: int,
    target: np.ndarray,
    case_id: int,
    T: int,
    learning_rate: float,
    iteration_number: int,
    loss_threshold: float,
    hidden_sizes: list,
    spsa_c: float,
    spsa_m: int,
    device: torch.device,
):
    """
    Run a single optimization case.
    
    Returns
    -------
    dict with keys: case_id, target_index, target, epochs, total_time, avg_epoch_time, loss_history, final_loss
    """
    # Time discretization
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    lams = torch.tensor(lams_np, dtype=torch.float32, device=device)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([0.05 / dlam, 0.05 / dlam], dtype=torch.float32)

    net = create_policy_model(
        input_size=1,
        hidden_sizes=hidden_sizes,
        output_size=2,
        bounds=bounds,
    ).to(device)

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

    # SPSA generator for reproducible random perturbations
    spsa_gen = torch.Generator(device=device)
    spsa_gen.manual_seed(123 + case_id)  # different seed for each case

    loss_hist = []
    epoch_dt_hist = []

    best_loss = float("inf")
    best_state = None

    # Wrap single target in arrays for the function interface
    target_positions = np.array([target], dtype=np.float64)
    target_indices = [target_index]

    # Training loop
    total_start_time = time.perf_counter()
    
    for epoch in range(iteration_number):
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        grads_list, loss = compute_dL_dtheta_spsa(
            net,
            lams,
            sim_manager,
            target_positions,
            target_indices,
            dlam,
            spsa_c=spsa_c,
            spsa_m=spsa_m,
            generator=spsa_gen,
            fail_loss=1e6,
        )

        params = [p for p in net.parameters() if p.requires_grad]
        for p, g in zip(params, grads_list):
            p.grad = g.detach()

        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()

        loss_val = float(loss)
        loss_hist.append(loss_val)

        epoch_dt = time.perf_counter() - t0
        epoch_dt_hist.append(epoch_dt)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(net.state_dict())

        grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_list)).cpu())
        print(f"  Epoch {epoch:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e} | dt {epoch_dt*1e3:.1f} ms")

        if loss_val < loss_threshold:
            print(f"  Converged at epoch {epoch}")
            break

    total_time = time.perf_counter() - total_start_time
    avg_epoch_time = np.mean(epoch_dt_hist)

    return {
        "case_id": case_id,
        "target_index": target_index,
        "target": target.tolist(),
        "epochs": len(loss_hist),
        "total_time": total_time,
        "avg_epoch_time": avg_epoch_time,
        "loss_history": loss_hist,
        "final_loss": best_loss,
    }


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
    controlled_percentages = CONFIG["controlled_percentages"]
    target_positions = np.array(CONFIG["target_positions"], dtype=np.float64)
    
    T = CONFIG["T"]
    learning_rate = CONFIG["learning_rate"]
    iteration_number = CONFIG["iteration_number"]
    loss_threshold = CONFIG["loss_threshold"]
    hidden_sizes = CONFIG["hidden_sizes"]
    spsa_c = CONFIG["spsa_c"]
    spsa_m = CONFIG["spsa_m"]
    
    num_cases = len(controlled_percentages)
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Point-to-Point Control Configuration (SPSA)")
    print(f"Running {num_cases} cases separately, then computing average")
    print(f"{'='*60}")
    print(f"  Target node indices: {controlled_percentages}")
    print(f"  Target positions: {target_positions.tolist()}")
    print(f"  Number of time steps: {T}")
    print(f"  Number of iterations per case: {iteration_number}")
    print(f"  SPSA perturbation (c): {spsa_c}")
    print(f"  SPSA pairs (m): {spsa_m}")
    print(f"{'='*60}\n")

    # Run all cases and collect results
    all_results = []
    overall_start_time = time.perf_counter()
    
    for case_id, (target_idx, target_pos) in enumerate(zip(controlled_percentages, target_positions)):
        print(f"\n{'='*60}")
        print(f"Case {case_id + 1}/{num_cases}: target_index={target_idx}, target={target_pos.tolist()}")
        print(f"{'='*60}")
        
        result = run_single_case(
            sim_manager=sim_manager,
            target_index=target_idx,
            target=target_pos,
            case_id=case_id,
            T=T,
            learning_rate=learning_rate,
            iteration_number=iteration_number,
            loss_threshold=loss_threshold,
            hidden_sizes=hidden_sizes,
            spsa_c=spsa_c,
            spsa_m=spsa_m,
            device=device,
        )
        all_results.append(result)
        
        print(f"\n  Case {case_id + 1} completed:")
        print(f"    Total time: {result['total_time']:.3f} s")
        print(f"    Best loss: {result['final_loss']:.6e}")
        print(f"    Avg epoch time: {result['avg_epoch_time']*1e3:.3f} ms")

    overall_time = time.perf_counter() - overall_start_time
    
    # Compute statistics across all cases
    all_times = [r["total_time"] for r in all_results]
    all_losses = [r["final_loss"] for r in all_results]
    all_avg_epoch_times = [r["avg_epoch_time"] for r in all_results]
    all_epochs = [r["epochs"] for r in all_results]
    
    avg_time = np.mean(all_times)
    avg_loss = np.mean(all_losses)
    avg_epoch_time = np.mean(all_avg_epoch_times)
    avg_epochs = np.mean(all_epochs)
    
    print(f"\n{'='*60}")
    print(f"ALL CASES COMPLETED (SPSA)")
    print(f"{'='*60}")
    print(f"  Number of cases: {num_cases}")
    print(f"  Overall time: {overall_time:.3f} s")
    print(f"\nPer-case statistics:")
    print(f"  Total time:     {avg_time:.3f} s")
    print(f"  Final loss:     {avg_loss:.6e}")
    print(f"  Avg epoch time: {avg_epoch_time*1e3:.3f} ms")
    print(f"  Epochs run:     {avg_epochs:.1f}")
    print(f"{'='*60}\n")

    # Save result to current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "any_node_spsa.txt")
    
    # Save summary to txt file
    with open(output_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Training Results Summary (SPSA) - Multiple Cases\n")
        f.write("="*60 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Target node indices: {controlled_percentages}\n")
        f.write(f"  Target positions: {target_positions.tolist()}\n")
        f.write(f"  Number of time steps: {T}\n")
        f.write(f"  Iteration number per case: {iteration_number}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Hidden sizes: {hidden_sizes}\n")
        f.write(f"  SPSA perturbation (c): {spsa_c}\n")
        f.write(f"  SPSA pairs (m): {spsa_m}\n\n")
        
        f.write("Per-case Results:\n")
        f.write("-"*60 + "\n")
        for r in all_results:
            f.write(f"  Case {r['case_id'] + 1}: target_idx={r['target_index']}, target={r['target']}\n")
            f.write(f"    Total time: {r['total_time']:.6f} s\n")
            f.write(f"    Final loss: {r['final_loss']:.10e}\n")
            f.write(f"    Epochs run: {r['epochs']}\n")
            f.write(f"    Avg epoch time: {r['avg_epoch_time']*1e3:.3f} ms\n")
            f.write("\n")
        
        f.write("-"*60 + "\n")
        f.write("Aggregate Statistics:\n")
        f.write(f"  Number of cases: {num_cases}\n")
        f.write(f"  Overall time: {overall_time:.6f} s\n")
        f.write(f"  Avg total time per case: {avg_time:.6f} s\n")
        f.write(f"  Avg final loss: {avg_loss:.10e}\n")
        f.write(f"  Avg epoch time: {avg_epoch_time:.6f} s ({avg_epoch_time*1e3:.3f} ms)\n")
        f.write(f"  Avg epochs run: {avg_epochs:.1f}\n")
        f.write("="*60 + "\n")
    
    # Save per-step loss for each case to txt files
    for r in all_results:
        case_id = r['case_id']
        target_index = r['target_index']
        loss_hist = r['loss_history']
        loss_file = os.path.join(script_dir, f"any_node_spsa_case{case_id}_node{target_index}_loss.txt")
        with open(loss_file, "w") as f:
            f.write("# Per-step loss history for any_node_spsa\n")
            f.write(f"# Case {case_id}: Node {target_index} -> {r['target']}\n")
            f.write("# Step, Loss\n")
            for step, loss_val in enumerate(loss_hist):
                f.write(f"{step}, {loss_val:.10e}\n")
        print(f"Loss history saved to: {loss_file}")
