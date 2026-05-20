import os, math, copy, time, random, sys
import numpy as np
import torch
import torch.nn as nn

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # Multiple cases: each case is (target_index, target_position)
    # Each case will be run independently from scratch
    "cases": [
        {"target_index": 20, "target_position": [0.2, 0.2]},
        {"target_index": 40, "target_position": [0.2, 0.2]},
        {"target_index": 60, "target_position": [-0.05, 0.1]},
        {"target_index": 80, "target_position": [-0.05, 0.1]},
    ],
    
    # MPC parameters
    "max_total_iterations": 10,     # Maximum total iterations per case
    "inner_iterations": 50,         # Inner optimization iterations per MPC step
    "learning_rate": 0.01,
    
    # Early stopping
    "patience": 5,
    "min_delta_rel": 1e-4,
    "loss_threshold": 1e-20,
    
    # Time discretization
    "T": 101,                       # Number of time steps per MPC horizon
    
    # Network parameters
    "hidden_sizes": [64, 64],
    
    # Control bounds (will be divided by dlam)
    "bounds_x": 0.05,
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
# Simulator state management for MPC
# =============================================================================
reset_state = None


def resetSim(sim_manager):
    sim_manager.resetSim()
    if reset_state is not None:
        set_sim_states(sim_manager, reset_state)


def get_sim_states(sim_manager):
    return {
        "vertices": np.asarray(sim_manager.getAllVertices()).copy(),
        "frames":   np.asarray(sim_manager.get_all_frames()).copy(),
    }


def set_sim_states(sim_manager, state):
    sim_manager.set_all_vertices(np.ascontiguousarray(state["vertices"], dtype=np.float64).reshape(-1))
    sim_manager.set_all_frames(np.ascontiguousarray(state["frames"], dtype=np.float64))


# =============================================================================
# Small helpers
# =============================================================================
def parameters_to_vector(params):
    """Flatten a list of parameters into one 1D tensor."""
    return torch.cat([p.detach().flatten() for p in params])


def grads_to_vector(grads_list):
    """Flatten a list of grads into one 1D tensor."""
    return torch.cat([g.detach().flatten() for g in grads_list])


def reinit_net_(net: nn.Module):
    """Reinitialize network weights."""
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
    net.apply(_init)

    with torch.no_grad():
        for name in ["log_mag", "log_mag_xy", "log_mag_a", "rho_xy", "rho_a", "log_metric"]:
            if hasattr(net, name):
                getattr(net, name).zero_()


def rebuild_optimizer(old_opt, net):
    """Rebuild optimizer with the same hyperparameters."""
    return old_opt.__class__([p for p in net.parameters() if p.requires_grad], **old_opt.defaults)


# =============================================================================
# Core: compute gradients dL/dtheta
# =============================================================================
def compute_dL_dtheta(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,
    sim_manager,
    target: np.ndarray,
    target_index: int,
    dlam: float,
    jac_reg: float = 1e-6,
):
    """
    Compute gradients for point-to-point reaching task with MPC.
    
    Parameters
    ----------
    target : np.ndarray
        Shape (2,) - target position for the tracked node
    target_index : int
        Index of the node to control
    
    Returns
    -------
    grads_list : list[torch.Tensor]
        Gradients w.r.t. policy_model parameters.
    L_total : float
        Scalar loss value.
    buckled : bool
        Whether the rod buckled during simulation.
    """
    policy_model.eval()

    # Query policy for control sequence
    T = int(lams.numel())
    u_seq_torch = policy_model(lams.view(T, 1))
    u_seq = u_seq_torch.detach().cpu().numpy()

    # Forward rollout in simulator
    resetSim(sim_manager)

    verts0 = np.asarray(sim_manager.getAllVertices()).copy()
    verts0_xy = verts0[:, :2]
    N = verts0_xy.shape[0]

    xb_k = verts0_xy[[0, 1, -2, -1], :].reshape(-1).copy()

    # Pre-allocate lists for adjoint
    A_list = np.zeros((T, 8, 8), dtype=np.float64)
    B_list = np.zeros((T, 8, 2), dtype=np.float64)
    dXf_dXb_list = []
    vertices_list = []

    buckled = False

    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam

        xb0_k = xb_k.copy()

        v0 = xb_k[:2].copy()
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

        jac = np.asarray(sim_manager.getJacobian()).copy()
        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        vertices_flat = verts_xy.reshape(-1)

        lhs = jac[4:-4, 4:-4]
        rhs = -np.hstack((jac[4:-4, :4], jac[4:-4, -4:]))

        lhs_reg = lhs + jac_reg * np.eye(lhs.shape[0], dtype=np.float64)
        try:
            dxf_dxb = np.linalg.solve(lhs_reg, rhs)
        except np.linalg.LinAlgError:
            dxf_dxb = np.linalg.lstsq(lhs_reg, rhs, rcond=None)[0]

        dXf_dXb_list.append(dxf_dxb)

        # Check for buckling
        xf0_k = vertices_list[-1][4:-4] if vertices_list else verts0_xy.reshape(-1)[4:-4]
        xf_try = xf0_k + dxf_dxb @ (xb_k - xb0_k)
        xf_k = vertices_flat[4:-4]
        e_metric = np.linalg.norm(xf_try - xf_k)
        if e_metric > 0.1 and i != 0:
            buckled = True

        A = np.zeros((8, 8), dtype=np.float64)
        B = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
        ], dtype=np.float64)

        A_list[i] = A
        B_list[i] = B.T

        vertices_list.append(vertices_flat.copy())

    # Compute loss (only at final step)
    final_vertices_flat = vertices_list[-1]
    v_f = final_vertices_flat.reshape(-1, 2)[target_index]
    dv = v_f - target
    L_total = 0.5 * float(dv @ dv)

    # Adjoint sensitivity
    a_q = np.zeros((2 * N,), dtype=np.float64)
    a_q[2 * target_index : 2 * target_index + 2] = dv

    lam_f = a_q[4:-4]
    lam_b = np.concatenate([a_q[:4], a_q[-4:]])

    v_u = np.zeros((T, 2), dtype=np.float64)
    I8 = np.eye(8, dtype=np.float64)

    for i in range(T - 1, -1, -1):
        dxf_dxb = dXf_dXb_list[i]
        A = A_list[i]
        B = B_list[i]

        v_u[i] = dlam * (B.T @ lam_b) + dlam * ((dxf_dxb @ B).T @ lam_f)
        lam_b = (I8 + dlam * A.T) @ lam_b + dlam * ((dxf_dxb @ A).T @ lam_f)

    # Torch VJP
    v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)
    surrogate = (u_seq_torch * v_u_torch).sum()

    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(
        surrogate, params, retain_graph=False, create_graph=False, allow_unused=False
    )

    return grads_list, L_total, buckled


def run_single_case(
    sim_manager,
    target_index: int,
    target: np.ndarray,
    config: dict,
    device: torch.device,
    case_idx: int,
):
    """
    Run MPC training for a single case.
    
    Returns
    -------
    result : dict
        Contains total_time, best_loss, total_mpc_steps, etc.
    """
    global reset_state
    
    # Initialize reset_state to None for fresh start
    reset_state = None
    
    # Reset simulator
    sim_manager.resetSim()

    # Training setup
    T = config["T"]
    max_total_iterations = config["max_total_iterations"]
    inner_iterations = config["inner_iterations"]
    learning_rate = config["learning_rate"]
    hidden_sizes = config["hidden_sizes"]
    bounds_x = config["bounds_x"]
    patience = config["patience"]
    min_delta_rel = config["min_delta_rel"]
    loss_threshold = config["loss_threshold"]

    lams_np = np.linspace(0.0, 1.0, T).astype(np.float32)
    lams = torch.tensor(lams_np, device=device, requires_grad=True)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([bounds_x/dlam, bounds_x/dlam], dtype=torch.float32)

    net = create_policy_model(
        input_size=1,
        hidden_sizes=hidden_sizes,
        output_size=2,
        bounds=bounds,
    ).to(device)

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

    # MPC Training loop
    best_loss = float('inf')
    loss_hist = []
    epoch_dt_hist = []

    start_time = time.perf_counter()
    
    mpc_step = 0
    total_iterations = 0
    while total_iterations < max_total_iterations and best_loss > loss_threshold:
        t0 = time.perf_counter()
        
        # Save current state for MPC horizon
        reset_state = get_sim_states(sim_manager)
        
        # Reinitialize network and optimizer for new MPC step
        reinit_net_(net)
        optimizer = rebuild_optimizer(optimizer, net)
        
        best_so_far = float('inf')
        stale_steps = 0
        buckled = False
        early_stop = False
        iter_inner = 0
        
        while (iter_inner <= inner_iterations or buckled) and total_iterations < max_total_iterations:
            optimizer.zero_grad(set_to_none=True)
            
            grads_list, loss, buckled = compute_dL_dtheta(
                net,
                lams,
                sim_manager,
                target,
                target_index,
                dlam,
            )
            
            loss_val = float(loss)
            improve = (best_so_far - loss_val) / max(abs(best_so_far), 1e-12)
            if loss_val < best_so_far:
                best_so_far = loss_val
            
            if improve < min_delta_rel:
                stale_steps += 1
            else:
                stale_steps = 0
            
            if stale_steps >= patience:
                early_stop = True
            
            params = [p for p in net.parameters() if p.requires_grad]
            for p, g in zip(params, grads_list):
                p.grad = g.detach()
            
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            loss_hist.append(loss_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
            
            grad_norm = float(torch.sqrt(sum((g.detach()**2).sum() for g in grads_list)).cpu())
            print(f"Case {case_idx:02d} | MPC {mpc_step:03d} | iter {iter_inner:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e} | buckled: {buckled}")
            
            if early_stop:
                break
            iter_inner += 1
            total_iterations += 1
        
        epoch_dt = time.perf_counter() - t0
        epoch_dt_hist.append(epoch_dt)
        
        print(f"\n[Case {case_idx}] MPC step {mpc_step} completed. Best loss so far: {best_loss:.6e}\n")
        mpc_step += 1

    total_time = time.perf_counter() - start_time
    avg_mpc_time = np.mean(epoch_dt_hist) if epoch_dt_hist else 0.0

    return {
        "target_index": target_index,
        "target_position": target.tolist(),
        "total_time": total_time,
        "best_loss": best_loss,
        "total_mpc_steps": mpc_step,
        "avg_mpc_step_time": avg_mpc_time,
        "loss_history": loss_hist,
    }


if __name__ == "__main__":
    configure_threads(num_threads=1)
    set_seed(42, deterministic=True)

    device = torch.device("cpu")

    # Load configuration
    cases = CONFIG["cases"]
    
    # Create simulator
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

    # Store results for all cases
    all_results = []

    print(f"\n{'='*70}")
    print(f"Running {len(cases)} cases with MPC")
    print(f"Max total iterations per case: {CONFIG['max_total_iterations']}")
    print(f"Inner iterations per MPC step: {CONFIG['inner_iterations']}")
    print(f"{'='*70}\n")

    for case_idx, case in enumerate(cases):
        # Reset seed for each case to ensure fair comparison
        set_seed(42)
        
        target_index = case["target_index"]
        target = np.array(case["target_position"], dtype=np.float64)

        print(f"\n{'='*70}")
        print(f"[{case_idx+1}/{len(cases)}] Case: node {target_index} -> {target.tolist()}")
        print(f"{'='*70}\n")

        result = run_single_case(
            sim_manager,
            target_index,
            target,
            CONFIG,
            device,
            case_idx,
        )

        all_results.append(result)

        print(f"\n[Case {case_idx}] Completed!")
        print(f"  Total time: {result['total_time']:.3f} s")
        print(f"  Best loss: {result['best_loss']:.6e}")
        print(f"  Total MPC steps: {result['total_mpc_steps']}")
        print(f"  Avg MPC step time: {result['avg_mpc_step_time']:.3f} s")

    # Compute averages
    avg_total_time = np.mean([r['total_time'] for r in all_results])
    avg_best_loss = np.mean([r['best_loss'] for r in all_results])
    avg_mpc_step_time = np.mean([r['avg_mpc_step_time'] for r in all_results])

    # Save results to txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "any_node_MPC.txt")

    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Summary Table (Any Node MPC Method):\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration: Max total iters={CONFIG['max_total_iterations']}, Inner iters={CONFIG['inner_iterations']}, T={CONFIG['T']}, LR={CONFIG['learning_rate']}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Case':<6} {'Target Node':<15} {'Target Pos':<25} {'Time (s)':<12} {'Best Loss':<15} {'MPC Steps':<10}\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(all_results):
            target_pos_str = str(result['target_position'])
            f.write(f"{i:<6} {result['target_index']:<15} {target_pos_str:<25} {result['total_time']:<12.4f} {result['best_loss']:<15.6e} {result['total_mpc_steps']:<10}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'AVG':<6} {'':<15} {'':<25} {avg_total_time:<12.4f} {avg_best_loss:<15.6e}\n")
        f.write("="*80 + "\n")
        f.write(f"\nAverage MPC step time: {avg_mpc_step_time:.4f} s\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"All Cases Completed!")
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Total cases: {len(all_results)}")
    print(f"  Average total time: {avg_total_time:.4f} s")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average MPC step time: {avg_mpc_step_time:.4f} s")
    print(f"{'='*70}")
    print(f"\nResults saved to {output_file}")
    
    # Save per-step loss for each case to txt files
    for case_idx, result in enumerate(all_results):
        target_index = result['target_index']
        loss_hist = result['loss_history']
        loss_file = os.path.join(script_dir, f"any_node_MPC_case{case_idx}_node{target_index}_loss.txt")
        with open(loss_file, "w") as f:
            f.write("# Per-step loss history for any_node_MPC\n")
            f.write(f"# Case {case_idx}: Node {target_index} -> {result['target_position']}\n")
            f.write("# Step, Loss\n")
            for step, loss_val in enumerate(loss_hist):
                f.write(f"{step}, {loss_val:.10e}\n")
        print(f"Loss history saved to: {loss_file}")
