import os, math, copy, time, random, sys
import numpy as np
import torch
import torch.nn as nn

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
    
    # MPC parameters
    "T_horizon": 10,                # Steps per MPC epoch
    "total_trajectory_length": 101, # Total length of target trajectory
    "max_inner_iters": 100,         # Max iterations per MPC epoch
    
    # Optimization parameters
    "learning_rate": 0.01,
    "iteration_number": 1000,          # Total iteration limit for the entire case
    "loss_threshold": 1e-6,
    
    # Network parameters
    "hidden_sizes": [64, 64],
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
    compute_grads: bool = True,
):
    """
    Compute gradients for trajectory tracking task.
    
    Parameters
    ----------
    target : np.ndarray
        Shape (T, 2) - target trajectory for the tracked node
    target_index : int
        Index of the node to track
    compute_grads : bool
        If True, compute gradients. If False, only do forward rollout.
    
    Returns
    -------
    grads_list : list[torch.Tensor] or None
        Gradients w.r.t. policy_model parameters. None if compute_grads=False.
    L_total : float
        Scalar loss value.
    vertices_list : list[np.ndarray]
        List of vertices at each time step.
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

    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam

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

        A = np.zeros((8, 8), dtype=np.float64)
        B = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
        ], dtype=np.float64)

        A_list[i] = A
        B_list[i] = B.T

        vertices_list.append(vertices_flat.copy())

    # Compute loss (tracking loss over all steps)
    a_q_array = np.zeros((T, vertices_list[0].shape[0]), dtype=np.float64)
    L_total = 0.0
    for i in range(T):
        v_i = vertices_list[i].reshape(-1, 2)[target_index]
        dv = v_i - target[i]
        L_total += 0.5 * (dv @ dv) * dlam
        a_q = np.zeros((2 * N,), dtype=np.float64)
        a_q[2 * target_index : 2 * target_index + 2] = dv
        a_q_array[i] = a_q.flatten()

    if not compute_grads:
        return None, L_total, vertices_list

    # Backward adjoint to compute v_u[i] = dl_i/du_i
    I8 = np.eye(8, dtype=np.float64)

    surrogate_total = u_seq_torch.new_tensor(0.0)
    for k in range(T):
        v_u = np.zeros((k + 1, 2), dtype=np.float64)
        a_q = a_q_array[k] * dlam
        lam_f = a_q[4:-4].copy()
        lam_b = np.concatenate([a_q[:4], a_q[-4:]]).copy()

        for i in range(k, -1, -1):
            dxf_dxb = dXf_dXb_list[i]
            A = A_list[i]
            B = B_list[i]
            v_u[i] = dlam * (B.T @ lam_b) + dlam * ((dxf_dxb @ B).T @ lam_f)
            lam_b = (I8 + dlam * A.T) @ lam_b + dlam * ((dxf_dxb @ A).T @ lam_f)

        v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)
        u_seq_tmp = u_seq_torch[:k+1, :]
        surrogate_total += (u_seq_tmp * v_u_torch).sum()

    # Torch VJP
    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(
        surrogate_total, params, retain_graph=False, create_graph=False, allow_unused=False
    )

    return grads_list, L_total, vertices_list


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


def rebuild_optimizer(old_opt, net):
    return old_opt.__class__([p for p in net.parameters() if p.requires_grad], **old_opt.defaults)


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
    sim_manager.resetSim()

    verts_init = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts_init.shape[0]

    # Load configuration parameters
    trajectory_types = CONFIG["trajectory_types"]
    trajectory_params = CONFIG["trajectory_params"]
    target_index = CONFIG["target_index"]
    T_horizon = CONFIG["T_horizon"]
    total_trajectory_length = CONFIG["total_trajectory_length"]
    max_inner_iters = CONFIG["max_inner_iters"]
    learning_rate = CONFIG["learning_rate"]
    iteration_number = CONFIG["iteration_number"]
    loss_threshold = CONFIG["loss_threshold"]
    hidden_sizes = CONFIG["hidden_sizes"]

    # Time discretization for each MPC horizon
    T = T_horizon
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    lams = torch.tensor(lams_np, dtype=torch.float32, device=device)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([0.1 / dlam, 0.1 / dlam], dtype=torch.float32)

    # Store results for all trajectories
    all_results = []

    print(f"\n{'='*60}")
    print(f"Testing {len(trajectory_types)} trajectory types with MPC")
    print(f"Horizon: {T_horizon}, Total iteration limit: {iteration_number}")
    print(f"{'='*60}\n")

    for traj_idx, trajectory_type in enumerate(trajectory_types):
        # Reset seed for each trajectory to ensure fair comparison
        set_seed(42)
        
        # Reset simulator to initial state
        reset_state = None
        sim_manager.resetSim()
        
        # Generate full target trajectory (total_trajectory_length steps)
        middle_node = verts_init[target_index, :].copy()
        target_full = generate_trajectory(trajectory_type, middle_node, total_trajectory_length, trajectory_params)
        
        traj_desc = get_trajectory_description(trajectory_type, trajectory_params)

        # Print configuration
        print(f"\n{'='*60}")
        print(f"[{traj_idx+1}/{len(trajectory_types)}] Trajectory: {traj_desc}")
        print(f"  Target node index: {target_index}")
        print(f"  Horizon: {T_horizon}, Total iteration limit: {iteration_number}")
        print(f"{'='*60}\n")

        # Create fresh network and optimizer for each trajectory
        net = create_policy_model(
            input_size=1,
            hidden_sizes=hidden_sizes,
            output_size=2,
            bounds=bounds,
        ).to(device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

        loss_hist = []
        epoch_dt_hist = []
        position_history = []  # Store middle point position at each iteration
        total_iters = 0
        mpc_epoch = 0

        best_overall_loss = float("inf")

        # MPC Training loop - continue until total iteration limit reached
        total_start_time = time.perf_counter()

        while total_iters < iteration_number:
            t0 = time.perf_counter()
            
            loss_val = float("inf")
            iter_inner_num = 0
            
            # Save current simulator state for this MPC epoch
            reset_state = get_sim_states(sim_manager)
            
            # Reinitialize network for each MPC epoch
            reinit_net_(net)
            optimizer = rebuild_optimizer(optimizer, net)

            best_loss = float("inf")
            best_state = None
            
            # Get current target window (sliding window like test4)
            target_window = target_full[mpc_epoch * T_horizon : mpc_epoch * T_horizon + T, :]
            
            # Check if we have enough target points left
            if target_window.shape[0] < T:
                print(f"  -> Reached end of target trajectory at MPC epoch {mpc_epoch}")
                break
            
            # Inner optimization loop for this MPC epoch
            while loss_val > loss_threshold and iter_inner_num < max_inner_iters and total_iters < iteration_number:
                optimizer.zero_grad(set_to_none=True)
                
                grads_list, loss, vertices_list = compute_dL_dtheta(
                    net,
                    lams,
                    sim_manager,
                    target_window,
                    target_index,
                    dlam,
                )

                # Assign grads to parameters
                params = [p for p in net.parameters() if p.requires_grad]
                for p, g in zip(params, grads_list):
                    p.grad = g.detach()

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                loss_val = float(loss)
                loss_hist.append(loss_val)
                
                # Collect middle point positions at each time step for this iteration
                iter_positions = []
                for v_flat in vertices_list:
                    v_xy = v_flat.reshape(-1, 2)
                    iter_positions.append(v_xy[target_index].tolist())
                position_history.append(iter_positions)

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_state = {
                        "mpc_epoch": mpc_epoch,
                        "iter": iter_inner_num,
                        "best_loss": best_loss,
                        "model_state_dict": copy.deepcopy(net.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    }

                grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_list)).cpu())
                print(f"MPC Epoch {mpc_epoch:03d} | iter {iter_inner_num:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e}")
                
                iter_inner_num += 1
                total_iters += 1

            epoch_dt = time.perf_counter() - t0
            epoch_dt_hist.append(epoch_dt)

            print(f"  -> Best loss for MPC epoch {mpc_epoch}: {best_loss:.6e}")

            if best_loss < best_overall_loss:
                best_overall_loss = best_loss

            # Load best model for this epoch
            if best_state is not None:
                net.load_state_dict(best_state["model_state_dict"])

            # Run one more forward pass to advance simulator state (no gradient computation)
            _, _, final_vertices_list = compute_dL_dtheta(
                net, lams, sim_manager, target_window, target_index, dlam,
                compute_grads=False
            )

            # Update reset_state for next MPC epoch (continue from current state)
            reset_state = get_sim_states(sim_manager)
            mpc_epoch += 1

        total_time = time.perf_counter() - total_start_time
        avg_epoch_time = np.mean(epoch_dt_hist) if epoch_dt_hist else 0.0
        num_mpc_epochs_done = mpc_epoch

        print(f"\n{'='*60}")
        print(f"[{trajectory_type}] MPC Training completed!")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Best overall loss: {best_overall_loss:.6e}")
        print(f"  Total iterations: {total_iters}")
        print(f"  MPC epochs completed: {num_mpc_epochs_done}")
        print(f"  Average MPC epoch time: {avg_epoch_time:.3f} s")
        print(f"{'='*60}\n")

        # Store results
        all_results.append({
            "trajectory_type": trajectory_type,
            "trajectory_desc": traj_desc,
            "total_time": total_time,
            "best_loss": best_overall_loss,
            "total_iters": total_iters,
            "avg_epoch_time": avg_epoch_time,
            "num_mpc_epochs": num_mpc_epochs_done,
            "position_history": position_history,
        })

    # Save all results to current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "middle_tracking_MPC.txt")

    # Save summary to txt file
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("Summary Table (MPC Method):\n")
        f.write("="*70 + "\n")
        f.write(f"MPC Configuration: Horizon={T_horizon}, Iteration limit={iteration_number}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Trajectory':<20} {'Total Time (s)':<15} {'Best Loss':<15} {'Total Iters':<12} {'Avg Epoch (s)':<15}\n")
        f.write("-"*70 + "\n")
        for result in all_results:
            f.write(f"{result['trajectory_type']:<20} {result['total_time']:<15.4f} {result['best_loss']:<15.6e} {result['total_iters']:<12} {result['avg_epoch_time']:<15.4f}\n")
        f.write("-"*70 + "\n")
        
        # Compute averages
        avg_total_time = np.mean([r['total_time'] for r in all_results])
        avg_best_loss = np.mean([r['best_loss'] for r in all_results])
        avg_total_iters = np.mean([r['total_iters'] for r in all_results])
        avg_epoch_time_all = np.mean([r['avg_epoch_time'] for r in all_results])
        
        f.write(f"{'AVERAGE':<20} {avg_total_time:<15.4f} {avg_best_loss:<15.6e} {avg_total_iters:<12.1f} {avg_epoch_time_all:<15.4f}\n")
        f.write("="*70 + "\n")

    print(f"\n{'='*60}")
    print(f"Overall Averages:")
    print(f"  Average total time: {avg_total_time:.4f} s")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average total iterations: {avg_total_iters:.1f}")
    print(f"  Average MPC epoch time: {avg_epoch_time_all:.4f} s")
    print(f"{'='*60}")
    print(f"\nAll results saved to {output_file}")
    
    # Save per-step middle point positions for each case to txt files
    for traj_idx, result in enumerate(all_results):
        traj_type = result['trajectory_type']
        position_hist = result['position_history']
        pos_file = os.path.join(script_dir, f"middle_tracking_MPC_case{traj_idx}_{traj_type}_positions.txt")
        with open(pos_file, "w") as f:
            f.write("# Per-step middle point position history for middle_tracking_MPC\n")
            f.write(f"# Case {traj_idx}: Trajectory type = {traj_type}\n")
            f.write(f"# Target node index: {target_index}\n")
            f.write("# Format: Iteration, TimeStep, X, Y\n")
            for iter_idx, iter_positions in enumerate(position_hist):
                for step_idx, (x, y) in enumerate(iter_positions):
                    f.write(f"{iter_idx}, {step_idx}, {x:.10e}, {y:.10e}\n")
        print(f"Position history saved to: {pos_file}")
