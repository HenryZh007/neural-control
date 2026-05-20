## [Supplementary materials for manuscript ''Neural Control: Adjoint Learning Through Equilibrium Constraints'']()

This anonymous website provides supplementary materials referenced in the rebuttal for the submission **“Neural Control: Adjoint Learning Through Equilibrium Constraints.”**

The materials are organized to directly address reviewer questions regarding:
- validation on a learned DEQ-style equilibrium model,
- comparison with a stronger modern derivative-free baseline (iCEM),
- and access to supplementary videos and plots.

All contents are provided in anonymized form for review purposes only.

---

## 1. Validation on a learned DEQ-style equilibrium model

This section provides supplementary materials for the additional validation experiment based on a learned DEQ-style equilibrium model.

### Overview
We collect force–strain measurements from a slinky under one-end actuation and use these data to train a neural energy model $E_\theta(\varepsilon)$. The resulting equilibrium state $\varepsilon^\star$ under control input $z$ is defined implicitly by

$$G(\varepsilon^\star, z; \theta) = F_\theta(\varepsilon^\star) - z = 0$$, $$ F_\theta = \partial E_\theta / \partial \varepsilon.$$

The forward equilibrium is solved to convergence, and training uses implicit differentiation / IFT without unrolling, in the same spirit as DEQ methods.

After training, this learned implicit model is frozen and used as the forward model for Neural Control, which optimizes a force trajectory $z(\lambda)$ so that the resulting equilibrium strain trajectory tracks the target

$$
\varepsilon^*(\lambda) = 0.05\sin(2\pi\lambda) + 0.05, \qquad \lambda \in [0,1].
$$

This experiment is intended to provide a concrete validation on a learned implicit / DEQ-style model beyond the original physics simulator.

### Data collection
A video of the force–strain data collection process is shown below.

<p align="center">
  <img src="DEQ_relevant/video/data_collection.gif" alt="Training data collection for the slinky force-strain dataset">
  <br>
  <em>Figure 1. Training data collection for the force–strain dataset of a slinky through robotic manipulation.</em>
</p>

Original video can be found in ```DEQ_relevant/video/```

### DEQ model training
The training curve of the learned DEQ-style equilibrium model is shown below.

<p align="center">
  <img src="DEQ_relevant/plots/training_loss.png" alt="Training of DEQ model">
  <br>
  <em>Figure 2. Training curve of the DEQ-style equilibrium model.</em>
</p>

### DEQ model inference
The learned force–strain relation and its agreement with experimental data are shown below.

<p align="center">
  <img src="DEQ_relevant/plots/DEQ_model.png" alt="Inference of DEQ model">
  <br>
  <em>Figure 3. Inference results of the learned DEQ-style equilibrium model compared with experimental data.</em>
</p>

### Neural Control on top of the learned DEQ model
We then apply Neural Control to optimize the force input so that the equilibrium strain follows the sinusoidal target above.

<p align="center">
  <img src="DEQ_relevant/plots/training_plot.png" alt="Learning of neural control on DEQ model">
  <br>
  <em>Figure 4. Optimization process of Neural Control on the learned DEQ-style equilibrium model.</em>
</p>

The final result shows near-perfect sinusoidal strain tracking, with segment losses on the order of \(10^{-7}\)–\(10^{-8}\).

## 2. iCEM baseline results

This section provides additional baseline comparison results with [iCEM](https://proceedings.mlr.press/v155/pinneri21a).

The plots below compare iCEM with our Neural Control method (Adjoint + RHC) on all three tasks. The results show that iCEM struggles on these challenging deformable manipulation problems, while Neural Control achieves substantially better performance.

<p align="center">
  <img src="iCEM_relevant/plots/task1.png" alt="iCEM comparison on task 1">
  <br>
  <em>Figure 5. Comparison between iCEM and Neural Control on Task 1.</em>
</p>

<p align="center">
  <img src="iCEM_relevant/plots/task2.png" alt="iCEM comparison on task 2">
  <br>
  <em>Figure 6. Comparison between iCEM and Neural Control on Task 2.</em>
</p>

<p align="center">
  <img src="iCEM_relevant/plots/task3.png" alt="iCEM comparison on task 3">
  <br>
  <em>Figure 7. Comparison between iCEM and Neural Control on Task 3.</em>
</p>

The quantitative results, together with the corresponding time complexity and memory efficiency, are summarized in the table below.

## Quantitative comparison and theoretical complexity
| Method | Time / update | Memory / update | Task 1 Time (s) ↓ | Task 1 Best loss ↓ | Task 2 Time (s) ↓ | Task 2 Best loss ↓ | Task 3 Time (s) ↓ | Task 3 Best loss ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iCEM | O(P K C<sub>eq</sub>) | O(P n<sub>θ</sub>) | `[2332.2 ± 126.7]` | `[1.5e-2 ± 1.5e-2]` | `[3208.0 ± 13.7s]` | `[9.8e-02 ± 3.6e-02]` | `[9907.3 ± 59.2]` | `[1.2e-2 ± 7.3e-3]` |
| **Adjoint + RHC** | O(H C<sub>eq</sub> + H C<sub>lin</sub>) ≈ O(H C<sub>eq</sub>)  |O(H (n<sub>x</sub> + n<sub>z</sub>) + n<sub>θ</sub>) | **`[16.1 ± 2.8s]`** | **`[2.3e-7 ± 3.0e-7]`** | **`[186.9 ± 24.8s]`** | **`[3.6e-8 ± 3.6e-8]`** | **`[50.9 ± 6.1s]`** | **`[3.8e-8 ± 7.4e-9]`** |

## 3. Original task videos

This website also contains supplementary videos for the original manuscript tasks.

These materials are provided to make the experimental outcomes easier to inspect and compare.

### Task 1: driving a selected node of an elastic strip to a prescribed target position

<p align="center">
  <img src="Experimental_Results/Task1_case1/case_1_sample.gif" alt="Task 1, case 1">
  <br>
  <em>Figure 8. Task 1, Example 1: driving a selected node of the elastic strip to a prescribed target position.</em>
</p>

<p align="center">
  <img src="Experimental_Results/Task1_case2/case_2_sample.gif" alt="Task 1, case 2">
  <br>
  <em>Figure 9. Task 1, Example 2: driving a selected node of the elastic strip to a prescribed target position.</em>
</p>

<p align="center">
  <img src="Experimental_Results/Task1_case3/case_3_sample.gif" alt="Task 1, case 3">
  <br>
  <em>Figure 10. Task 1, Example 3: driving a selected node of the elastic strip to a prescribed target position.</em>
</p>

<p align="center">
  <img src="Experimental_Results/Task1_case4/case_4_sample.gif" alt="Task 1, case 4">
  <br>
  <em>Figure 11. Task 1, Example 4: driving a selected node of the elastic strip to a prescribed target position.</em>
</p>

### Task 2: tracing the middle node of an elastic strip along a prescribed trajectory

<p align="center">
  <img src="Experimental_Results/Task2_case1/t2c1_sample.gif" alt="Task 2, case 1">
  <br>
  <em>Figure 12. Task 2, Example 1: tracing the middle node of the elastic strip along a prescribed trajectory.</em>
</p>

<p align="center">
  <img src="Experimental_Results/Task2_case2/t2c2_sample.gif" alt="Task 2, case 2">
  <br>
  <em>Figure 13. Task 2, Example 2: tracing the middle node of the elastic strip along a prescribed trajectory.</em>
</p>

<p align="center">
  <img src="Experimental_Results/Task2_case3/t2c3_sample.gif" alt="Task 2, case 3">
  <br>
  <em>Figure 14. Task 2, Example 3: tracing the middle node of the elastic strip along a prescribed trajectory.</em>
</p>

### Task 3: shape control of an elastic strip

<p align="center">
  <img src="Experimental_Results/Task3/t3_sample.gif" alt="Task 3">
  <br>
  <em>Figure 15. Task 3: shape control of the elastic strip toward a prescribed target configuration.</em>
</p>

Original videos and images can be found in ``Experimental_Results/video/``



