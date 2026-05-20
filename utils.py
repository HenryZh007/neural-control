"""
Utility functions for neural network policy models and geometric transformations.

This module provides:
- PolicyNetwork class for bounded control outputs
- Geometric transformation utilities
- Coordinate conversion functions
"""

from __future__ import annotations

from typing import Optional, Tuple, Callable, Dict, Any, Union, Sequence
import numpy as np
import math

import torch
from torch import nn


# =============================================================================
# Policy Network
# =============================================================================
class PolicyNetwork(nn.Module):
    """
    Neural network for generating bounded control actions.
    
    Uses tanh squashing to ensure outputs stay within specified bounds.
    
    Parameters
    ----------
    input_size : int
        Dimension of input features.
    hidden_sizes : list of int
        Sizes of hidden layers.
    output_size : int
        Dimension of output (number of control variables).
    bounds : Tensor or tuple or None
        Control bounds. If Tensor, symmetric bounds [-L, L].
        If tuple, (low, high) bounds.
    activation : nn.Module class
        Activation function class (default: nn.ReLU).
    beta : float
        Scaling factor for tanh squashing.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (64, 64),
        output_size: int = 2,
        bounds=None,
        activation=nn.ReLU,
        beta: float = 0.5
    ):
        super(PolicyNetwork, self).__init__()
        
        # Build hidden layers
        layers = []
        prev = input_size

        for h in hidden_sizes:
            lin = nn.Linear(prev, h)
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            nn.init.zeros_(lin.bias)
            layers += [lin, activation()]
            prev = h

        self.out = nn.Linear(prev, output_size)
        self.network = nn.Sequential(*layers)
        self.beta = beta

        self._set_bounds(bounds, output_size)

    def _set_bounds(self, bounds, output_size: int) -> None:
        """Set output bounds as registered buffers."""
        if bounds is None:
            self.register_buffer("low", None, persistent=False)
            self.register_buffer("high", None, persistent=False)
            return
        
        if isinstance(bounds, tuple):
            low, high = bounds
            low = torch.as_tensor(low).view(1, -1)
            high = torch.as_tensor(high).view(1, -1)
        else:
            L = torch.as_tensor(bounds).view(1, -1)
            if L.numel() == 1:
                L = L.expand(1, output_size)
            low, high = -L, L

        assert low.shape[-1] == output_size and high.shape[-1] == output_size, \
            "Bounds shape mismatch"

        low, high = torch.minimum(low, high), torch.maximum(low, high)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional tanh squashing.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        out : torch.Tensor
            Output tensor, bounded if bounds were specified.
        """
        z = self.out(self.network(x))
        if (self.low is None) or (self.high is None):
            return z
        u = torch.tanh(self.beta * z)  # [-1, 1]
        L = (self.high - self.low) * 0.5
        return L * u


def create_policy_model(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    bounds=None,
    actovation=nn.ReLU
) -> PolicyNetwork:
    """
    Create a policy network model.
    
    Parameters
    ----------
    input_size : int
        Dimension of input features.
    hidden_sizes : list of int
        Sizes of hidden layers.
    output_size : int
        Dimension of output.
    bounds : optional
        Control bounds for output squashing.
    actovation : nn.Module class
        Activation function class.
        
    Returns
    -------
    model : PolicyNetwork
        Initialized policy network.
    """
    model = PolicyNetwork(
        input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        bounds=bounds,
        activation=actovation
    )
    return model


# =============================================================================
# Geometric Transformations
# =============================================================================
def sine_curve_between_points(
    A: np.ndarray,
    B: np.ndarray,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    n_points: int = 200,
    mode: str = 'sin'
) -> np.ndarray:
    """
    Generate a sinusoidal curve between two points.
    
    Parameters
    ----------
    A : np.ndarray
        Starting point (2D).
    B : np.ndarray
        Ending point (2D).
    amplitude : float
        Amplitude of the wave.
    frequency : float
        Frequency of the wave.
    n_points : int
        Number of points on the curve.
    mode : str
        'sin' or 'cos' for wave type.
        
    Returns
    -------
    curve : np.ndarray
        Array of shape (n_points, 2) containing curve points.
    """
    A = np.array(A)
    B = np.array(B)

    # Direction vector from A to B
    d = B - A
    L = np.linalg.norm(d)
    d_hat = d / L

    # Perpendicular vector in 2D (rotate 90 degrees CCW)
    n_hat = np.array([-d_hat[1], d_hat[0]])

    # Linear interpolation from A to B
    t = np.linspace(0, 1, n_points)
    line = A + np.outer(t, d)

    # Sinusoidal offset perpendicular to the line
    if mode == 'cos':
        offset = amplitude * np.cos(2 * np.pi * frequency * t)
    else:
        offset = amplitude * np.sin(2 * np.pi * frequency * t)

    curve = line + np.outer(offset, n_hat)
    return curve


def translate_and_rotate_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    dx: float,
    dy: float,
    angle: float,
    degrees: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate and rotate a line segment about its midpoint.
    
    Parameters
    ----------
    p1 : np.ndarray
        First endpoint of the segment.
    p2 : np.ndarray
        Second endpoint of the segment.
    dx : float
        Translation in x direction.
    dy : float
        Translation in y direction.
    angle : float
        Rotation angle (radians by default).
    degrees : bool
        If True, angle is in degrees.
        
    Returns
    -------
    p1_new, p2_new : tuple of np.ndarray
        New positions of the segment endpoints.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    
    if degrees:
        angle = np.deg2rad(angle)

    # Step 1: Translate both points by (dx, dy)
    p1_trans = p1 + np.array([dx, dy])
    p2_trans = p2 + np.array([dx, dy])

    # Step 2: Rotate around the midpoint
    mid = (p1_trans + p2_trans) / 2.0
    v1 = p1_trans - mid
    v2 = p2_trans - mid
    
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return R @ v1 + mid, R @ v2 + mid


# =============================================================================
# Coordinate Conversion Functions
# =============================================================================
def get_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Convert 2D vertices to 3D format with zero y-coordinate.
    
    Parameters
    ----------
    vertices : np.ndarray
        Input vertices of shape (N, 2) or (N*2,).
        
    Returns
    -------
    vertices_3d : np.ndarray
        Output vertices of shape (N, 3) with [x, 0, z] format.
    """
    vertices = vertices.reshape(-1, 2)
    new_vertices = np.vstack([
        vertices[:, 0],
        np.zeros_like(vertices[:, 0]),
        vertices[:, -1]
    ])
    return new_vertices.T


def to_kappa(vertices: np.ndarray) -> np.ndarray:
    """
    Extract x and z coordinates from 3D vertices for curvature computation.
    
    Parameters
    ----------
    vertices : np.ndarray
        Input vertices of shape (N, 3).
        
    Returns
    -------
    kappa : np.ndarray
        Flattened array of [x, z] coordinates.
    """
    vertices_3d = vertices.reshape(-1, 3)
    return vertices_3d[:, [0, 2]].reshape(-1)


def to_3d(vertices: np.ndarray) -> np.ndarray:
    """
    Convert 2D vertices to 3D format with zero y-coordinate.
    
    Parameters
    ----------
    vertices : np.ndarray
        Input vertices of shape (N, 2) or (N*2,).
        
    Returns
    -------
    vertices_3d : np.ndarray
        Output vertices of shape (N, 3) with [x, 0, z] format.
    """
    vertices = vertices.reshape(-1, 2)
    new_vertices = np.vstack([
        vertices[:, 0],
        np.zeros_like(vertices[:, 0]),
        vertices[:, -1]
    ])
    return new_vertices.T


def to_one_hot(vertices: np.ndarray, _3D: bool = True) -> np.ndarray:
    """
    Extract x and z components from 3D vertices or flatten 2D vertices.
    
    Parameters
    ----------
    vertices : np.ndarray
        Input vertices array.
    _3D : bool
        If True, treat as 3D and extract [x, z]. If False, just flatten.
        
    Returns
    -------
    result : np.ndarray
        Flattened array of relevant coordinates.
    """
    if not _3D:
        return vertices.reshape(-1)
    vertices_3d = vertices.reshape(-1, 3)
    return vertices_3d[:, [0, 2]].reshape(-1)