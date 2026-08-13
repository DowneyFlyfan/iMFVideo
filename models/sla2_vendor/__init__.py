"""Vendored SLA2 (Sparse-Linear Attention v2) triton backend.

Copied from ../SLA/SLA2 (+ sparse_linear_attention/kernel.py backward
kernels) so server training does not need the SLA repo checked out.
CuTeDSL files (cute_linear*.py) are intentionally NOT vendored: use
backend='triton' when constructing SLA2 here.
"""
from .core import SLA2, sla2_dense, full_attention
from .router import Router
