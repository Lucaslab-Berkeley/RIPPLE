"""Tests for helpers the particle-polishing code relies on."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_motion_correction import correct_motion_two_grids

from ripple.core.core_utils import _make_differentiable_refine_manager


def test_two_grid_correction_keeps_the_spline_grids_gradient():
    """Polishing passes its spline grids straight to the two-grid correction."""
    image = torch.randn(4, 32, 24)
    base = CubicCatmullRomGrid3d.from_grid_data(torch.randn(2, 3, 2, 2))
    new = CubicCatmullRomGrid3d(resolution=(3, 2, 2), n_channels=2)
    with torch.no_grad():
        new._data.normal_(0, 0.5)  # pylint: disable=protected-access

    corrected = correct_motion_two_grids(
        image=image,
        new_deformation_field=new,
        base_deformation_field=base,
        pixel_spacing=1.0,
    )
    corrected.square().sum().backward()

    assert corrected.shape == image.shape
    grads = [p.grad for p in new.parameters()]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in base.parameters())


def test_refine_manager_disables_its_movie_config(monkeypatch):
    """The manager comes from the refine config, with its movie config off."""
    # Importing Leopard-EM's backend turns autograd off: keep it on for later tests
    grad_enabled = torch.is_grad_enabled()
    try:
        pytest.importorskip("leopard_em")
        # pylint: disable-next=import-outside-toplevel
        from leopard_em.pydantic_models.managers import RefineTemplateManager
    finally:
        torch.set_grad_enabled(grad_enabled)

    loaded = SimpleNamespace(movie_config=SimpleNamespace(enabled=True))
    monkeypatch.setattr(
        RefineTemplateManager, "from_yaml", classmethod(lambda cls, path: loaded)
    )

    manager = _make_differentiable_refine_manager("refine.yaml")

    assert manager is loaded
    assert manager.movie_config.enabled is False


def test_refine_manager_leaves_autograd_on():
    """Importing Leopard-EM's backend turns autograd off; the helper restores it.

    Run in a fresh interpreter, so that Leopard-EM is imported for the first time.
    """
    pytest.importorskip("leopard_em")
    code = """
import sys
import torch
from ripple.core.core_utils import _make_differentiable_refine_manager
assert not any(name.startswith("leopard_em") for name in sys.modules)
try:
    _make_differentiable_refine_manager("missing_refine_config.yaml")
except Exception:  # the config does not exist; only the import matters here
    pass
assert any(name.startswith("leopard_em") for name in sys.modules)
assert torch.is_grad_enabled(), "autograd was left off"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
