import torch

def F(W):
    return 0.5 * W @ H @ W

def make_hvp_operator(objective, point):
    """Create a flat-vector HVP operator at a fixed point."""
    fixed_point = point.detach().clone()

    def operator(v):
        W = fixed_point.clone().requires_grad_(True)
        value = objective(W)

        gradient, = torch.autograd.grad(
            value, W, create_graph=True
        )
        Hv, = torch.autograd.grad(
            (gradient * v.detach().reshape_as(W)).sum(), W
        )
        return Hv.detach().reshape(-1)

    return operator

def cg(operator, rhs, tol=1e-8, max_iter=200):
    v = torch.zeros_like(rhs)
    r = rhs - operator(v)
    p = r.clone()
    rr = r @ r
    target = tol * torch.linalg.vector_norm(rhs)

    if torch.sqrt(rr) <= target:
        return v, 0

    for iteration in range(1, max_iter + 1):
        Hp = operator(p)
        curvature = p @ Hp
        if curvature <= 0:
            raise RuntimeError("CG encountered non-positive curvature")

        step = rr / curvature
        v = v + step * p
        r = r - step * Hp
        rr_new = r @ r

        relative_residual = (
            torch.sqrt(rr_new) / torch.linalg.vector_norm(rhs)
        )
        print(
            f"CG {iteration}: relative residual={relative_residual.item():.3e}",
            flush=True,
        )

        if torch.sqrt(rr_new) <= target:
            return v, iteration

        p = r + (rr_new / rr) * p
        rr = rr_new

    raise RuntimeError("CG did not converge")

if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    d = 6
    A = torch.randn(d, d)
    H = A.T @ A + torch.eye(d)

    w = torch.randn(d, requires_grad=True)
    g = torch.randn(d)
    # First validate HVP independently.
    operator = make_hvp_operator(F, w)
    direction = torch.randn(d)
    assert torch.allclose(operator(direction), H @ direction)

    # Then validate the CG solution.
    v, iterations = cg(operator, g.reshape(-1))
    exact = torch.linalg.solve(H, g)
    relative_residual = (
        torch.linalg.vector_norm(operator(v) - g)
        / torch.linalg.vector_norm(g)
    )

    print("CG solution:", v)
    print("Exact solution:", exact)
    print("Iterations:", iterations)
    print("Relative residual:", relative_residual.item())

    assert torch.allclose(v, exact, atol=1e-10, rtol=1e-10)
    assert relative_residual < 1e-10
