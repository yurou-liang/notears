import torch

torch.set_default_dtype(torch.float64)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

d = 6
A = torch.randn(d, d)
H = A.T @ A + torch.eye(d)

w = torch.randn(d, requires_grad=True)
g = torch.randn(d)

def F(w):
    return 0.5 * w @ H @ w

def hvp(direction):
    # Rebuild the graph for each call.
    grad_F, = torch.autograd.grad(F(w), w, create_graph=True)
    product, = torch.autograd.grad(
        (grad_F * direction.detach()).sum(), w
    )
    return product.detach()

def cg(operator, rhs, tol=1e-12, max_iter=20):
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

        if torch.sqrt(rr_new) <= target:
            return v, iteration

        p = r + (rr_new / rr) * p
        rr = rr_new

    raise RuntimeError("CG did not converge")

# First validate HVP independently.
direction = torch.randn(d)
assert torch.allclose(hvp(direction), H @ direction)

# Then validate the CG solution.
v, iterations = cg(hvp, g)
exact = torch.linalg.solve(H, g)
relative_residual = (
    torch.linalg.vector_norm(hvp(v) - g)
    / torch.linalg.vector_norm(g)
)

print("CG solution:", v)
print("Exact solution:", exact)
print("Iterations:", iterations)
print("Relative residual:", relative_residual.item())

assert torch.allclose(v, exact, atol=1e-10, rtol=1e-10)
assert relative_residual < 1e-10