import numpy as np

from risk_pipeline.qubo import BudgetConstraint, build_proxy_qubo, qubo_energy, qubo_to_ising, ising_energy, verify_qubo_ising_equivalence


def test_qubo_ising_equivalence_monte_carlo():
    rng = np.random.default_rng(0)
    n = 6
    risk = rng.uniform(0, 1, size=n)
    C = rng.uniform(0, 1, size=(n, n))
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 0)

    Q, const = build_proxy_qubo(risk, C, lambda_corr=0.3, budget=BudgetConstraint(kind="exact", k=2, lam=5.0))
    assert verify_qubo_ising_equivalence(Q, const=const, n_samples=256, seed=1)


def test_energy_matches_exact_enumeration_small():
    rng = np.random.default_rng(1)
    n = 5
    Q = rng.normal(size=(n, n))
    Q = 0.5 * (Q + Q.T)
    const = 0.7
    h, J, off = qubo_to_ising(Q, const=const)

    for x_int in range(2 ** n):
        x = np.array([(x_int >> b) & 1 for b in range(n)], dtype=int)
        z = 1.0 - 2.0 * x
        e_qubo = qubo_energy(Q, x, const=const)
        e_ising = ising_energy(h, J, z, offset=off)
        assert np.isclose(e_qubo, e_ising, atol=1e-8, rtol=1e-6)
