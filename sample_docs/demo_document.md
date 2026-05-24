Quantum Computing in Financial Risk Modelling
1. Introduction
Quantum computing represents a paradigm shift in computational finance.
Traditional Monte Carlo simulations for Value-at-Risk (VaR) require O(N) samples
to achieve ε-accuracy. Quantum amplitude estimation reduces this to O(1/ε) queries,
delivering a quadratic speedup over classical methods.
2. Key Algorithms
2.1 Quantum Amplitude Estimation (QAE)
QAE was introduced by Brassard et al. (2002). Given a quantum operator A such that
A|0⟩ = √(1−a)|ψ₀⟩|0⟩ + √a|ψ₁⟩|1⟩, the algorithm estimates the probability a
with precision ε using M = O(1/ε) oracle calls.
2.2 Variational Quantum Eigensolver (VQE)
VQE approximates the ground state of a Hamiltonian H by minimising ⟨ψ(θ)|H|ψ(θ)⟩
over a parameterised ansatz |ψ(θ)⟩. Applications include portfolio optimisation
over Ising-model formulations.
3. Financial Metrics
The 99% VaR for a portfolio with normally distributed returns μ=0.05 and σ=0.12
is calculated as: VaR₀.₉₉ = μ − z₀.₉₉ × σ = 0.05 − 2.326 × 0.12 = −0.229 (22.9% loss).
3.1 CVaR (Conditional Value-at-Risk)
CVaR at the 99% confidence level equals −(μ − σ × φ(z₀.₉₉) / (1−0.99))
where φ is the standard normal PDF. For the above portfolio, CVaR₀.₉₉ ≈ 26.1%.
4. Conclusion
Quantum-enhanced Monte Carlo methods are projected to achieve practical advantage
on error-corrected hardware with ≥1000 logical qubits, expected circa 2028–2030
according to IBM Quantum roadmap projections.
