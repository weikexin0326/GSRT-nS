import numpy as np
from scipy import sparse


def theory_paper2_final1_2():

    dt = 0.01
    Tmax = 12
    n_steps = int(np.ceil(Tmax / dt))
    t_series = np.arange(n_steps + 1) * dt

    N = 1000
    k_avg = 10

    # Weibull Recovery
    aR = 2.0
    bR = 0.5

    # Weibull Attempt
    aI_list = np.array([0.6, 0.8, 1.0, 1.2], dtype=float)

    # Dual modulation
    betaA = 1.2
    betaN = 1.0
    kappaA = 0.5
    kappaN = 1.0
    chiA = 0.4
    chiN = 1.0

    # Memory kernel g(u) = gamma_mem * exp(-gamma_mem * u)
    # When gamma_mem = 1, g(u) = exp(-u), matching the paper.
    gamma_mem = 1.0

    # Game payoff
    c = 0.02
    uNN = 0.1
    uNA = 0.05
    uAA = 0.01
    behavior_dt = 1.0

    # Initial conditions
    init_rho = 0.05
    init_p = 0.05

    # Tail truncation for infection-age support
    eps_tail = 1e-8


    rng = np.random.default_rng(1)
    Adj = ER_network(N, k_avg, rng)
    deg = np.asarray(Adj.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-20)


    num_alpha = len(aI_list)
    rho_mat = np.zeros((n_steps + 1, num_alpha), dtype=float)
    x_mat = np.zeros((n_steps + 1, num_alpha), dtype=float)

    # Memory coefficients for exponential kernel
    if gamma_mem > 0:
        decayH = np.exp(-gamma_mem * dt)
        addH = 1.0 - decayH
    else:
        decayH = 0.0
        addH = 1.0

    print("THEORY: core-mechanism version aligned with paper (no Jensen).")

    for ai_idx, aI in enumerate(aI_list):
        print(f"  alpha_I = {aI:.2f} ...")

        # ---- Infection-age truncation from recovery survival ----
        tauI_max = bR * (np.log(1.0 / eps_tail)) ** (1.0 / aR)
        L_I = int(np.ceil(min(Tmax, tauI_max) / dt)) + 1

        # 1. Recovery conditional probability in each dt interval
        p_rec = weibull_condprob(L_I, dt, aR, bR)
        s_rec = 1.0 - p_rec

        # 2. Strategy-dependent infection-age kernel eta(tau_I | X)
        eta_rate_A = renewal_attempt_kernel(L_I, dt, aI, betaA)
        eta_rate_N = renewal_attempt_kernel(L_I, dt, aI, betaN)
        etaA = eta_rate_A
        etaN = eta_rate_N

        # ---- Initialization ----
        # PA: probability / mass of adopting protection for each node
        PA = np.zeros(N, dtype=float)
        num_init_p = max(1, int(np.round(init_p * N)))
        PA[rng.permutation(N)[:num_init_p]] = 1.0

        # Infection-age masses separated by current strategy label
        IA_age = np.zeros((N, L_I), dtype=float)
        IN_age = np.zeros((N, L_I), dtype=float)

        num_init_inf = max(1, int(np.round(init_rho * N)))
        init_inf_idx = rng.permutation(N)[:num_init_inf]
        IA_age[init_inf_idx, 0] = PA[init_inf_idx]
        IN_age[init_inf_idx, 0] = 1.0 - PA[init_inf_idx]

        # Memory state H
        H = np.zeros(N, dtype=float)
        next_update_time = behavior_dt * rng.random(N)

        rho_mat[0, ai_idx] = np.mean(IA_age.sum(axis=1) + IN_age.sum(axis=1))
        x_mat[0, ai_idx] = np.mean(PA)

        t_now = 0.0

        # broadcast 用
        sRec = s_rec.reshape(1, -1)

        for step in range(n_steps):
            t_end = t_now + dt

            # ---------------- (1) Current infected masses ----------------
            IA_before = IA_age.sum(axis=1)
            IN_before = IN_age.sum(axis=1)
            I_before = IA_before + IN_before

            # ---------------- (2) Source-side pressure (Eq. 14) ----------
            # Phi_j(t) = int kappa_X * eta(tau_I | X) * I_j(tau_I; t) d tau_I
            phi_A = kappaA * (IA_age @ etaA)
            phi_N = kappaN * (IN_age @ etaN)
            Phi = phi_A + phi_N

            # ---------------- (3) Spatial aggregation (Eq. 15) -----------
            Lambda = Adj @ Phi

            # ---------------- (4) Memory evolution (Eq. 16) --------------
            # For g(u) = gamma * exp(-gamma u):
            # dH/dt = -gamma H + gamma Lambda
            if gamma_mem > 0:
                H = decayH * H + addH * Lambda
            else:
                H = Lambda.copy()

            H = np.maximum(H, 0.0)

            # ---------------- (5) Infection (Eq. 17) ---------------------
            SA = np.maximum(0.0, PA - IA_before)
            SN = np.maximum(0.0, (1.0 - PA) - IN_before)

            p_inf_A = 1.0 - np.exp(-chiA * H * dt)
            p_inf_N = 1.0 - np.exp(-chiN * H * dt)
            p_inf_A = np.clip(p_inf_A, 0.0, 1.0)
            p_inf_N = np.clip(p_inf_N, 0.0, 1.0)

            newA = SA * p_inf_A
            newN = SN * p_inf_N

            # ---------------- (6) Recovery + infection-age shift ---------
            IA_surv = IA_age * sRec
            IN_surv = IN_age * sRec

            IA_next = np.zeros((N, L_I), dtype=float)
            IN_next = np.zeros((N, L_I), dtype=float)

            IA_next[:, 0] = newA
            IN_next[:, 0] = newN

            if L_I > 1:
                IA_next[:, 1:] = IA_surv[:, :-1]
                IN_next[:, 1:] = IN_surv[:, :-1]

            IA_age = IA_next
            IN_age = IN_next

            IA_after = IA_age.sum(axis=1)
            IN_after = IN_age.sum(axis=1)
            I_after = IA_after + IN_after

            # ---------------- (7) Game update (Eq. 3) --------------------
            due = next_update_time <= (t_end + 1e-12)

            if np.any(due):
                neigh_A = Adj @ PA
                neigh_N = deg - neigh_A

                rho_global = np.mean(I_after)
                F_risk = 1.0 / max(1e-5, 1.0 - rho_global)

                pi_N = uNN * neigh_N + uNA * neigh_A
                pi_A = (uNA - c) * neigh_N + uAA * neigh_A

                z_N = pi_N
                z_A = pi_A * F_risk

                max_z = np.maximum(z_N, z_A)
                prob_A = np.exp(z_A - max_z) / (
                    np.exp(z_N - max_z) + np.exp(z_A - max_z) + 1e-20
                )

                idx_due = np.where(due)[0]
                prob_due = prob_A[idx_due]
                PA[idx_due] = prob_due

                # Repartition infected-age mass by the CURRENT strategy state.
                # This keeps the theory aligned with the paper's mechanism,
                # where current strategy determines kappa_X, chi_X, eta(.|X).
                I_age_total = IA_age[idx_due, :] + IN_age[idx_due, :]
                IA_age[idx_due, :] = I_age_total * prob_due[:, None]
                IN_age[idx_due, :] = I_age_total * (1.0 - prob_due)[:, None]

                next_update_time[idx_due] += behavior_dt

            # ---------------- (8) Record ---------------------------------
            rho_mat[step + 1, ai_idx] = np.mean(IA_age.sum(axis=1) + IN_age.sum(axis=1))
            x_mat[step + 1, ai_idx] = np.mean(PA)

            t_now = t_end

    theory_out = {
        "t_series": t_series,
        "rho": rho_mat,
        "x": x_mat,
        "aI_list": aI_list,
        "params": {
            "N": N,
            "dt": dt,
            "aR": aR,
            "bR": bR,
            "betaA": betaA,
            "betaN": betaN,
            "kappaA": kappaA,
            "kappaN": kappaN,
            "chiA": chiA,
            "chiN": chiN,
            "gamma": gamma_mem,
            "behavior_dt": behavior_dt,
            "init_rho": init_rho,
            "init_p": init_p,
            "k_avg": k_avg,
            "Tmax": Tmax,
            "c": c,
            "uNN": uNN,
            "uNA": uNA,
            "uAA": uAA,
        },
    }

    return theory_out


# ==== Helper Functions ====

def weibull_condprob(L, dt, alpha, beta):
    tau0 = np.arange(L, dtype=float) * dt
    tau1 = tau0 + dt
    dH = (tau1 / beta) ** alpha - (tau0 / beta) ** alpha
    pcond = -np.expm1(-dH)
    pcond = np.clip(pcond, 0.0, 1.0)
    return pcond


def renewal_attempt_kernel(L, dt, alpha, beta):
    """
    Discrete renewal version of Eq. (10):
      eta(t) = psi(t) + integral_0^t eta(s) psi(t-s) ds

    We work with bin probabilities q_mass(n) for the Weibull waiting-time:
      q_mass(n) = P(T in ((n-1)dt, n dt])

    eta_mass(n) is the expected number of attempt events occurring in the
    nth time bin since infection, and eta_rate = eta_mass / dt.
    """
    tau0 = np.arange(L, dtype=float) * dt
    tau1 = tau0 + dt

    surv0 = np.exp(-((tau0 / beta) ** alpha))
    surv1 = np.exp(-((tau1 / beta) ** alpha))
    q_mass = surv0 - surv1
    q_mass = np.maximum(q_mass, 0.0)

    eta_mass = np.zeros(L, dtype=float)
    eta_mass[0] = q_mass[0]

    for n in range(1, L):
        conv_sum = 0.0
        for m in range(n):
            conv_sum += eta_mass[m] * q_mass[n - 1 - m]
        eta_mass[n] = q_mass[n] + conv_sum

    eta_rate = eta_mass / dt
    return eta_rate


def ER_network(N, k_avg, rng=None):
    """
    MATLAB:
        A = sprand(N, N, k_avg/(N-1));
        A = spones(triu(A, 1));
        A = A + A';


    """
    if rng is None:
        rng = np.random.default_rng()

    p = k_avg / (N - 1)

    upper_mask = rng.random((N, N)) < p
    upper_mask = np.triu(upper_mask, k=1)

    rows, cols = np.where(upper_mask)
    data = np.ones(len(rows), dtype=float)

    A_upper = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = A_upper + A_upper.T
    return A


if __name__ == "__main__":
    result = theory_paper2_final1_2()

    print("\nDone.")
    print("t_series shape:", result["t_series"].shape)
    print("rho shape:", result["rho"].shape)
    print("x shape:", result["x"].shape)
    print("aI_list:", result["aI_list"])
