function theory_out = theory_paper2_final1_2()


    % ---------------- 1. 参数设置 ----------------
    dt   = 0.01;
    Tmax = 12;
    nSteps = ceil(Tmax/dt);
    t_series = (0:nSteps)' * dt;

    N     = 1000;
    k_avg = 10;

    % Weibull Recovery
    aR = 2.0; bR = 0.5;

    % Weibull Attempt
    aI_list = [0.6, 0.8, 1.0, 1.2];

    % Dual modulation
    betaA  = 1.2; betaN  = 1.0;
    kappaA = 0.5; kappaN = 1.0;
    chiA   = 0.4; chiN   = 1.0;

    % Memory kernel g(u) = gamma_mem * exp(-gamma_mem * u)
    % When gamma_mem = 1, g(u) = exp(-u), matching the paper.
    gamma_mem = 1.0;

    % Game payoff
    c   = 0.02;
    uNN = 0.1;  uNA = 0.05;  uAA = 0.01;
    behavior_dt = 1.0;

    % Initial conditions
    init_rho = 0.05;
    init_p   = 0.05;

    % Tail truncation for infection-age support
    epsTail = 1e-8;

    % ---------------- 2. 网络构建 ----------------
    rng(1);
    Adj = ER_network(N, k_avg);
    deg = full(sum(Adj,2));
    deg = max(deg, 1e-20);

    % ---------------- 3. 主循环 ----------------
    numAlpha = numel(aI_list);
    rho_mat = zeros(nSteps+1, numAlpha);
    x_mat   = zeros(nSteps+1, numAlpha);

    % Memory coefficients for exponential kernel
    if gamma_mem > 0
        decayH = exp(-gamma_mem*dt);
        addH   = 1 - decayH;
    else
        decayH = 0;
        addH   = 1;
    end

    fprintf('THEORY: core-mechanism version aligned with paper (no Jensen).\n');

    for ai_idx = 1:numAlpha
        aI = aI_list(ai_idx);
        fprintf('  alpha_I = %.2f ...\n', aI);

        % ---- Infection-age truncation from recovery survival ----
        tauI_max = bR * (log(1/epsTail))^(1/aR);
        L_I = ceil(min(Tmax, tauI_max)/dt) + 1;

        % 1. Recovery conditional probability in each dt interval
        p_rec = weibull_condprob(L_I, dt, aR, bR);
        s_rec = 1 - p_rec;
        pRec = p_rec(:)';
        sRec = s_rec(:)';

        % 2. Strategy-dependent infection-age kernel eta(tau_I | X)
        %    eta_rate_X(n) approximates eta(tau_n | X) used in Eq. (14).
        eta_rate_A = renewal_attempt_kernel(L_I, dt, aI, betaA);
        eta_rate_N = renewal_attempt_kernel(L_I, dt, aI, betaN);
        etaA = eta_rate_A(:);
        etaN = eta_rate_N(:);

        % ---- Initialization ----
        % PA: probability / mass of adopting protection for each node
        PA = zeros(N,1);
        PA(randperm(N, max(1,round(init_p*N)))) = 1.0;

        % Infection-age masses separated by current strategy label
        IA_age = zeros(N, L_I);
        IN_age = zeros(N, L_I);
        init_inf_idx = randperm(N, max(1,round(init_rho*N)));
        IA_age(init_inf_idx,1) = PA(init_inf_idx);
        IN_age(init_inf_idx,1) = 1 - PA(init_inf_idx);

        % Memory state H
        H = zeros(N,1);
        next_update_time = behavior_dt * rand(N,1);

        rho_mat(1, ai_idx) = mean(sum(IA_age,2) + sum(IN_age,2));
        x_mat(1, ai_idx)   = mean(PA);

        t_now = 0;

        for step = 1:nSteps
            t_end = t_now + dt;

            % ---------------- (1) Current infected masses ----------------
            IA_before = sum(IA_age, 2);
            IN_before = sum(IN_age, 2);
            I_before  = IA_before + IN_before;

            % ---------------- (2) Source-side pressure (Eq. 14) ----------
            % Phi_j(t) = int kappa_X * eta(tau_I | X) * I_j(tau_I; t) d tau_I
            phi_A = kappaA * (IA_age * etaA);
            phi_N = kappaN * (IN_age * etaN);
            Phi   = phi_A + phi_N;

            % ---------------- (3) Spatial aggregation (Eq. 15) -----------
            Lambda = Adj * Phi;

            % ---------------- (4) Memory evolution (Eq. 16) --------------
            % For g(u) = gamma * exp(-gamma u):
            % dH/dt = -gamma H + gamma Lambda
            if gamma_mem > 0
                H = decayH * H + addH * Lambda;
            else
                H = Lambda;
            end
            H = max(H, 0);

            % ---------------- (5) Infection (Eq. 17) ---------------------
            SA = max(0, PA - IA_before);
            SN = max(0, (1 - PA) - IN_before);

            p_inf_A = 1 - exp(-chiA * H * dt);
            p_inf_N = 1 - exp(-chiN * H * dt);
            p_inf_A = min(max(p_inf_A,0),1);
            p_inf_N = min(max(p_inf_N,0),1);

            newA = SA .* p_inf_A;
            newN = SN .* p_inf_N;

            % ---------------- (6) Recovery + infection-age shift ---------
            IA_surv = IA_age .* sRec;
            IN_surv = IN_age .* sRec;

            IA_next = zeros(N, L_I);
            IN_next = zeros(N, L_I);
            IA_next(:,1) = newA;
            IN_next(:,1) = newN;
            IA_next(:,2:end) = IA_surv(:,1:end-1);
            IN_next(:,2:end) = IN_surv(:,1:end-1);

            IA_age = IA_next;
            IN_age = IN_next;

            IA_after = sum(IA_age, 2);
            IN_after = sum(IN_age, 2);
            I_after  = IA_after + IN_after;

            % ---------------- (7) Game update (Eq. 3) --------------------
            due = (next_update_time <= t_end + 1e-12);
            if any(due)
                neigh_A = Adj * PA;
                neigh_N = deg - neigh_A;
                rho_global = mean(I_after);
                F_risk = 1 / max(1e-5, 1 - rho_global);

                pi_N = uNN .* neigh_N + uNA .* neigh_A;
                pi_A = (uNA - c) .* neigh_N + uAA .* neigh_A;
                z_N = pi_N;
                z_A = pi_A .* F_risk;

                max_z = max(z_N, z_A);
                prob_A = exp(z_A - max_z) ./ (exp(z_N - max_z) + exp(z_A - max_z) + 1e-20);

                idx_due = find(due);
                prob_due = prob_A(idx_due);
                PA(idx_due) = prob_due;

                % Repartition infected-age mass by the CURRENT strategy state.
                % This keeps the theory aligned with the paper's mechanism,
                % where current strategy determines kappa_X, chi_X, eta(.|X).
                I_age_total = IA_age(idx_due,:) + IN_age(idx_due,:);
                IA_age(idx_due,:) = I_age_total .* prob_due;
                IN_age(idx_due,:) = I_age_total .* (1 - prob_due);

                next_update_time(idx_due) = next_update_time(idx_due) + behavior_dt;
            end

            % ---------------- (8) Record ---------------------------------
            rho_mat(step+1, ai_idx) = mean(sum(IA_age,2) + sum(IN_age,2));
            x_mat(step+1, ai_idx)   = mean(PA);
            t_now = t_end;
        end
    end

    theory_out.t_series = t_series;
    theory_out.rho = rho_mat;
    theory_out.x = x_mat;
    theory_out.aI_list = aI_list;
    theory_out.params = struct('N',N,'dt',dt,'aR',aR,'betaA',betaA, ...
                               'betaN',betaN,'kappaA',kappaA,'kappaN',kappaN, ...
                               'chiA',chiA,'chiN',chiN,'gamma',gamma_mem, ...
                               'behavior_dt',behavior_dt,'init_rho',init_rho, ...
                               'init_p',init_p,'k_avg',k_avg,'Tmax',Tmax, ...
                               'c',c,'uNN',uNN,'uNA',uNA,'uAA',uAA);
end

% ==== Helper Functions ====
function pcond = weibull_condprob(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;
    dH = (tau1./beta).^alpha - (tau0./beta).^alpha;
    pcond = -expm1(-dH);
    pcond = min(max(pcond,0),1)';
end

function eta_rate = renewal_attempt_kernel(L, dt, alpha, beta)
% ------------------------------------------------------------
% Discrete renewal version of Eq. (10):
%   eta(t) = psi(t) + integral_0^t eta(s) psi(t-s) ds
%
% We work with bin probabilities q_mass(n) for the Weibull waiting-time:
%   q_mass(n) = P(T in ((n-1)dt, n dt])
%
% eta_mass(n) is the expected number of attempt events occurring in the
% nth time bin since infection, and eta_rate = eta_mass / dt.
% ------------------------------------------------------------
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;

    surv0 = exp(-(tau0./beta).^alpha);
    surv1 = exp(-(tau1./beta).^alpha);
    q_mass = surv0 - surv1;
    q_mass = max(q_mass, 0);

    eta_mass = zeros(L,1);
    eta_mass(1) = q_mass(1);
    for n = 2:L
        conv_sum = 0;
        for m = 1:n-1
            conv_sum = conv_sum + eta_mass(m) * q_mass(n-m);
        end
        eta_mass(n) = q_mass(n) + conv_sum;
    end

    eta_rate = eta_mass / dt;
end

function A = ER_network(N, k_avg)
    A = sprand(N, N, k_avg/(N-1));
    A = spones(triu(A, 1));
    A = A + A';
end
