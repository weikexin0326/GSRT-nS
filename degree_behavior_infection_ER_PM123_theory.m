function out = degree_behavior_infection_ER_PM123_theory()
% ============================================================
% Degree-Behavior-Infection correlation experiment for GSRT-nSIS
% on a fixed ER network, extended to PM1 / PM2 / PM3.
% ------------------------------------------------------------
% This version is modified STRICTLY on top of the original experiment idea:
%   - keep the same ER network / degree grouping / PM1-PM3 overlay;
%   - keep repeated runs + averaging, to reduce the effect of random
%     initial allocations of infection/protection and random behaviour clocks;
%   - replace the bottom-layer Monte Carlo simulator by the node-level
%     THEORY solver aligned with "theory代码.txt".
%
% The aligned theory solver uses:
%   1) infection-age densities IA_age / IN_age,
%   2) source pressure Phi from the renewal attempt kernel eta(.),
%   3) exponential memory H,
%   4) Weibull recovery,
%   5) asynchronous game update with local neighbour payoffs,
%   6) repartition of infected-age mass by current strategy probability PA.
%
% Degree-resolved observables:
%   1) x(k)    : steady-state average protection level in degree group k
%   2) rho(k)  : steady-state average infection level in degree group k
%   3) qIN(k)  : steady-state average I-N mass in degree group k
%   4) tauA(k) : first time x_k(t) reaches
%                x_k(0) + frac_tauA * (x_k^* - x_k(0))
%
% Output:
%   out : struct containing network, grouped statistics, PM1/2/3 results,
%         and figure handle.
%
% Notes:
%   - If Parallel Computing Toolbox is unavailable, replace PARFOR by FOR.
%   - This code saves the figure to folder "四月实验结果图" as FIG and JPG.
% ============================================================

    clc; clear; close all;

    %% ---------------- 1. Global parameters ----------------
    dt   = 0.01;
    Tmax = 20;
    t_series = (0:dt:Tmax)';
    Tn = numel(t_series);

    % Network
    N = 1000;
    k_avg = 10;
    p_edge = k_avg/(N-1);

    % Recovery Weibull
    aR = 2.0;
    bR = 0.5;

    % Attempt renewal kernel (same representative alpha_I as original experiment)
    aI = 0.8;
    betaA  = 1.2;
    betaN  = 1.0;

    % Dual modulation
    kappaA = 0.5;
    kappaN = 1.0;
    chiA   = 0.4;
    chiN   = 1.0;

    % Exponential memory kernel g(u)=gamma*exp(-gamma*u)
    gamma_mem = 1.0;

    % Behaviour update period
    behavior_dt = 1.0;

    % Initial conditions
    init_rho = 0.05;
    init_p   = 0.05;

    % Repetitions (average over random initial allocations / clocks)
    run_reps = 20;
    base_seed = 1;

    % Degree-group filtering for stability on ER network
    min_nodes_per_group = 8;

    % Steady-state window: last 20% of the trajectory
    steady_frac = 0.20;

    % tauA definition
    frac_tauA = 0.50;
    tau_tol = 1e-8;

    % Payoff structures: PM1 / PM2 / PM3
    PM_display = {'PM1', 'PM2', 'PM3'};
    PM_colors = [0.0000 0.4470 0.7410; ...
                 0.8500 0.3250 0.0980; ...
                 0.4660 0.6740 0.1880];
    PM_markers = {'o', 's', '^'};

    PMs(1) = struct('name','PM1','uAA',0.01,'uNA',0.05,'uNN',0.10,'c',0.02);
    PMs(2) = struct('name','PM2','uAA',0.05,'uNA',0.05,'uNN',0.10,'c',0.02);
    PMs(3) = struct('name','PM3','uAA',0.10,'uNA',0.05,'uNN',0.01,'c',0.02);
    numPM = numel(PMs);

    fprintf('Running THEORY-aligned degree-behavior-infection experiment on ER network\n');
    fprintf('N=%d, <k>=%.1f, dt=%.3g, Tmax=%.2f, alpha_I=%.2f, reps=%d\n', ...
            N, k_avg, dt, Tmax, aI, run_reps);
    fprintf('Bottom solver aligned to theory代码.txt\n');

    %% ---------------- 2. Fixed ER network ----------------
    rng(base_seed);
    Adj = sprand(N, N, p_edge);
    Adj = spones(triu(Adj, 1));
    Adj = Adj + Adj';
    Adj = Adj - diag(diag(Adj));
    Adj = sparse(Adj);

    deg = full(sum(Adj, 2));

    %% ---------------- 3. Degree grouping ----------------
    deg_values_all = unique(deg);
    keep_mask = false(size(deg_values_all));
    deg_counts_all = zeros(size(deg_values_all));
    for ii = 1:numel(deg_values_all)
        nk = sum(deg == deg_values_all(ii));
        deg_counts_all(ii) = nk;
        keep_mask(ii) = (nk >= min_nodes_per_group);
    end

    degree_values = deg_values_all(keep_mask);
    degree_counts = deg_counts_all(keep_mask);
    K = numel(degree_values);
    if K == 0
        error('No degree groups survived the min_nodes_per_group filter.');
    end

    groups = cell(K,1);
    for kk = 1:K
        groups{kk} = find(deg == degree_values(kk));
    end

    ss_start_idx = max(1, floor((1 - steady_frac) * Tn) + 1);
    ss_idx = ss_start_idx:Tn;

    %% ---------------- 4. Precompute theory kernels ----------------
    nSteps = Tn - 1;
    L_I = nSteps + 1;  % long enough for the full trajectory

    p_rec = weibull_condprob(L_I, dt, aR, bR);
    sRec = 1 - p_rec;
    etaA = renewal_attempt_kernel(L_I, dt, aI, betaA);
    etaN = renewal_attempt_kernel(L_I, dt, aI, betaN);

    if gamma_mem > 0
        decayH = exp(-gamma_mem * dt);
        addH   = (1 - decayH) / gamma_mem;
    else
        decayH = 0;
        addH   = 1;
    end

    %% ---------------- 5. Loop over PM1 / PM2 / PM3 ----------------
    results = repmat(struct(), 1, numPM);

    for pm_idx = 1:numPM
        pm = PMs(pm_idx);
        fprintf('  Processing %s ...\n', pm.name);

        xk_all   = zeros(Tn, K, run_reps);
        rhok_all = zeros(Tn, K, run_reps);
        qINk_all = zeros(Tn, K, run_reps);

        xk_ss_rep   = zeros(run_reps, K);
        rhok_ss_rep = zeros(run_reps, K);
        qINk_ss_rep = zeros(run_reps, K);
        tauA_rep    = nan(run_reps, K);

        parfor rep = 1:run_reps
            rng(base_seed + 1000 * pm_idx + rep);

            % ---------- random initial allocation (kept from original experiment) ----------
            PA0 = zeros(N,1);
            I0  = zeros(N,1);

            num_init_p = max(1, round(init_p * N));
            num_init_I = max(1, round(init_rho * N));

            idxA0 = randperm(N, num_init_p);
            idxI0 = randperm(N, num_init_I);
            PA0(idxA0) = 1;
            I0(idxI0)  = 1;

            next_update_time0 = behavior_dt * rand(N,1);

            [xk_run, rhok_run, qINk_run] = run_one_theory_realization( ...
                Adj, deg, groups, t_series, dt, nSteps, L_I, ...
                PA0, I0, next_update_time0, ...
                etaA, etaN, sRec, decayH, addH, ...
                kappaA, kappaN, chiA, chiN, ...
                gamma_mem, behavior_dt, pm);

            xk_ss_i   = mean(xk_run(ss_idx,:),   1);
            rhok_ss_i = mean(rhok_run(ss_idx,:), 1);
            qINk_ss_i = mean(qINk_run(ss_idx,:), 1);

            tauA_i = nan(1, K);
            for kk = 1:K
                tauA_i(kk) = compute_tauA_from_series( ...
                    t_series, xk_run(:,kk), xk_ss_i(kk), frac_tauA, tau_tol);
            end

            xk_all(:,:,rep)   = xk_run;
            rhok_all(:,:,rep) = rhok_run;
            qINk_all(:,:,rep) = qINk_run;

            xk_ss_rep(rep,:)   = xk_ss_i;
            rhok_ss_rep(rep,:) = rhok_ss_i;
            qINk_ss_rep(rep,:) = qINk_ss_i;
            tauA_rep(rep,:)    = tauA_i;
        end

        % ---------- aggregate ----------
        xk_mean   = mean(xk_all,   3);
        rhok_mean = mean(rhok_all, 3);
        qINk_mean = mean(qINk_all, 3);

        xk_ss   = mean(xk_ss_rep,   1);
        rhok_ss = mean(rhok_ss_rep, 1);
        qINk_ss = mean(qINk_ss_rep, 1);
        tauA_k  = mean_omit_nan(tauA_rep, 1);

        x_node_ss = sum(xk_ss .* degree_counts') / sum(degree_counts);
        x_stub_ss = sum((degree_values .* degree_counts)' .* xk_ss) / sum(degree_values .* degree_counts);
        delta_x = x_stub_ss - x_node_ss;

        results(pm_idx).name = pm.name;
        results(pm_idx).display_name = PM_display{pm_idx};
        results(pm_idx).color = PM_colors(pm_idx,:);
        results(pm_idx).marker = PM_markers{pm_idx};
        results(pm_idx).payoff = pm;

        results(pm_idx).xk_mean = xk_mean;
        results(pm_idx).rhok_mean = rhok_mean;
        results(pm_idx).qINk_mean = qINk_mean;

        results(pm_idx).xk_ss = xk_ss;
        results(pm_idx).rhok_ss = rhok_ss;
        results(pm_idx).qINk_ss = qINk_ss;
        results(pm_idx).tauA_k = tauA_k;

        results(pm_idx).x_node_ss = x_node_ss;
        results(pm_idx).x_stub_ss = x_stub_ss;
        results(pm_idx).delta_x = delta_x;

        fprintf('    %s done: x* = %.4f, x_stub = %.4f, Delta_x = %.4f\n', ...
                pm.name, x_node_ss, x_stub_ss, delta_x);
    end

    %% ---------------- 6. Plot overlay figure ----------------
    if ~exist('四月实验结果图', 'dir')
        mkdir('四月实验结果图');
    end

    fig = figure('Color', 'w', 'Position', [100 60 1180 860]);

    % --- subplot 1: rho(k) ---
    subplot(2,2,1); hold on;
    for pm_idx = 1:numPM
        r = results(pm_idx);
        plot(degree_values, r.rhok_ss, ...
            'Color', r.color, 'Marker', r.marker, 'LineStyle', '-', ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', r.display_name);
    end
    grid on; box on;
    xlabel('Degree k'); ylabel('\rho(k)');
    title('Steady-state infection by degree');
    legend('Location', 'best');

    % --- subplot 2: x(k) ---
    subplot(2,2,2); hold on;
    for pm_idx = 1:numPM
        r = results(pm_idx);
        plot(degree_values, r.xk_ss, ...
            'Color', r.color, 'Marker', r.marker, 'LineStyle', '-', ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', r.display_name);
    end
    grid on; box on;
    xlabel('Degree k'); ylabel('x(k)');
    title('Steady-state protection by degree');
    legend('Location', 'best');

    % --- subplot 3: qIN(k) ---
    subplot(2,2,3); hold on;
    for pm_idx = 1:numPM
        r = results(pm_idx);
        plot(degree_values, r.qINk_ss, ...
            'Color', r.color, 'Marker', r.marker, 'LineStyle', '-', ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', r.display_name);
    end
    grid on; box on;
    xlabel('Degree k'); ylabel('q_{IN}(k)');
    title('Steady-state I-N mass by degree');
    legend('Location', 'best');

    % --- subplot 4: tauA(k) ---
    subplot(2,2,4); hold on;
    for pm_idx = 1:numPM
        r = results(pm_idx);
        plot(degree_values, r.tauA_k, ...
            'Color', r.color, 'Marker', r.marker, 'LineStyle', '-', ...
            'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', r.display_name);
    end
    grid on; box on;
    xlabel('Degree k'); ylabel('\tau_A(k)');
    title(sprintf('Protection response time by degree (%.0f%% rise)', 100 * frac_tauA));
    legend('Location', 'best');

    delta_summary = sprintf('\\Delta_x = x_{stub}-x^*: PM1 %.4f, PM2 %.4f, PM3 %.4f', ...
        results(1).delta_x, results(2).delta_x, results(3).delta_x);

    sgtitle({['GSRT-nSIS on ER network: degree-resolved experiment (PM1 / PM2 / PM3)'], ...
             ['theory-aligned solver,  \alpha_I = ' num2str(aI) ...
              ',  N = ' num2str(N) ',  <k> = ' num2str(k_avg) ...
              ',  reps = ' num2str(run_reps) ',  ' delta_summary]});

    savefig(fig, fullfile('四月实验结果图', 'degree_behavior_ER_PM123_theory.fig'));
    saveas(fig, fullfile('四月实验结果图', 'degree_behavior_ER_PM123_theory.jpg'));

    %% ---------------- 7. Pack output ----------------
    out = struct();
    out.t_series = t_series;
    out.Adj = Adj;
    out.deg = deg;
    out.degree_values = degree_values;
    out.degree_counts = degree_counts;
    out.groups = groups;
    out.results = results;
    out.figure_handle = fig;
    out.params = struct( ...
        'dt', dt, 'Tmax', Tmax, 'N', N, 'k_avg', k_avg, 'p_edge', p_edge, ...
        'aR', aR, 'bR', bR, 'aI', aI, 'betaA', betaA, 'betaN', betaN, ...
        'kappaA', kappaA, 'kappaN', kappaN, 'chiA', chiA, 'chiN', chiN, ...
        'gamma_mem', gamma_mem, 'behavior_dt', behavior_dt, ...
        'init_rho', init_rho, 'init_p', init_p, 'run_reps', run_reps, ...
        'base_seed', base_seed, 'steady_frac', steady_frac, ...
        'frac_tauA', frac_tauA, 'tau_tol', tau_tol, ...
        'min_nodes_per_group', min_nodes_per_group, 'PMs', PMs);
end

%% ================= Core theory-aligned realization =================
function [xk_run, rhok_run, qINk_run] = run_one_theory_realization( ...
    Adj, deg, groups, t_series, dt, nSteps, L_I, ...
    PA0, I0, next_update_time, ...
    etaA, etaN, sRec, decayH, addH, ...
    kappaA, kappaN, chiA, chiN, ...
    gamma_mem, behavior_dt, pm)

    N = numel(PA0);
    K = numel(groups);
    Tn = numel(t_series);

    % Infection-age masses
    IA_age = zeros(N, L_I);
    IN_age = zeros(N, L_I);
    IA_age(:,1) = PA0 .* I0;
    IN_age(:,1) = (1 - PA0) .* I0;

    % Protection probability state
    PA = PA0;

    % Memory state
    H = zeros(N,1);

    xk_run   = zeros(Tn, K);
    rhok_run = zeros(Tn, K);
    qINk_run = zeros(Tn, K);

    % Record t = 0
    [xk_run(1,:), rhok_run(1,:), qINk_run(1,:)] = ...
        compute_degree_metrics_theory(PA, IA_age, IN_age, groups);

    t_now = 0;
    for step = 1:nSteps
        t_end = t_now + dt;

        % ---------------- (1) Current infected masses ----------------
        IA_before = sum(IA_age, 2);
        IN_before = sum(IN_age, 2);
        I_before  = IA_before + IN_before;

        % ---------------- (2) Source pressure ------------------------
        phi_A = kappaA * (IA_age * etaA(:));
        phi_N = kappaN * (IN_age * etaN(:));
        Phi   = phi_A + phi_N;

        % ---------------- (3) Spatial aggregation --------------------
        Lambda = Adj * Phi;

        % ---------------- (4) Memory evolution -----------------------
        if gamma_mem > 0
            H = decayH * H + addH * Lambda;
        else
            H = Lambda;
        end
        H = max(H, 0);

        % ---------------- (5) Infection ------------------------------
        SA = max(0, PA - IA_before);
        SN = max(0, (1 - PA) - IN_before);

        p_inf_A = 1 - exp(-chiA * H * dt);
        p_inf_N = 1 - exp(-chiN * H * dt);
        p_inf_A = min(max(p_inf_A, 0), 1);
        p_inf_N = min(max(p_inf_N, 0), 1);

        newA = SA .* p_inf_A;
        newN = SN .* p_inf_N;

        % ---------------- (6) Recovery + infection-age shift --------
        IA_surv = IA_age .* sRec(:)';
        IN_surv = IN_age .* sRec(:)';

        IA_next = zeros(N, L_I);
        IN_next = zeros(N, L_I);
        IA_next(:,1) = newA;
        IN_next(:,1) = newN;
        if L_I > 1
            IA_next(:,2:end) = IA_surv(:,1:end-1);
            IN_next(:,2:end) = IN_surv(:,1:end-1);
        end

        IA_age = IA_next;
        IN_age = IN_next;

        IA_after = sum(IA_age, 2);
        IN_after = sum(IN_age, 2);
        I_after  = IA_after + IN_after;

        % ---------------- (7) Asynchronous game update --------------
        due = (next_update_time <= t_end + 1e-12);
        if any(due)
            neigh_A = Adj * PA;
            neigh_N = deg - neigh_A;
            rho_global = mean(I_after);
            F_risk = 1 / max(1e-5, 1 - rho_global);

            pi_N = pm.uNN .* neigh_N + pm.uNA .* neigh_A;
            pi_A = (pm.uNA - pm.c) .* neigh_N + pm.uAA .* neigh_A;
            z_N = pi_N;
            z_A = pi_A .* F_risk;

            max_z = max(z_N, z_A);
            prob_A = exp(z_A - max_z) ./ (exp(z_N - max_z) + exp(z_A - max_z) + 1e-20);

            idx_due = find(due);
            prob_due = prob_A(idx_due);
            PA(idx_due) = prob_due;

            % Repartition infected-age mass according to current PA,
            % exactly in the spirit of theory代码.txt
            I_age_total = IA_age(idx_due,:) + IN_age(idx_due,:);
            IA_age(idx_due,:) = I_age_total .* prob_due;
            IN_age(idx_due,:) = I_age_total .* (1 - prob_due);

            next_update_time(idx_due) = next_update_time(idx_due) + behavior_dt;
        end

        % ---------------- (8) Record --------------------------------
        [xk_run(step+1,:), rhok_run(step+1,:), qINk_run(step+1,:)] = ...
            compute_degree_metrics_theory(PA, IA_age, IN_age, groups);
        t_now = t_end;
    end
end

%% ================= Helper functions =================
function [xk, rhok, qINk] = compute_degree_metrics_theory(PA, IA_age, IN_age, groups)
    K = numel(groups);
    xk   = zeros(1, K);
    rhok = zeros(1, K);
    qINk = zeros(1, K);

    I  = sum(IA_age,2) + sum(IN_age,2);
    IN = sum(IN_age,2);

    for kk = 1:K
        idx = groups{kk};
        xk(kk)   = mean(PA(idx));
        rhok(kk) = mean(I(idx));
        qINk(kk) = mean(IN(idx));
    end
end

function tauA = compute_tauA_from_series(t_series, x_series, x_ss, frac_tauA, tol)
    x0 = x_series(1);
    dx = x_ss - x0;

    if dx <= tol
        tauA = NaN;
        return;
    end

    target = x0 + frac_tauA * dx;
    hit = find(x_series >= target, 1, 'first');

    if isempty(hit)
        tauA = NaN;
    else
        tauA = t_series(hit);
    end
end

function m = mean_omit_nan(X, dim)
    if nargin < 2, dim = 1; end
    mask = ~isnan(X);
    num = sum(X .* mask, dim);
    den = sum(mask, dim);
    m = num ./ den;
    m(den == 0) = NaN;
end

function pcond = weibull_condprob(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;
    dH = (tau1 / beta).^alpha - (tau0 / beta).^alpha;
    pcond = -expm1(-dH);
    pcond = min(max(pcond, 0), 1);
end

function eta_rate = renewal_attempt_kernel(L, dt, alpha, beta)
% Discrete renewal kernel aligned with theory代码.txt
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;

    surv0 = exp(-(tau0 / beta).^alpha);
    surv1 = exp(-(tau1 / beta).^alpha);
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
