function out = diff_distfamily_rho_x_compare_beta()
% ============================================================
% PM1: Pure-theory comparison of transmission-attempt waiting-time families
% under two matching schemes:
%   (1) median-matched across families
%   (2) fixed beta_A = 1.2, beta_N = 1.0 across families
%
% Theory semantics are aligned with the uploaded
% theory_paper2_final1_2() "core-mechanism" version:
%   - No explicit attempt-age mass states EA/EN
%   - Source pressure computed from infection-age densities via eta(tau_I|X)
%   - Infection probability: P_inf = 1 - exp(-chi_X * H * dt)
%   - No Jensen correction
%
% Output:
%   out.t_series
%   out.curves.(scheme).(dist).rho / x
%   out.tss.(scheme).(dist)
%   out.fig_path
%   out.jpg_path
%
% Result figure is saved to folder:
%   四月实验结果图
% in both .fig and .jpg formats.
% ============================================================

    clc; clear; close all;
    %% ---------------- 1. Base parameters (PM1, same as theory baseline) ----------------
    P = struct();
    P.dt        = 0.01;
    P.Tmax_plot  = 40;   % displayed horizon
    P.Tmax_calc  = 60;   % longer horizon for steady-state-time detection

    P.N     = 1000;
    P.k_avg = 10;

    % Weibull recovery
    P.aR = 2.0;
    P.bR = 0.5;

    % Dual modulation
    P.kappaA = 0.5;
    P.kappaN = 1.0;
    P.chiA   = 0.4;
    P.chiN   = 1.0;

    % Memory kernel g(u) = gamma * exp(-gamma u)
    P.gamma_mem = 1.0;

    % PM1 game payoff
    P.c   = 0.02;
    P.uNN = 0.10;
    P.uNA = 0.05;
    P.uAA = 0.01;
    P.behavior_dt = 1.0;

    % Initial conditions
    P.init_rho = 0.05;
    P.init_p   = 0.05;

    % Tail truncation for infection-age support
    P.epsTail = 1e-8;

    % Fixed network and initialization seeds for comparability
    P.net_seed  = 1;
    P.init_seed = 1;

    %% ---------------- 2. Waiting-time family setup ----------------
    % Reference Weibull baseline used to define median-matched scheme
    alphaW = 0.8;
    betaN_ref = 1.0;
    betaA_ref = 1.2;

    % Additional family shapes
    alphaLL = 2.0;   % unimodal hazard
    aLo     = 2.0;   % Lomax shape

    % Median times implied by the reference Weibull baseline
    mN = betaN_ref * (log(2))^(1/alphaW);
    mA = betaA_ref * (log(2))^(1/alphaW);

    % Median-matched betas
    beta_median = struct();
    beta_median.Weibull.betaN = betaN_ref;
    beta_median.Weibull.betaA = betaA_ref;

    beta_median.LogLogistic.betaN = mN;
    beta_median.LogLogistic.betaA = mA;

    beta_median.Lomax.betaN = mN / (2^(1/aLo) - 1);
    beta_median.Lomax.betaA = mA / (2^(1/aLo) - 1);

    % Fixed-beta scheme
    beta_fixed = struct();
    beta_fixed.Weibull.betaN = betaN_ref;
    beta_fixed.Weibull.betaA = betaA_ref;

    beta_fixed.LogLogistic.betaN = betaN_ref;
    beta_fixed.LogLogistic.betaA = betaA_ref;

    beta_fixed.Lomax.betaN = betaN_ref;
    beta_fixed.Lomax.betaA = betaA_ref;

    %% ---------------- 3. Shared network ----------------
    rng(P.net_seed);
    Adj = ER_network(P.N, P.k_avg);
    Adj = spones(Adj);
    Adj = Adj - diag(diag(Adj));
    Adj = spones(0.5 * (Adj + Adj'));

    %% ---------------- 4. Run theory under two schemes ----------------
    dist_names = {'Weibull', 'LogLogistic', 'Lomax'};
    scheme_names = {'median_matched', 'fixed_beta'};

    curves = struct();
    tss = struct();

    fprintf('Running pure-theory dual-scheme comparison...\n');

    for s = 1:numel(scheme_names)
        scheme = scheme_names{s};
        for d = 1:numel(dist_names)
            dist = dist_names{d};

            cfg = struct();
            cfg.dist_name = dist;
            switch dist
                case 'Weibull'
                    cfg.shape = alphaW;
                case 'LogLogistic'
                    cfg.shape = alphaLL;
                case 'Lomax'
                    cfg.shape = aLo;
            end

            if strcmp(scheme, 'median_matched')
                cfg.betaN = beta_median.(dist).betaN;
                cfg.betaA = beta_median.(dist).betaA;
            else
                cfg.betaN = beta_fixed.(dist).betaN;
                cfg.betaA = beta_fixed.(dist).betaA;
            end

            fprintf('  Scheme=%s, Dist=%s, betaN=%.6f, betaA=%.6f\n', ...
                scheme, dist, cfg.betaN, cfg.betaA);

            res = run_core_theory_family(P, Adj, cfg);
            curves.(scheme).(dist) = res;
            tss.(scheme).(dist) = compute_joint_steady_time(res.t_series, res.rho, res.x);
        end
    end

    %% ---------------- 5. Plot ----------------
    fig = figure('Color','w','Position',[80 90 1500 620]);
    tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    %title(tl, 'comparison of waiting-time families', ...
    %    'FontWeight','bold','FontSize',20);

    colors = struct();
    colors.Weibull     = [0.22 0.62 0.17];
    colors.LogLogistic = [0.00 0.4470 0.7410];
    colors.Lomax       = [0.8500 0.3250 0.0980];

    markers = struct();
    markers.Weibull     = 'o';
    markers.LogLogistic = 's';
    markers.Lomax       = 'd';

    % indices for displayed horizon
    plot_mask = curves.median_matched.Weibull.t_series <= P.Tmax_plot + 1e-12;

    % -------- panel (a): rho(t) --------
    ax1 = nexttile; hold(ax1,'on'); box(ax1,'on');

    % plot curves first
    plot_handles_1 = gobjects(6,1);
    labels_1 = cell(6,1);
    k = 0;
    for d = 1:numel(dist_names)
        dist = dist_names{d};
        c = colors.(dist);
        marker = markers.(dist);

        % median-matched: solid line
        k = k + 1;
        rr = curves.median_matched.(dist);
        plot_handles_1(k) = plot(ax1, rr.t_series(plot_mask), rr.rho(plot_mask), '-', 'Color', c, 'LineWidth', 2.6);
        labels_1{k} = sprintf('%s (median-matched)', pretty_dist_name(dist));

        % fixed-beta: dashed + sparse markers
        k = k + 1;
        rr2 = curves.fixed_beta.(dist);
        mkIdx = find(plot_mask);
        mkIdx = mkIdx(1:40:end);
        plot_handles_1(k) = plot(ax1, rr2.t_series(plot_mask), rr2.rho(plot_mask), '--', 'Color', c, 'LineWidth', 1.9, ...
            'Marker', marker, 'MarkerIndices', 1:200:sum(plot_mask), 'MarkerSize', 5.5, ...
            'MarkerFaceColor', 'w', 'MarkerEdgeColor', c);
        labels_1{k} = sprintf('%s (fixed-\\beta)', pretty_dist_name(dist));
    end

    xlabel(ax1, 'Time (t)', 'FontSize', 18);
    ylabel(ax1, '\rho(t)', 'FontSize', 18);
    title(ax1, '(a)', 'FontSize', 18, 'FontWeight', 'normal');
    set(ax1, 'FontSize', 15, 'LineWidth', 1.2, 'XLim', [0 P.Tmax_plot]);

    % add steady-state-time vertical lines
    yl1 = ylim(ax1);
    pad1 = 0.06 * max(yl1(2) - yl1(1), 1e-6);   % 顶部留 6% 空白
    yl1 = [yl1(1), yl1(2) + pad1];
    ylim(ax1, yl1);
    draw_tss_lines(ax1, tss, curves, 'rho', yl1);
    grid(ax1, 'on');

    legend(ax1, plot_handles_1, labels_1, 'Location', 'southeast', 'FontSize', 12, 'Box', 'off', 'Interpreter', 'tex');

    % -------- panel (b): x(t) --------
    ax2 = nexttile; hold(ax2,'on'); box(ax2,'on');

    plot_handles_2 = gobjects(6,1);
    labels_2 = cell(6,1);
    k = 0;
    for d = 1:numel(dist_names)
        dist = dist_names{d};
        c = colors.(dist);
        marker = markers.(dist);

        k = k + 1;
        rr = curves.median_matched.(dist);
        plot_handles_2(k) = plot(ax2, rr.t_series(plot_mask), rr.x(plot_mask), '-', 'Color', c, 'LineWidth', 2.6);
        labels_2{k} = sprintf('%s (median-matched)', pretty_dist_name(dist));

        k = k + 1;
        rr2 = curves.fixed_beta.(dist);
        mkIdx = find(plot_mask);
        mkIdx = mkIdx(1:40:end);
        plot_handles_2(k) = plot(ax2, rr2.t_series(plot_mask), rr2.x(plot_mask), '--', 'Color', c, 'LineWidth', 1.9, ...
            'Marker', marker, 'MarkerIndices', 1:200:sum(plot_mask), 'MarkerSize', 5.5, ...
            'MarkerFaceColor', 'w', 'MarkerEdgeColor', c);
        labels_2{k} = sprintf('%s (fixed-\\beta)', pretty_dist_name(dist));
    end

    xlabel(ax2, 'Time (t)', 'FontSize', 18);
    ylabel(ax2, 'x(t)', 'FontSize', 18);
    title(ax2, '(b)', 'FontSize', 18, 'FontWeight', 'normal');
    set(ax2, 'FontSize', 15, 'LineWidth', 1.2, 'XLim', [0 P.Tmax_plot]);

    yl2 = ylim(ax2);
    pad2 = 0.06 * max(yl2(2) - yl2(1), 1e-6);   % 顶部留 6% 空白
    yl2 = [yl2(1), yl2(2) + pad2];
    ylim(ax2, yl2);
    draw_tss_lines(ax2, tss, curves, 'x', yl2);
    grid(ax2, 'on');

    legend(ax2, plot_handles_2, labels_2, 'Location', 'southeast', 'FontSize', 12, 'Box', 'off', 'Interpreter', 'tex');

    %% ---------------- 6. Save figure ----------------
    outdir = '四月实验结果图';
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    fig_path = fullfile(outdir, 'diff_distfamily_rho_x_compare_beta.fig');
    jpg_path = fullfile(outdir, 'diff_distfamily_rho_x_compare_beta.jpg');

    savefig(fig, fig_path);
    print(fig, jpg_path, '-djpeg', '-r300');

    %% ---------------- 7. Pack outputs ----------------
    out = struct();
    out.t_series = curves.median_matched.Weibull.t_series;
    out.curves = curves;
    out.tss = tss;
    out.fig_path = fig_path;
    out.jpg_path = jpg_path;
    out.params = P;
    out.matching = struct('median_mN', mN, 'median_mA', mA, ...
                          'alphaW', alphaW, 'alphaLL', alphaLL, 'aLo', aLo, ...
                          'beta_median', beta_median, 'beta_fixed', beta_fixed);

    fprintf('Saved figure to:\n  %s\n  %s\n', fig_path, jpg_path);
end

%% ============================================================
% Core theory runner aligned with theory代码.txt semantics
%% ============================================================
function theory_out = run_core_theory_family(P, Adj, cfg)
    dt = P.dt;
    Tmax = P.Tmax_calc;
    t_series = (0:dt:Tmax)';
    nSteps = numel(t_series) - 1;

    N = P.N;
    deg = full(sum(Adj,2));
    deg = max(deg, 1e-20);

    % Memory coefficients for exponential kernel
    if P.gamma_mem > 0
        decayH = exp(-P.gamma_mem * dt);
        addH   = 1 - decayH;
    else
        decayH = 0;
        addH   = 1;
    end

    % Infection-age truncation from recovery survival
    tauI_max = P.bR * (log(1 / P.epsTail))^(1 / P.aR);
    L_I = ceil(min(Tmax, tauI_max) / dt) + 1;

    % Recovery conditional probability in each dt interval
    p_rec = weibull_condprob(L_I, dt, P.aR, P.bR);   % 1 x L_I
    s_rec = 1 - p_rec;
    pRec = p_rec(:)';
    sRec = s_rec(:)';

    % Strategy-dependent infection-age kernels eta(tau_I | X)
    qA = waiting_time_bin_mass(L_I, dt, cfg.dist_name, cfg.shape, cfg.betaA);
    qN = waiting_time_bin_mass(L_I, dt, cfg.dist_name, cfg.shape, cfg.betaN);

    eta_rate_A = renewal_attempt_kernel_from_mass(qA, dt);
    eta_rate_N = renewal_attempt_kernel_from_mass(qN, dt);
    etaA = eta_rate_A(:);
    etaN = eta_rate_N(:);

    % Initialization
    rng(P.init_seed);
    PA = zeros(N,1);
    PA(randperm(N, max(1, round(P.init_p * N)))) = 1.0;

    IA_age = zeros(N, L_I);
    IN_age = zeros(N, L_I);
    init_inf_idx = randperm(N, max(1, round(P.init_rho * N)));
    IA_age(init_inf_idx,1) = PA(init_inf_idx);
    IN_age(init_inf_idx,1) = 1 - PA(init_inf_idx);

    H = zeros(N,1);
    next_update_time = P.behavior_dt * rand(N,1);

    rho_vec = zeros(nSteps+1,1);
    x_vec   = zeros(nSteps+1,1);
    rho_vec(1) = mean(sum(IA_age,2) + sum(IN_age,2));
    x_vec(1)   = mean(PA);

    t_now = 0;
    for step = 1:nSteps
        t_end = t_now + dt;

        % (1) Current infected masses
        IA_before = sum(IA_age, 2);
        IN_before = sum(IN_age, 2);
        I_before  = IA_before + IN_before;

        % (2) Source-side pressure from infection-age densities
        phi_A = cfg_kappaA(cfg) * (IA_age * etaA);
        phi_N = cfg_kappaN(cfg) * (IN_age * etaN);
        Phi   = phi_A + phi_N;

        % (3) Spatial aggregation
        Lambda = Adj * Phi;

        % (4) Memory evolution
        if P.gamma_mem > 0
            H = decayH * H + addH * Lambda;
        else
            H = Lambda;
        end
        H = max(H, 0);

        % (5) Infection (no Jensen correction)
        SA = max(0, PA - IA_before);
        SN = max(0, (1 - PA) - IN_before);

        p_inf_A = 1 - exp(-P.chiA * H * dt);
        p_inf_N = 1 - exp(-P.chiN * H * dt);
        p_inf_A = min(max(p_inf_A,0),1);
        p_inf_N = min(max(p_inf_N,0),1);

        newA = SA .* p_inf_A;
        newN = SN .* p_inf_N;

        % (6) Recovery + infection-age shift
        IA_surv = IA_age .* sRec;
        IN_surv = IN_age .* sRec;

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

        % (7) Game update
        due = (next_update_time <= t_end + 1e-12);
        if any(due)
            neigh_A = Adj * PA;
            neigh_N = deg - neigh_A;
            rho_global = mean(I_after);
            F_risk = 1 / max(1e-5, 1 - rho_global);

            pi_N = P.uNN .* neigh_N + P.uNA .* neigh_A;
            pi_A = (P.uNA - P.c) .* neigh_N + P.uAA .* neigh_A;
            z_N = pi_N;
            z_A = pi_A .* F_risk;

            max_z = max(z_N, z_A);
            prob_A = exp(z_A - max_z) ./ (exp(z_N - max_z) + exp(z_A - max_z) + 1e-20);

            idx_due = find(due);
            prob_due = prob_A(idx_due);
            PA(idx_due) = prob_due;

            % Repartition infected-age mass by current strategy state
            I_age_total = IA_age(idx_due,:) + IN_age(idx_due,:);
            IA_age(idx_due,:) = I_age_total .* prob_due;
            IN_age(idx_due,:) = I_age_total .* (1 - prob_due);

            next_update_time(idx_due) = next_update_time(idx_due) + P.behavior_dt;
        end

        % (8) Record
        rho_vec(step+1) = mean(sum(IA_age,2) + sum(IN_age,2));
        x_vec(step+1)   = mean(PA);
        t_now = t_end;
    end

    theory_out = struct();
    theory_out.t_series = t_series;
    theory_out.rho = rho_vec;
    theory_out.x   = x_vec;
    theory_out.cfg = cfg;
end

%% ============================================================
% Helpers
%% ============================================================
function q_mass = waiting_time_bin_mass(L, dt, dist_name, shape, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;

    F0 = waiting_time_cdf(tau0, dist_name, shape, beta);
    F1 = waiting_time_cdf(tau1, dist_name, shape, beta);

    q_mass = F1 - F0;
    q_mass = max(q_mass, 0);
    q_mass = min(q_mass, 1);
end

function F = waiting_time_cdf(t, dist_name, shape, beta)
    t = max(t, 0);
    switch dist_name
        case 'Weibull'
            alpha = shape;
            F = 1 - exp(-(t ./ beta) .^ alpha);
        case 'LogLogistic'
            alpha = shape;
            z = (t ./ beta) .^ alpha;
            F = z ./ (1 + z);
        case 'Lomax'
            a = shape;
            F = 1 - (1 + t ./ beta) .^ (-a);
        otherwise
            error('Unknown distribution family: %s', dist_name);
    end
end

function eta_rate = renewal_attempt_kernel_from_mass(q_mass, dt)
% Discrete renewal version:
%   eta_mass(n) = q_mass(n) + sum_{m=1}^{n-1} eta_mass(m) q_mass(n-m)
    L = numel(q_mass);
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

function pcond = weibull_condprob(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;
    dH = (tau1 ./ beta) .^ alpha - (tau0 ./ beta) .^ alpha;
    pcond = -expm1(-dH);
    pcond = min(max(pcond,0),1)';
end

function A = ER_network(N, k_avg)
    p_edge = k_avg / max(1, N-1);
    A = sprand(N, N, p_edge);
    A = spones(triu(A,1));
    A = A + A';
    A = A - diag(diag(A));
end

function tss = compute_joint_steady_time(t, rho, x)
% Earliest time after which BOTH rho and x remain within strict tolerance
% bands around terminal averages over a sufficiently long tail window.
    dt = t(2) - t(1);
    tail_window = max(round(5 / dt), round(0.15 * numel(t))); % at least last 5 time units
    idx0 = numel(t) - tail_window + 1;
    rho_ss = mean(rho(idx0:end));
    x_ss   = mean(x(idx0:end));

    tol_rho = max(1e-3, 5e-3 * max(rho_ss, 1e-6));
    tol_x   = max(1e-3, 5e-3 * max(x_ss,   1e-6));

    tss = NaN;
    for i = 1:numel(t)
        cond_rho = all(abs(rho(i:end) - rho_ss) <= tol_rho);
        cond_x   = all(abs(x(i:end)   - x_ss)   <= tol_x);
        if cond_rho && cond_x
            tss = t(i);
            break;
        end
    end
end

function draw_tss_lines(ax, tss, curves, field_name, yl)
    dist_names = {'Weibull','LogLogistic','Lomax'};
    xl = xlim(ax);

    for d = 1:numel(dist_names)
        dist = dist_names{d};

        % median-matched: light gray solid
        tm = tss.median_matched.(dist);
        if ~isnan(tm) && tm >= xl(1) && tm <= xl(2)
            ym = interp1(curves.median_matched.(dist).t_series, curves.median_matched.(dist).(field_name), tm, 'linear');
            line(ax, [tm tm], [yl(1) ym], 'Color', [0.45 0.45 0.45], 'LineStyle', '-', ...
                'LineWidth', 1.8, 'HandleVisibility', 'off');
        end

        % fixed-beta: darker gray dashed
        tf = tss.fixed_beta.(dist);
        if ~isnan(tf) && tf >= xl(1) && tf <= xl(2)
            yf = interp1(curves.fixed_beta.(dist).t_series, curves.fixed_beta.(dist).(field_name), tf, 'linear');
            line(ax, [tf tf], [yl(1) yf], 'Color', [0.35 0.35 0.35], 'LineStyle', '--', ...
                'LineWidth', 1.6, 'HandleVisibility', 'off');
        end
    end
end

function s = pretty_dist_name(dist)
    switch dist
        case 'Weibull'
            s = 'Weibull';
        case 'LogLogistic'
            s = 'Log-logistic';
        case 'Lomax'
            s = 'Shifted Pareto (Lomax)';
        otherwise
            s = dist;
    end
end

function val = cfg_kappaA(~)
    val = 0.5;
end

function val = cfg_kappaN(~)
    val = 1.0;
end
