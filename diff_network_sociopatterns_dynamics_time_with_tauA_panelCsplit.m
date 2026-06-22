function out = diff_network_sociopatterns_dynamics_time_with_tauA_panelCsplit(data_file)

    clc; clear; close all;
    
    if nargin < 1 || isempty(data_file)
        data_file = 'thiers_2012.csv';
    end

    %% ---------------- 1) 参数 ----------------
    params = struct();

    % 时间
    params.dt   = 0.01;
    params.Tmax = 20;
    params.nSteps = ceil(params.Tmax / params.dt);
    params.t_series = (0:params.nSteps)' * params.dt;
    params.Tn = numel(params.t_series);
    params.steady_frac = 0.20;

    % 恢复核：Weibull
    params.aR = 2.0;
    params.bR = 0.5;

    % 传播尝试核：Weibull（PM1 + Weibull）
    params.aI = 0.8;
    params.betaA  = 1.2;
    params.betaN  = 1.0;

    % 三重调制
    params.kappaA = 0.5;
    params.kappaN = 1.0;
    params.chiA   = 0.4;
    params.chiN   = 1.0;
    params.gamma_mem = 1.0;

    % 行为更新
    params.behavior_dt = 1.0;

    % 初值
    params.init_rho = 0.05;
    params.init_p   = 0.05;

    % PM1 payoff
    params.payoff = struct('name','PM1', 'uAA',0.01, 'uNA',0.05, 'uNN',0.10, 'c',0.02);

    % 理论离散与截断
    params.epsTail = 1e-8;

    % 实验设计
    params.num_syn_networks   = 10;
    params.num_theory_reps_syn  = 10;
    params.num_theory_reps_real = 100;
    params.base_seed = 20260418;

    % WS 参数
    params.p_rewire_WS = 0.10;

    % tau_ss 判定参数
    params.ss_abs_tol_rho = 1e-3;
    params.ss_abs_tol_x   = 1e-3;
    params.ss_rel_tol     = 0.005;
    params.ss_window_time = 1.0;

    % tau_A 判定参数
    params.tauA_frac = 0.50;
    params.tauA_tol  = 1e-10;

    % 保存
    params.save_dir  = '四月实验结果图';
    params.save_name = 'diff_network_sociopatterns_dynamics_time_tauA_panelCsplit';

    % 局部放大图参数
    params.zoom_window_time = 5.0;
    params.inset_pad_frac_rho = 0.12;
    params.inset_pad_frac_x   = 0.12;
    params.inset_min_span_rho = 0.012;
    params.inset_min_span_x   = 0.010;

    fprintf('============================================================\n');
    fprintf('GSRT-nSIS on matched ER / BA / WS / SocioPatterns\n');
    fprintf('Theory-only dynamics | bottom solver aligned to theory代码.txt\n');
    fprintf('Data file: %s\n', data_file);
    fprintf('============================================================\n');

    %% ---------------- 2) 读取 SocioPatterns，聚合静态二值网络 ----------------
    socio = read_sociopatterns_static_binary(data_file);
    As = socio.A;
    Ns = socio.N;
    Es = socio.E;
    k_avg_s = socio.k_avg;

    fprintf('SocioPatterns static aggregation finished.\n');
    fprintf('  N_s    = %d\n', Ns);
    fprintf('  E_s    = %d\n', Es);
    fprintf('  <k>_s  = %.6f\n', k_avg_s);
    fprintf('  raw IDs= %d\n', numel(socio.unique_ids));
    fprintf('  time   = [%g, %g]\n', socio.t_min, socio.t_max);

    %% ---------------- 3) 构造匹配 ER / BA / WS ----------------
    syn_nets = struct('name',{},'A',{},'N',{},'E',{},'k_avg',{});
    ctr = 0;

    for net_type_idx = 1:3
        for r = 1:params.num_syn_networks
            seed_here = params.base_seed + 1000 * net_type_idx + r;
            switch net_type_idx
                case 1
                    A = ER_exact_edges(Ns, Es, seed_here);
                    net_name = 'ER';

                case 2
                    rng(seed_here);
                    m_BA = choose_m_BA(Ns, Es);
                    A = BA_network(Ns, m_BA, seed_here + 7);
                    A = adjust_edge_count(A, Es, seed_here + 11);
                    net_name = 'BA';

                case 3
                    rng(seed_here);
                    K_WS = choose_K_WS(Ns, Es);
                    A = WS_network(Ns, K_WS, params.p_rewire_WS, seed_here + 19);
                    A = adjust_edge_count(A, Es, seed_here + 22);
                    net_name = 'WS';
            end

            ctr = ctr + 1;
            syn_nets(ctr).name  = net_name;
            syn_nets(ctr).A     = A;
            syn_nets(ctr).N     = size(A,1);
            syn_nets(ctr).E     = nnz(triu(A,1));
            syn_nets(ctr).k_avg = full(mean(sum(A,2)));
        end
    end

    fprintf('Synthetic matched networks prepared.\n');
    fprintf('  ER reps = %d, BA reps = %d, WS reps = %d\n', ...
        params.num_syn_networks, params.num_syn_networks, params.num_syn_networks);
    fprintf('  BA uses m = %d (before edge-count adjustment)\n', choose_m_BA(Ns, Es));
    fprintf('  WS uses K = %d, p_rewire = %.2f (before edge-count adjustment)\n', ...
        choose_K_WS(Ns, Es), params.p_rewire_WS);

    %% ---------------- 4) 预计算理论核 ----------------
    theory_cache = prepare_theory_cache(params);

    %% ---------------- 5) 任务列表 ----------------
    family_names = {'ER','BA','WS','SocioPatterns'};
    family_code = containers.Map(family_names, 1:4);

    task_family = [];
    task_seed   = [];
    task_A      = {};

    % synthetic families
    for fam_idx = 1:3
        fam_name = family_names{fam_idx};
        net_idx = find(strcmp({syn_nets.name}, fam_name));
        for ii = 1:numel(net_idx)
            A = syn_nets(net_idx(ii)).A;
            for rep = 1:params.num_theory_reps_syn
                seed_run = params.base_seed + fam_idx * 100000 + ii * 1000 + rep;
                task_family(end+1,1) = family_code(fam_name); %#ok<AGROW>
                task_seed(end+1,1)   = seed_run; %#ok<AGROW>
                task_A{end+1,1}      = A; %#ok<AGROW>
            end
        end
    end

    % SocioPatterns family
    for rep = 1:params.num_theory_reps_real
        seed_run = params.base_seed + 999000 + rep;
        task_family(end+1,1) = family_code('SocioPatterns'); %#ok<AGROW>
        task_seed(end+1,1)   = seed_run; %#ok<AGROW>
        task_A{end+1,1}      = As; %#ok<AGROW>
    end

    nTasks = numel(task_seed);
    rho_paths = zeros(nTasks, params.Tn);
    x_paths   = zeros(nTasks, params.Tn);
    rho_task_ss = zeros(nTasks,1);
    x_task_ss   = zeros(nTasks,1);
    tau_task_ss = nan(nTasks,1);
    tau_task_A  = nan(nTasks,1);

    %% ---------------- 6) 并行运行 theory trajectories ----------------
    try
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            parpool;
        end
    catch ME
        warning('Parallel pool was not started automatically: %s', ME.message);
    end

    fprintf('Running %d theory tasks with parfor ...\n', nTasks);
    parfor tt = 1:nTasks
        A = task_A{tt};
        seed_run = task_seed(tt);
        [rho_run, x_run, rho_ss, x_ss, tau_ss, tau_A] = ...
            run_one_gsrt_nsis_theory_on_fixed_graph(A, params, theory_cache, seed_run);

        rho_paths(tt,:) = rho_run(:).';
        x_paths(tt,:)   = x_run(:).';
        rho_task_ss(tt) = rho_ss;
        x_task_ss(tt)   = x_ss;
        tau_task_ss(tt) = tau_ss;
        tau_task_A(tt)  = tau_A;
    end

    %% ---------------- 7) 家族平均轨迹与稳态时间 ----------------
    rho_mean = zeros(4, params.Tn);
    x_mean   = zeros(4, params.Tn);
    rho_ss_mean = zeros(1,4);
    x_ss_mean   = zeros(1,4);
    tau_ss_mean = nan(1,4);
    tau_A_mean  = nan(1,4);

    rho_samples = cell(1,4);
    x_samples   = cell(1,4);
    tau_ss_samples = cell(1,4);
    tau_A_samples  = cell(1,4);

    for fam_idx = 1:4
        mask = (task_family == fam_idx);

        rho_samples{fam_idx}    = rho_task_ss(mask);
        x_samples{fam_idx}      = x_task_ss(mask);
        tau_ss_samples{fam_idx} = tau_task_ss(mask);
        tau_A_samples{fam_idx}  = tau_task_A(mask);

        rho_mean(fam_idx,:) = mean(rho_paths(mask,:), 1);
        x_mean(fam_idx,:)   = mean(x_paths(mask,:), 1);

        rho_ss_mean(fam_idx) = mean(rho_task_ss(mask));
        x_ss_mean(fam_idx)   = mean(x_task_ss(mask));

        tau_valid_ss = tau_task_ss(mask);
        tau_valid_ss = tau_valid_ss(~isnan(tau_valid_ss));
        if ~isempty(tau_valid_ss)
            tau_ss_mean(fam_idx) = mean(tau_valid_ss);
        else
            tau_ss_mean(fam_idx) = estimate_tau_ss_joint( ...
                params.t_series, rho_mean(fam_idx,:).', x_mean(fam_idx,:).', params);
        end

        tau_valid_A = tau_task_A(mask);
        tau_valid_A = tau_valid_A(~isnan(tau_valid_A));
        if ~isempty(tau_valid_A)
            tau_A_mean(fam_idx) = mean(tau_valid_A);
        else
            tau_A_mean(fam_idx) = estimate_tauA_from_series( ...
                params.t_series, x_mean(fam_idx,:).', x_ss_mean(fam_idx), params);
        end

        fprintf('  %-13s: rho* = %.6f | x* = %.6f | tau_ss = %.4f | tau_A = %.4f\n', ...
            family_names{fam_idx}, rho_ss_mean(fam_idx), x_ss_mean(fam_idx), ...
            tau_ss_mean(fam_idx), tau_A_mean(fam_idx));
    end

    %% ---------------- 8) 绘图 ----------------
    t = params.t_series;

    % Paper-style palette
    cmap = [ 53 98 232; ...
            118 199 107; ...
            240 155  70; ...
            227 113 195] / 255;
    cmap_light = 0.55 + 0.45 * cmap;

    f = figure('Color','w', 'Position',[70 90 1450 460]);
    tl = tiledlayout(f, 1, 3, 'TileSpacing','compact', 'Padding','compact');

    % ---- rho(t)
    ax1 = nexttile(tl, 1); hold(ax1,'on');
    hLeg = gobjects(1,4);
    for fam_idx = 1:4
        hLeg(fam_idx) = plot(ax1, t, rho_mean(fam_idx,:), '-', ...
            'Color', cmap(fam_idx,:), 'LineWidth', 2.2, ...
            'DisplayName', family_names{fam_idx});
    end
    xlabel(ax1, 't', 'FontSize', 12);
    ylabel(ax1, '\rho(t)', 'Interpreter','tex', 'FontSize', 12);
    title(ax1, '(a) Infection density', 'FontSize', 12);
    box(ax1,'on'); grid(ax1,'on');
    y1max = max(rho_mean(:));
    ylim(ax1, [0, min(1, 1.08 * y1max + 1e-3)]);
    legend(ax1, hLeg, 'Location','southeast', 'Box','off', 'FontSize', 10);

    % ---- x(t)
    ax2 = nexttile(tl, 2); hold(ax2,'on');
    for fam_idx = 1:4
        plot(ax2, t, x_mean(fam_idx,:), '-', ...
            'Color', cmap(fam_idx,:), 'LineWidth', 2.2, ...
            'DisplayName', family_names{fam_idx});
    end
    xlabel(ax2, 't', 'FontSize', 12);
    ylabel(ax2, 'x(t)', 'Interpreter','tex', 'FontSize', 12);
    title(ax2, '(b) Protection level', 'FontSize', 12);
    box(ax2,'on'); grid(ax2,'on');
    y2max = max(x_mean(:));
    ylim(ax2, [0, min(1, 1.08 * y2max + 1e-3)]);

    drawnow;
    pos1 = get(ax1, 'Position');
    pos2 = get(ax2, 'Position');

    % ---- 最后 5 个时间单位的局部放大子图
    t_zoom_start = max(t(1), t(end) - params.zoom_window_time);
    idx_zoom = (t >= t_zoom_start);
    t_zoom = t(idx_zoom);

    % ================= rho(t) inset =================
    rho_zoom_all = rho_mean(:, idx_zoom);
    [rho_inset_y0, rho_inset_y1] = compute_inset_ylim( ...
        rho_zoom_all, params.inset_pad_frac_rho, params.inset_min_span_rho);

    inset1 = axes('Parent', f, 'Position', [pos1(1)+0.60*pos1(3), pos1(2)+0.42*pos1(4), 0.25*pos1(3), 0.22*pos1(4)]);
    hold(inset1, 'on');
    for fam_idx = 1:4
        plot(inset1, t_zoom, rho_mean(fam_idx, idx_zoom), '-', ...
            'Color', cmap(fam_idx,:), 'LineWidth', 1.7, 'HandleVisibility','off');
    end
    xlim(inset1, [t_zoom_start, t(end)]);
    ylim(inset1, [rho_inset_y0, rho_inset_y1]);
    grid(inset1, 'on'); box(inset1, 'on');
    set(inset1, 'FontSize', 8.5, 'LineWidth', 0.9);

    % ================= x(t) inset =================
    x_zoom_all = x_mean(:, idx_zoom);
    [x_inset_y0, x_inset_y1] = compute_inset_ylim( ...
        x_zoom_all, params.inset_pad_frac_x, params.inset_min_span_x);

    inset2 = axes('Parent', f, 'Position', [pos2(1)+0.60*pos2(3), pos2(2)+0.42*pos2(4), 0.25*pos2(3), 0.22*pos2(4)]);
    hold(inset2, 'on');
    for fam_idx = 1:4
        plot(inset2, t_zoom, x_mean(fam_idx, idx_zoom), '-', ...
            'Color', cmap(fam_idx,:), 'LineWidth', 1.7, 'HandleVisibility','off');
    end
    xlim(inset2, [t_zoom_start, t(end)]);
    ylim(inset2, [x_inset_y0, x_inset_y1]);
    grid(inset2, 'on'); box(inset2, 'on');
    set(inset2, 'FontSize', 8.5, 'LineWidth', 0.9);

    % ---- panel (c): vertically stacked tau_ss / tau_A with shared x-axis
    ax3base = nexttile(tl, 3);
    set(ax3base, 'Visible', 'off');
    drawnow;
    pos3 = get(ax3base, 'Position');
    delete(ax3base);

    xcat = 1:4;

    top_h = 0.40 * pos3(4);
    bot_h = 0.40 * pos3(4);
    gap_h = 0.06 * pos3(4);

    % ===== upper: tau_ss =====
    ax3_top = axes('Parent', f, ...
        'Position', [pos3(1), pos3(2) + bot_h + gap_h, pos3(3), top_h]);
    hold(ax3_top, 'on');

    b1 = bar(ax3_top, xcat, tau_ss_mean, 0.62, 'FaceColor', 'flat', 'LineWidth', 0.9);
    b1.CData = cmap;
    b1.EdgeColor = [0.25 0.25 0.25];

    for i = 1:numel(xcat)
        text(ax3_top, xcat(i), tau_ss_mean(i) + 0.02*max(tau_ss_mean), sprintf('%.2f', tau_ss_mean(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 9, 'Color', [0.15 0.15 0.15]);
    end

    xlim(ax3_top, [0.4, 4.6]);
    ylabel(ax3_top, '\tau_{ss}', 'Interpreter', 'tex', 'FontSize', 11);
    title(ax3_top, '(c) Arrival times', 'FontSize', 12);
    box(ax3_top, 'on'); grid(ax3_top, 'on');
    set(ax3_top, 'XTick', xcat, 'XTickLabel', []);
    ylim(ax3_top, [max(0, min(tau_ss_mean) - 0.12), max(tau_ss_mean) + 0.20 * max(1, max(tau_ss_mean))]);

    % ===== lower: tau_A =====
    ax3_bot = axes('Parent', f, ...
        'Position', [pos3(1), pos3(2), pos3(3), bot_h]);
    hold(ax3_bot, 'on');

    b2 = bar(ax3_bot, xcat, tau_A_mean, 0.62, 'FaceColor', 'flat', 'LineWidth', 0.9);
    b2.CData = cmap_light;
    b2.EdgeColor = [0.25 0.25 0.25];

    for i = 1:numel(xcat)
        text(ax3_bot, xcat(i), tau_A_mean(i) + 0.02*max(tau_A_mean), sprintf('%.2f', tau_A_mean(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 9, 'Color', [0.15 0.15 0.15]);
    end

    xlim(ax3_bot, [0.4, 4.6]);
    xticks(ax3_bot, xcat);
    xticklabels(ax3_bot, family_names);
    xtickangle(ax3_bot, 18);
    xlabel(ax3_bot, 'network family', 'FontSize', 11);
    ylabel(ax3_bot, '\tau_A', 'Interpreter', 'tex', 'FontSize', 11);
    box(ax3_bot, 'on'); grid(ax3_bot, 'on');
    ylim(ax3_bot, [max(0, min(tau_A_mean) - 0.08), max(tau_A_mean) + 0.24 * max(1, max(tau_A_mean))]);

    linkaxes([ax3_top, ax3_bot], 'x');

    sgtitle(sprintf(['GSRT-nSIS on matched ER / BA / WS / SocioPatterns | PM1 + Weibull\n' ...
                     'Theory-only dynamics, N_s = %d, E_s = %d, <k>_s = %.4f'], ...
                     Ns, Es, k_avg_s), 'FontSize', 12);

    %% ---------------- 9) 保存 ----------------
    if ~exist(params.save_dir, 'dir')
        mkdir(params.save_dir);
    end

    fig_path = fullfile(params.save_dir, [params.save_name, '.fig']);
    jpg_path = fullfile(params.save_dir, [params.save_name, '.jpg']);
    savefig(f, fig_path);
    print(f, jpg_path, '-djpeg', '-r300');

    fprintf('Saved figure to:\n  %s\n  %s\n', fig_path, jpg_path);

    %% ---------------- 10) 输出 ----------------
    out = struct();
    out.params = params;
    out.sociopatterns = socio;
    out.As = As;
    out.Ns = Ns;
    out.Es = Es;
    out.k_avg_s = k_avg_s;
    out.synthetic_networks = syn_nets;
    out.family_names = family_names;
    out.task_family = task_family;
    out.task_seed   = task_seed;
    out.rho_paths   = rho_paths;
    out.x_paths     = x_paths;
    out.rho_mean    = rho_mean;
    out.x_mean      = x_mean;
    out.rho_ss_mean = rho_ss_mean;
    out.x_ss_mean   = x_ss_mean;
    out.tau_ss_mean = tau_ss_mean;
    out.tau_A_mean  = tau_A_mean;
    out.tau_ss_samples = tau_ss_samples;
    out.tau_A_samples  = tau_A_samples;
    out.fig = f;
    out.fig_path = fig_path;
    out.jpg_path = jpg_path;
end

%% ========================================================================
function [y0, y1] = compute_inset_ylim(Y, pad_frac, min_span)
    y_min = min(Y(:));
    y_max = max(Y(:));
    y_span = y_max - y_min;

    if y_span < min_span
        y_center = 0.5 * (y_min + y_max);
        y0 = y_center - 0.5 * min_span;
        y1 = y_center + 0.5 * min_span;
    else
        pad = pad_frac * y_span;
        y0 = y_min - pad;
        y1 = y_max + pad;
    end
end

%% ========================================================================
function socio = read_sociopatterns_static_binary(data_file)
    if ~isfile(data_file)
        error('Cannot find data file: %s', data_file);
    end

    fid = fopen(data_file, 'r');
    if fid < 0
        error('Cannot open file: %s', data_file);
    end

    C = textscan(fid, '%f%f%f%*[^\n]', ...
        'Delimiter', {' ', '\t', ','}, ...
        'MultipleDelimsAsOne', true, ...
        'HeaderLines', 0, ...
        'ReturnOnError', false);
    fclose(fid);

    if isempty(C) || numel(C) < 3
        error('Failed to parse the first three columns from %s', data_file);
    end

    t   = C{1};
    id1 = C{2};
    id2 = C{3};

    valid = ~isnan(id1) & ~isnan(id2) & (id1 ~= id2);
    t   = t(valid);
    id1 = id1(valid);
    id2 = id2(valid);

    if isempty(id1)
        error('No valid contact pairs were read from %s', data_file);
    end

    pairs = sort([id1, id2], 2);
    pairs = unique(pairs, 'rows');

    ids = unique(pairs(:));
    N = numel(ids);

    [~, u] = ismember(pairs(:,1), ids);
    [~, v] = ismember(pairs(:,2), ids);

    A = sparse(u, v, 1, N, N);
    A = A + A.';
    A(A > 0) = 1;
    A(1:N+1:end) = 0;

    E = nnz(triu(A,1));
    k = full(sum(A,2));
    k_avg = mean(k);

    socio = struct();
    socio.A = A;
    socio.N = N;
    socio.E = E;
    socio.k = k;
    socio.k_avg = k_avg;
    socio.unique_ids = ids;
    socio.t_min = min(t);
    socio.t_max = max(t);
end

%% ========================================================================
function cache = prepare_theory_cache(params)
    dt = params.dt;
    Tmax = params.Tmax;
    aR = params.aR;
    bR = params.bR;
    aI = params.aI;
    betaA = params.betaA;
    betaN = params.betaN;
    gamma_mem = params.gamma_mem;
    epsTail = params.epsTail;

    tauI_max = bR * (log(1 / epsTail))^(1 / aR);
    L_I = ceil(min(Tmax, tauI_max) / dt) + 1;

    p_rec = weibull_condprob(L_I, dt, aR, bR);
    s_rec = 1 - p_rec;

    etaA = renewal_attempt_kernel(L_I, dt, aI, betaA);
    etaN = renewal_attempt_kernel(L_I, dt, aI, betaN);

    if gamma_mem > 0
        decayH = exp(-gamma_mem * dt);
        addH   = 1 - decayH;
    else
        decayH = 0;
        addH   = 1;
    end

    cache = struct();
    cache.L_I = L_I;
    cache.sRec = s_rec(:)';
    cache.etaA = etaA(:);
    cache.etaN = etaN(:);
    cache.decayH = decayH;
    cache.addH = addH;
end

%% ========================================================================
function pcond = weibull_condprob(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;
    dH = (tau1./beta).^alpha - (tau0./beta).^alpha;
    pcond = -expm1(-dH);
    pcond = min(max(pcond, 0), 1);
end

%% ========================================================================
function eta_rate = renewal_attempt_kernel(L, dt, alpha, beta)
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
        for m = 1:(n-1)
            conv_sum = conv_sum + eta_mass(m) * q_mass(n-m);
        end
        eta_mass(n) = q_mass(n) + conv_sum;
    end

    eta_rate = eta_mass / dt;
end

%% ========================================================================
function A = ER_exact_edges(N, Etarget, seed)
    rng(seed);
    if Etarget > N*(N-1)/2
        error('Etarget exceeds the number of possible undirected edges.');
    end

    mask = triu(true(N), 1);
    [ii_all, jj_all] = find(mask);
    sel = randperm(numel(ii_all), Etarget);

    ii = ii_all(sel);
    jj = jj_all(sel);

    A = sparse(ii, jj, 1, N, N);
    A = A + A.';
    A(A > 0) = 1;
end

%% ========================================================================
function m_best = choose_m_BA(N, Etarget)
    best_gap = inf;
    m_best = 1;
    for m = 1:max(1, floor((N-1)/2))
        m0 = m + 1;
        E = m0*(m0-1)/2 + m*(N - m0);
        gap = abs(E - Etarget);
        if gap < best_gap
            best_gap = gap;
            m_best = m;
        end
    end
end

%% ========================================================================
function K_best = choose_K_WS(N, Etarget)
    k_target = 2 * Etarget / N;
    cand = 2:2:(N-1 - mod(N-1,2));
    [~, idx] = min(abs(cand - k_target));
    K_best = cand(idx);
end

%% ========================================================================
function A = BA_network(N, m, seed)
    rng(seed);

    if m < 1
        error('BA parameter m must be >= 1.');
    end
    if N <= m + 1
        A = ones(N) - eye(N);
        A = sparse(A);
        return;
    end

    m0 = m + 1;
    A = sparse(N, N);

    for i = 1:m0
        for j = (i+1):m0
            A(i,j) = 1;
            A(j,i) = 1;
        end
    end

    deg = full(sum(A,2));

    for newNode = (m0+1):N
        targets = zeros(m,1);
        chosen = false(newNode-1,1);

        for kk = 1:m
            prob = deg(1:newNode-1);
            prob(chosen) = 0;
            s = sum(prob);
            if s <= 0
                candidates = find(~chosen);
                pick = candidates(randi(numel(candidates)));
            else
                prob = prob / s;
                pick = weighted_pick(prob);
                while chosen(pick)
                    pick = weighted_pick(prob);
                end
            end
            targets(kk) = pick;
            chosen(pick) = true;
        end

        A(newNode, targets) = 1;
        A(targets, newNode) = 1;

        deg(newNode) = m;
        deg(targets) = deg(targets) + 1;
    end

    A(A > 0) = 1;
    A = sparse(A);
end

%% ========================================================================
function A = WS_network(N, K, p_rewire, seed)
    rng(seed);

    if mod(K,2) ~= 0
        error('WS parameter K must be even.');
    end
    if K >= N
        error('WS parameter K must satisfy K < N.');
    end

    A = sparse(N,N);
    halfK = K / 2;

    for i = 1:N
        for d = 1:halfK
            j = i + d;
            if j > N
                j = j - N;
            end
            A(i,j) = 1;
            A(j,i) = 1;
        end
    end

    for i = 1:N
        for d = 1:halfK
            j = i + d;
            if j > N
                j = j - N;
            end

            if rand < p_rewire
                A(i,j) = 0;
                A(j,i) = 0;

                forbidden = find(A(i,:) > 0);
                forbidden = unique([forbidden, i]);

                candidates = setdiff(1:N, forbidden);
                if isempty(candidates)
                    A(i,j) = 1;
                    A(j,i) = 1;
                else
                    newj = candidates(randi(numel(candidates)));
                    A(i,newj) = 1;
                    A(newj,i) = 1;
                end
            end
        end
    end

    A(A > 0) = 1;
    A(1:N+1:end) = 0;
    A = sparse(A);
end

%% ========================================================================
function A = adjust_edge_count(A, Etarget, seed)
    rng(seed);

    N = size(A,1);
    A = spones(A);
    A = spones(triu(A,1));
    A = A + A.';
    A(A > 0) = 1;
    A(1:N+1:end) = 0;

    currE = nnz(triu(A,1));

    while currE > Etarget
        [u_all, v_all] = find(triu(A,1));
        deg = full(sum(A,2));
        removable = find(deg(u_all) > 1 & deg(v_all) > 1);
        if isempty(removable)
            removable = 1:numel(u_all);
        end
        pick = removable(randi(numel(removable)));
        u = u_all(pick);
        v = v_all(pick);
        A(u,v) = 0;
        A(v,u) = 0;
        currE = currE - 1;
    end

    while currE < Etarget
        u = randi(N);
        v = randi(N);
        if u ~= v && A(u,v) == 0
            A(u,v) = 1;
            A(v,u) = 1;
            currE = currE + 1;
        end
    end

    A(A > 0) = 1;
    A(1:N+1:end) = 0;
    A = sparse(A);
end

%% ========================================================================
function idx = weighted_pick(prob)
    c = cumsum(prob(:));
    if c(end) <= 0
        idx = randi(numel(prob));
        return;
    end
    r = rand * c(end);
    idx = find(c >= r, 1, 'first');
    if isempty(idx)
        idx = numel(prob);
    end
end

%% ========================================================================
function [rho_run, x_run, rho_ss, x_ss, tau_ss, tau_A] = ...
    run_one_gsrt_nsis_theory_on_fixed_graph(Adj, params, cache, seed)

    rng(seed);

    N = size(Adj,1);
    deg = full(sum(Adj,2));
    deg = max(deg, 1e-20);

    dt = params.dt;
    nSteps = params.nSteps;
    steady_frac = params.steady_frac;
    behavior_dt = params.behavior_dt;
    gamma_mem = params.gamma_mem;
    init_rho = params.init_rho;
    init_p = params.init_p;
    payoff = params.payoff;

    kappaA = params.kappaA;
    kappaN = params.kappaN;
    chiA   = params.chiA;
    chiN   = params.chiN;

    L_I  = cache.L_I;
    sRec = cache.sRec;
    etaA = cache.etaA;
    etaN = cache.etaN;
    decayH = cache.decayH;
    addH   = cache.addH;

    PA = zeros(N,1);
    num_init_A = max(1, round(init_p * N));
    idxA0 = randperm(N, num_init_A);
    PA(idxA0) = 1.0;

    IA_age = zeros(N, L_I);
    IN_age = zeros(N, L_I);
    num_init_I = max(1, round(init_rho * N));
    idxI0 = randperm(N, num_init_I);
    IA_age(idxI0,1) = PA(idxI0);
    IN_age(idxI0,1) = 1 - PA(idxI0);

    H = zeros(N,1);
    next_update_time = behavior_dt * rand(N,1);

    rho_run = zeros(nSteps+1, 1);
    x_run   = zeros(nSteps+1, 1);

    rho_run(1) = mean(sum(IA_age,2) + sum(IN_age,2));
    x_run(1)   = mean(PA);

    t_now = 0;

    for step = 1:nSteps
        t_end = t_now + dt;

        IA_before = sum(IA_age, 2);
        IN_before = sum(IN_age, 2);

        phi_A = kappaA * (IA_age * etaA);
        phi_N = kappaN * (IN_age * etaN);
        Phi   = phi_A + phi_N;

        Lambda = Adj * Phi;

        if gamma_mem > 0
            H = decayH * H + addH * Lambda;
        else
            H = Lambda;
        end
        H = max(H, 0);

        SA = max(0, PA - IA_before);
        SN = max(0, (1 - PA) - IN_before);

        p_inf_A = 1 - exp(-chiA * H * dt);
        p_inf_N = 1 - exp(-chiN * H * dt);
        p_inf_A = min(max(p_inf_A, 0), 1);
        p_inf_N = min(max(p_inf_N, 0), 1);

        newA = SA .* p_inf_A;
        newN = SN .* p_inf_N;

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

        due = (next_update_time <= t_end + 1e-12);
        if any(due)
            neigh_A = Adj * PA;
            neigh_N = deg - neigh_A;

            rho_global = mean(I_after);
            F_risk = 1 / max(1e-5, 1 - rho_global);

            pi_N = payoff.uNN * neigh_N + payoff.uNA * neigh_A;
            pi_A = (payoff.uNA - payoff.c) * neigh_N + payoff.uAA * neigh_A;
            z_N = pi_N;
            z_A = pi_A * F_risk;

            max_z = max(z_N, z_A);
            prob_A = exp(z_A - max_z) ./ (exp(z_N - max_z) + exp(z_A - max_z) + 1e-20);

            idx_due = find(due);
            prob_due = prob_A(idx_due);
            PA(idx_due) = prob_due;

            I_age_total = IA_age(idx_due,:) + IN_age(idx_due,:);
            IA_age(idx_due,:) = I_age_total .* prob_due;
            IN_age(idx_due,:) = I_age_total .* (1 - prob_due);

            next_update_time(idx_due) = next_update_time(idx_due) + behavior_dt;
        end

        rho_run(step+1) = mean(sum(IA_age,2) + sum(IN_age,2));
        x_run(step+1)   = mean(PA);
        t_now = t_end;
    end

    ss_start_idx = max(1, floor((1 - steady_frac) * (nSteps+1)) + 1);
    ss_idx = ss_start_idx:(nSteps+1);
    rho_ss = mean(rho_run(ss_idx));
    x_ss   = mean(x_run(ss_idx));

    tau_ss = estimate_tau_ss_joint(params.t_series, rho_run, x_run, params);
    tau_A  = estimate_tauA_from_series(params.t_series, x_run, x_ss, params);
end

%% ========================================================================
function tau_ss = estimate_tau_ss_joint(t, rho, x, params)
% 联合 rho(t), x(t) 判定 steady-state arrival time

    N = numel(t);
    ss_start_idx = max(1, floor((1 - params.steady_frac) * N) + 1);
    ss_idx = ss_start_idx:N;

    rho_ss = mean(rho(ss_idx));
    x_ss   = mean(x(ss_idx));

    rho_range = max(rho) - min(rho);
    x_range   = max(x) - min(x);

    tol_rho = max(params.ss_abs_tol_rho, params.ss_rel_tol * max(rho_range, 1e-8));
    tol_x   = max(params.ss_abs_tol_x,   params.ss_rel_tol * max(x_range,   1e-8));

    dt = max(t(2) - t(1), eps);
    win = max(5, round(params.ss_window_time / dt));

    tau_ss = NaN;
    for k = 1:(N - win + 1)
        idx_win = k:(k + win - 1);

        cond_window = ...
            all(abs(rho(idx_win) - rho_ss) <= tol_rho) && ...
            all(abs(x(idx_win)   - x_ss)   <= tol_x);

        cond_remain = ...
            all(abs(rho(k:end) - rho_ss) <= 1.5 * tol_rho) && ...
            all(abs(x(k:end)   - x_ss)   <= 1.5 * tol_x);

        cond_flat = ...
            (max(rho(idx_win)) - min(rho(idx_win)) <= tol_rho) && ...
            (max(x(idx_win))   - min(x(idx_win))   <= tol_x);

        if cond_window && cond_remain && cond_flat
            tau_ss = t(k);
            return;
        end
    end

    if isnan(tau_ss)
        tau_ss = t(end);
    end
end

function tau_A = estimate_tauA_from_series(t, x, x_ss, params)
% tau_A = inf { t : x(t) >= x(0) + frac * [x* - x(0)] }

    x0 = x(1);
    dx = x_ss - x0;

    if dx <= params.tauA_tol
        tau_A = NaN;
        return;
    end

    target = x0 + params.tauA_frac * dx;
    idx = find(x >= target, 1, 'first');

    if isempty(idx)
        tau_A = NaN;
    else
        tau_A = t(idx);
    end
end
