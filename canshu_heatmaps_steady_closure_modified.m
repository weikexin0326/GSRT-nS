function results = canshu_heatmaps_steady_closure_modified()
% ========================================================================


    clc; close all;

    %% ===================== 1. 全局配置 =====================
    cfg = struct();

    % ---------------- 数值设置 ----------------
    cfg.dt = 0.01;
    cfg.Tmax_kernel = 20;     % 仅用于计算 LambdaTilde 的时间截断

    % ---------------- 网络平均度 ----------------
    cfg.k_avg = 10;

    % ---------------- 恢复过程：Weibull ----------------
    cfg.aR = 2.0;
    cfg.bR = 0.5;

    % ---------------- 传播尝试过程：Weibull ----------------
    cfg.aI = 0.8;             % 固定 alpha_I

    % ---------------- 三通道基线参数 ----------------
    cfg.beta0  = 1.0;         % beta_N
    cfg.beta1  = 1.2;         % beta_A
    cfg.kappa0 = 1.0;         % kappa_N
    cfg.kappa1 = 0.5;         % kappa_A
    cfg.zeta0  = 1.0;         % zeta_N
    cfg.zeta1  = 0.4;         % zeta_A

    % ---------------- 博弈参数（当前沿用 PM1） ----------------
    cfg.c   = 0.02;
    cfg.uNN = 0.10;
    cfg.uNA = 0.05;
    cfg.uAA = 0.01;

    % ---------------- 年龄网格截断精度 ----------------
    cfg.epsTail = 1e-10;

    % ---------------- 热图网格分辨率 ----------------
    cfg.nGrid_kappa = 55;
    cfg.nGrid_zeta  = 55;
    cfg.nGrid_beta  = 55;

    % ---------------- 热图扫描范围 ----------------
    cfg.kappa_range = linspace(0, 1, cfg.nGrid_kappa);
    cfg.zeta_range  = linspace(0, 1, cfg.nGrid_zeta);
    cfg.beta_range  = linspace(0.6, 1.6, cfg.nGrid_beta);

    % ---------------- 输出目录（保存图） ----------------
    out_dir = fullfile(pwd, '四月实验结果图');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    fprintf('============================================================\n');
    fprintf('GSRT-nSIS closure-steady heatmaps (>= version)\n');
    fprintf('不保存 csv / mat，仅保留工作区结果\n');
    fprintf('图像保存目录：%s\n', out_dir);
    fprintf('============================================================\n');

    %% ===================== 2. 预计算基线的时间结构因子 =====================
    LambdaTildeA_base = compute_LambdaTilde(cfg.aI, cfg.beta1, cfg.aR, cfg.bR, ...
                                            cfg.dt, cfg.Tmax_kernel, cfg.epsTail);
    LambdaTildeN_base = compute_LambdaTilde(cfg.aI, cfg.beta0, cfg.aR, cfg.bR, ...
                                            cfg.dt, cfg.Tmax_kernel, cfg.epsTail);

    % 对 beta 扫描，把所有 beta 值对应的 LambdaTilde 预先算好，加速
    beta_Ltilde = zeros(size(cfg.beta_range));
    for ib = 1:numel(cfg.beta_range)
        beta_Ltilde(ib) = compute_LambdaTilde(cfg.aI, cfg.beta_range(ib), ...
                            cfg.aR, cfg.bR, cfg.dt, cfg.Tmax_kernel, cfg.epsTail);
    end

    %% ===================== 3. 扫描三组参数 =====================
    results = struct();
    results.config = cfg;
    results.output_dir = out_dir;

    fprintf('\n[1/3] 扫描 (kappa0, kappa1) ...\n');
    [RHO_kappa, X_kappa, R0eff_kappa] = sweep_two_params_closure_geq( ...
        cfg.kappa_range, cfg.kappa_range, 'kappa', cfg, LambdaTildeN_base, LambdaTildeA_base, beta_Ltilde);

    results.kappa.kappa0_list = cfg.kappa_range;
    results.kappa.kappa1_list = cfg.kappa_range;
    results.kappa.rho_star = RHO_kappa;
    results.kappa.x_star = X_kappa;
    results.kappa.R0eff = R0eff_kappa;

    fprintf('\n[2/3] 扫描 (zeta0, zeta1) ...\n');
    [RHO_zeta, X_zeta, R0eff_zeta] = sweep_two_params_closure_geq( ...
        cfg.zeta_range, cfg.zeta_range, 'zeta', cfg, LambdaTildeN_base, LambdaTildeA_base, beta_Ltilde);

    results.zeta.zeta0_list = cfg.zeta_range;
    results.zeta.zeta1_list = cfg.zeta_range;
    results.zeta.rho_star = RHO_zeta;
    results.zeta.x_star = X_zeta;
    results.zeta.R0eff = R0eff_zeta;

    fprintf('\n[3/3] 扫描 (beta0, beta1) ...\n');
    [RHO_beta, X_beta, R0eff_beta] = sweep_two_params_closure_geq( ...
        cfg.beta_range, cfg.beta_range, 'beta', cfg, LambdaTildeN_base, LambdaTildeA_base, beta_Ltilde);

    results.beta.beta0_list = cfg.beta_range;
    results.beta.beta1_list = cfg.beta_range;
    results.beta.rho_star = RHO_beta;
    results.beta.x_star = X_beta;
    results.beta.R0eff = R0eff_beta;

    %% ===================== 4. 绘图（2×3 合并大图） =====================
    fig_path_prefix = fullfile(out_dir, 'heatmap_closure_2x3_combined_geq1');
    fig_combined = plot_heatmap_2x3_combined( ...
        cfg.kappa_range, cfg.kappa_range, RHO_kappa, X_kappa, ...
        cfg.zeta_range,  cfg.zeta_range,  RHO_zeta,  X_zeta, ...
        cfg.beta_range,  cfg.beta_range,  RHO_beta,  X_beta, ...
        fig_path_prefix);

    results.figure_handle = fig_combined;
    results.figure_prefix = fig_path_prefix;

    %% ===================== 5. 工作区输出 =====================
    assignin('base', 'closure_heatmap_results', results);
    assignin('base', 'RHO_kappa_closure', RHO_kappa);
    assignin('base', 'X_kappa_closure', X_kappa);
    assignin('base', 'RHO_zeta_closure', RHO_zeta);
    assignin('base', 'X_zeta_closure', X_zeta);
    assignin('base', 'RHO_beta_closure', RHO_beta);
    assignin('base', 'X_beta_closure', X_beta);

    fprintf('\n全部计算完成。\n');
    fprintf('结果已写入工作区变量：\n');
    fprintf('  closure_heatmap_results\n');
    fprintf('  RHO_kappa_closure, X_kappa_closure\n');
    fprintf('  RHO_zeta_closure,  X_zeta_closure\n');
    fprintf('  RHO_beta_closure,  X_beta_closure\n');
    fprintf('合并图像已保存到：\n');
    fprintf('  %s.fig\n', fig_path_prefix);
    fprintf('  %s.jpg\n', fig_path_prefix);
end

function [RHO_mat, X_mat, R0eff_mat] = sweep_two_params_closure_geq(list0, list1, mode, cfg, LambdaTildeN_base, LambdaTildeA_base, beta_Ltilde)

    n0 = numel(list0);
    n1 = numel(list1);

    RHO_mat   = NaN(n1, n0);
    X_mat     = NaN(n1, n0);
    R0eff_mat = NaN(n1, n0);

    [P0, P1] = meshgrid(list0, list1);

    switch lower(mode)
        case 'kappa'
            valid_mask = (P0 >= P1);
        case 'zeta'
            valid_mask = (P0 >= P1);
        case 'beta'
            valid_mask = (P1 >= P0);
        otherwise
            error('未知扫描模式：%s', mode);
    end

    progress_total = nnz(valid_mask);
    progress_count = 0;
    progress_stride = max(1, floor(progress_total / 20));

    fprintf('  模式 %s：有效参数点总数 = %d\n', mode, progress_total);

    for j = 1:n1
        prev_x = [];

        for i = 1:n0
            if ~valid_mask(j, i)
                continue;
            end

            params = cfg;

            switch lower(mode)
                case 'kappa'
                    params.kappa0 = list0(i);
                    params.kappa1 = list1(j);
                    LambdaTildeN = LambdaTildeN_base;
                    LambdaTildeA = LambdaTildeA_base;

                case 'zeta'
                    params.zeta0 = list0(i);
                    params.zeta1 = list1(j);
                    LambdaTildeN = LambdaTildeN_base;
                    LambdaTildeA = LambdaTildeA_base;

                case 'beta'
                    params.beta0 = list0(i);
                    params.beta1 = list1(j);
                    LambdaTildeN = beta_Ltilde(i);
                    LambdaTildeA = beta_Ltilde(j);
            end

            out = solve_closure_equilibrium(params, LambdaTildeN, LambdaTildeA, prev_x);

            RHO_mat(j, i)   = out.rho_star;
            X_mat(j, i)     = out.x_star;
            R0eff_mat(j, i) = out.R0eff;

            prev_x = out.x_star;

            progress_count = progress_count + 1;
            if mod(progress_count, progress_stride) == 0 || progress_count == progress_total
                fprintf('  %s 扫描进度：%4d / %4d\n', mode, progress_count, progress_total);
            end
        end
    end
end

function out = solve_closure_equilibrium(p, LambdaTildeN, LambdaTildeA, prev_x)
    zeta_bar   = @(x) x .* p.zeta1 + (1 - x) .* p.zeta0;
    Lambda_bar = @(x) x .* (p.kappa1 * LambdaTildeA) + (1 - x) .* (p.kappa0 * LambdaTildeN);

    rho_from_x = @(x) rho_given_x(x, p.k_avg, zeta_bar, Lambda_bar);
    F = @(x) behaviour_residual(x, rho_from_x(x), p.uAA, p.uNA, p.uNN, p.c, p.k_avg);

    lo = 1e-8;
    hi = 1 - 1e-8;

    x_grid = linspace(lo, hi, 2001);
    f_grid = arrayfun(F, x_grid);

    brackets = [];
    for k = 1:numel(x_grid)-1
        f1 = f_grid(k);
        f2 = f_grid(k+1);
        if isfinite(f1) && isfinite(f2) && sign(f1) ~= sign(f2)
            brackets = [brackets; x_grid(k), x_grid(k+1)]; %#ok<AGROW>
        end
    end

    if ~isempty(brackets)
        mids = mean(brackets, 2);
        if nargin >= 4 && ~isempty(prev_x) && isfinite(prev_x)
            [~, idx] = min(abs(mids - prev_x));
        else
            [~, idx] = min(abs(mids - 0.5));
        end
        br = brackets(idx,:);
        x_star = fzero(F, br);
    else
        obj = @(x) abs(F(x));
        if nargin >= 4 && ~isempty(prev_x) && isfinite(prev_x)
            lo2 = max(lo, prev_x - 0.2);
            hi2 = min(hi, prev_x + 0.2);
            if lo2 < hi2
                x_star = fminbnd(obj, lo2, hi2);
            else
                x_star = fminbnd(obj, lo, hi);
            end
        else
            x_star = fminbnd(obj, lo, hi);
        end
    end

    x_star = min(max(x_star, lo), hi);
    rho_star = rho_from_x(x_star);
    R0eff = zeta_bar(x_star) * Lambda_bar(x_star) * p.k_avg;

    out = struct();
    out.x_star = x_star;
    out.rho_star = rho_star;
    out.R0eff = R0eff;
end

function val = behaviour_residual(x, rho, uAA, uNA, uNN, c, k_avg)
    F_risk = 1 / max(1e-8, 1 - rho);

    Gamma0 = F_risk * (uNA - c) - uNN;
    Gamma1 = F_risk * (uAA - uNA + c) - (uNA - uNN);

    val = log(x ./ (1 - x)) - k_avg .* (Gamma0 + Gamma1 .* x);
end

function rho = rho_given_x(x, k_avg, zeta_bar, Lambda_bar)
    R = zeta_bar(x) .* Lambda_bar(x) .* k_avg;

    if R <= 1
        rho = 0;
    else
        rho = 1 - 1 ./ R;
    end
end

function Ltilde = compute_LambdaTilde(aI, betaI, aR, bR, dt, Tmax_kernel, epsTail)
    tauI_max = bR * (log(1/epsTail))^(1/aR);
    L_I = ceil(min(Tmax_kernel, tauI_max)/dt) + 1;

    tau = (0:L_I-1)' * dt;
    Psi_rec = exp(-(tau./bR).^aR);

    eta_rate = renewal_attempt_kernel(L_I, dt, aI, betaI);
    Ltilde = sum(eta_rate(:) .* Psi_rec(:)) * dt;
end

function eta_rate = renewal_attempt_kernel(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;

    surv0 = exp(- (tau0 ./ beta) .^ alpha);
    surv1 = exp(- (tau1 ./ beta) .^ alpha);
    q_mass = surv0 - surv1;
    q_mass = max(q_mass, 0);

    eta_mass = zeros(L, 1);
    eta_mass(1) = q_mass(1);

    for n = 2:L
        conv_sum = 0;
        for m = 1:n-1
            conv_sum = conv_sum + eta_mass(m) * q_mass(n - m);
        end
        eta_mass(n) = q_mass(n) + conv_sum;
    end

    eta_rate = eta_mass / dt;
end

function fig = plot_heatmap_2x3_combined( ...
    kappa0_list, kappa1_list, rho_kappa, x_kappa, ...
    zeta0_list, zeta1_list, rho_zeta, x_zeta, ...
    beta0_list, beta1_list, rho_beta, x_beta, ...
    save_prefix)

    fig = figure('Color', 'w', 'Position', [80, 60, 1450, 780]);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    rho_all = [rho_kappa(~isnan(rho_kappa)); rho_zeta(~isnan(rho_zeta)); rho_beta(~isnan(rho_beta))];
    x_all   = [x_kappa(~isnan(x_kappa));   x_zeta(~isnan(x_zeta));   x_beta(~isnan(x_beta))];

    rho_clim = [min(rho_all), max(rho_all)];
    x_clim   = [min(x_all),   max(x_all)];

    if abs(diff(rho_clim)) < 1e-12
        rho_clim = rho_clim + [-1, 1] * 1e-6;
    end
    if abs(diff(x_clim)) < 1e-12
        x_clim = x_clim + [-1, 1] * 1e-6;
    end

    cmap_rho = parula(256);
    cmap_x   = roseorangemap_local(256);

    ax1 = nexttile(tl, 1);
    draw_one_heatmap(ax1, kappa0_list, kappa1_list, rho_kappa, ...
        '$\kappa_0$', '$\kappa_1$', '$\rho^*$ in $(\kappa_0,\kappa_1)$', ...
        rho_clim, cmap_rho, true);

    ax2 = nexttile(tl, 2);
    draw_one_heatmap(ax2, zeta0_list, zeta1_list, rho_zeta, ...
        '$\zeta_0$', '$\zeta_1$', '$\rho^*$ in $(\zeta_0,\zeta_1)$', ...
        rho_clim, cmap_rho, true);

    ax3 = nexttile(tl, 3);
    draw_one_heatmap(ax3, beta0_list, beta1_list, rho_beta, ...
        '$\beta_0$', '$\beta_1$', '$\rho^*$ in $(\beta_0,\beta_1)$', ...
        rho_clim, cmap_rho, true);

    ax4 = nexttile(tl, 4);
    draw_one_heatmap(ax4, kappa0_list, kappa1_list, x_kappa, ...
        '$\kappa_0$', '$\kappa_1$', '$x^*$ in $(\kappa_0,\kappa_1)$', ...
        x_clim, cmap_x, true);

    ax5 = nexttile(tl, 5);
    draw_one_heatmap(ax5, zeta0_list, zeta1_list, x_zeta, ...
        '$\zeta_0$', '$\zeta_1$', '$x^*$ in $(\zeta_0,\zeta_1)$', ...
        x_clim, cmap_x, true);

    ax6 = nexttile(tl, 6);
    draw_one_heatmap(ax6, beta0_list, beta1_list, x_beta, ...
        '$\beta_0$', '$\beta_1$', '$x^*$ in $(\beta_0,\beta_1)$', ...
        x_clim, cmap_x, true);

    savefig(fig, [save_prefix, '.fig']);
    saveas(fig, [save_prefix, '.jpg']);
end

function draw_one_heatmap(ax, x_list, y_list, mat, xlab, ylab, ttl, clim_val, cmap_val, addContour)
    valid_mask = ~isnan(mat);

    h = imagesc(ax, x_list, y_list, mat);
    set(ax, 'YDir', 'normal', 'FontSize', 12);
    set(h, 'AlphaData', double(valid_mask));

    xlabel(ax, xlab, 'Interpreter', 'latex', 'FontSize', 14);
    ylabel(ax, ylab, 'Interpreter', 'latex', 'FontSize', 14);
    title(ax, ttl, 'Interpreter', 'latex', 'FontSize', 15);

    colormap(ax, cmap_val);
    caxis(ax, clim_val);
    colorbar(ax);

    grid(ax, 'on');
    box(ax, 'on');

    if addContour
        hold(ax, 'on');
        valid_vals = mat(~isnan(mat));
        if ~isempty(valid_vals)
            vmin = min(valid_vals);
            vmax = max(valid_vals);
            if vmax > vmin
                levels = linspace(vmin + 0.15*(vmax-vmin), vmax - 0.10*(vmax-vmin), 6);
                levels = unique(levels);
                [C, hcont] = contour(ax, x_list, y_list, mat, levels, ...
                    '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.9);
                clabel(C, hcont, 'Color', [0.35 0.35 0.35], 'FontSize', 8, ...
                    'BackgroundColor', 'none', 'Margin', 2);
            end
        end
    end
end

function cmap = roseorangemap_local(m)
    if nargin < 1
        m = 256;
    end

    % dark rose -> pink -> coral -> soft orange -> pale cream
    c1 = [0.38, 0.07, 0.22];
    c2 = [0.73, 0.24, 0.42];
    c3 = [0.93, 0.52, 0.55];
    c4 = [0.96, 0.69, 0.42];
    c5 = [0.99, 0.95, 0.88];

    n1 = round(0.22 * m);
    n2 = round(0.24 * m);
    n3 = round(0.24 * m);
    n4 = m - n1 - n2 - n3;

    part1 = [linspace(c1(1), c2(1), n1)', ...
             linspace(c1(2), c2(2), n1)', ...
             linspace(c1(3), c2(3), n1)'];

    part2 = [linspace(c2(1), c3(1), n2)', ...
             linspace(c2(2), c3(2), n2)', ...
             linspace(c2(3), c3(3), n2)'];

    part3 = [linspace(c3(1), c4(1), n3)', ...
             linspace(c3(2), c4(2), n3)', ...
             linspace(c3(3), c4(3), n3)'];

    part4 = [linspace(c4(1), c5(1), n4)', ...
             linspace(c4(2), c5(2), n4)', ...
             linspace(c4(3), c5(3), n4)'];

    cmap = [part1; part2; part3; part4];
end
