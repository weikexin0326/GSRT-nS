function exp6_pulse_disturbance_aligned_to_theory_new11()
% ============================================================

clc; clear; close all;

%% ------------------ 1. 实验参数设置 ------------------
dt = 0.01;
Tmax = 40;
t_series = 0:dt:Tmax;

% --- 关键时间节点 ---
T_early_step      = 3;   % 早期永久阶跃
T_steady_start    = 20;  % 稳态阶段开始施加干扰
T_pulse_duration  = 5;   % 临时脉冲持续时间
T_steady_pulse_end = T_steady_start + T_pulse_duration;

% --- 干扰强度参数 ---
pulse_kappaN_factor = 2.0;
pulse_c_factor      = 3.0;

% --- 理论模型参数（对齐 theory代码.txt） ---
N = 1000;
k_avg = 10;
aR = 2.0; bR = 0.5;
aI = 0.8; betaA = 1.2; betaN = 1.0;
kappaN_base = 1.0;
kappa_ratio = 0.5;
kappaA_base = kappa_ratio * kappaN_base;
chiA = 0.4; chiN = 1.0;
gamma_mem = 1.0;
behavior_dt = 1.0;
epsTail = 1e-8;

% 初始条件
init_rho = 0.05;
init_p   = 0.05;

% --- 三种博弈矩阵 ---
PMs = struct();
PMs(1).name = 'PM1 (Prisoner''s Dilemma)';
PMs(1).uAA = 0.01; PMs(1).uNA = 0.05; PMs(1).uNN = 0.10; PMs(1).c = 0.02;

PMs(2).name = 'PM2 (Intermediate)';
PMs(2).uAA = 0.05; PMs(2).uNA = 0.05; PMs(2).uNN = 0.10; PMs(2).c = 0.01;

PMs(3).name = 'PM3 (Coordination)';
PMs(3).uAA = 0.10; PMs(3).uNA = 0.05; PMs(3).uNN = 0.01; PMs(3).c = 0.01;

%% ------------------ 2. 网络构建 ------------------
rng(1);
Adj = ER_network(N, k_avg);
deg = full(sum(Adj,2));
deg = max(deg, 1e-20);
fprintf('Theory engine initialized. N=%d, <k>=%.1f\n', N, k_avg);

%% ------------------ 3. 循环三种博弈矩阵 ------------------
for pm_idx = 1:3
    CurrentPM = PMs(pm_idx);
    fprintf('\nProcessing %s ...\n', CurrentPM.name);

    uAA = CurrentPM.uAA;
    uNA = CurrentPM.uNA;
    uNN = CurrentPM.uNN;
    c_base = CurrentPM.c;

    kappaN_high = kappaN_base * pulse_kappaN_factor;
    kappaA_high = kappa_ratio * kappaN_high;
    c_high      = c_base * pulse_c_factor;

    P_stat = struct( ...
        'N', N, 'deg', deg, 'Adj', Adj, 'dt', dt, 'Tmax', Tmax, ...
        'aR', aR, 'bR', bR, 'aI', aI, 'betaA', betaA, 'betaN', betaN, ...
        'chiA', chiA, 'chiN', chiN, 'gamma_mem', gamma_mem, ...
        'uNN', uNN, 'uNA', uNA, 'uAA', uAA, 'behavior_dt', behavior_dt, ...
        'epsTail', epsTail);

    fprintf('  Calculating scenarios...\n');

    % [1] Baseline
    [R_base, X_base] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        0, 0, ...
        kappaA_base, kappaN_base, c_base, ...
        0, 0, 0);

    % [2] Holiday Temp
    [R_hol_temp, X_hol_temp] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        T_steady_start, T_steady_pulse_end, ...
        kappaA_base, kappaN_base, c_base, ...
        kappaA_high, kappaN_high, c_base);

    % [3] Fatigue Temp
    [R_fat_temp, X_fat_temp] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        T_steady_start, T_steady_pulse_end, ...
        kappaA_base, kappaN_base, c_base, ...
        kappaA_base, kappaN_base, c_high);

    % [4] Holiday Perm Early
    [R_hol_early, X_hol_early] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        T_early_step, Tmax + 1, ...
        kappaA_base, kappaN_base, c_base, ...
        kappaA_high, kappaN_high, c_base);

    % [5] Fatigue Perm Early
    [R_fat_early, X_fat_early] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        T_early_step, Tmax + 1, ...
        kappaA_base, kappaN_base, c_base, ...
        kappaA_base, kappaN_base, c_high);

    % [6] Holiday Perm Steady
    [R_hol_steady, X_hol_steady] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        T_steady_start, Tmax + 1, ...
        kappaA_base, kappaN_base, c_base, ...
        kappaA_high, kappaN_high, c_base);

    % [7] Fatigue Perm Steady
    [R_fat_steady, X_fat_steady] = solve_theory_scenario_aligned(P_stat, init_rho, init_p, ...
        T_steady_start, Tmax + 1, ...
        kappaA_base, kappaN_base, c_base, ...
        kappaA_base, kappaN_base, c_high);

    %% ------------------ 4. 绘图 ------------------
    fprintf('  Plotting results for %s...\n', CurrentPM.name);
    figure('Color','w', 'Position', [50 50 1200 800]);

    col_base        = [0.2 0.2 0.2];
    col_temp        = [0 0.5 0];
    col_perm_early  = [1 0.6 0];
    col_perm_steady = [0.8 0 0];
    col_shadow      = [0.92 0.92 0.92];
    col_pulse_line  = [0.35 0.35 0.35];   % 深灰色脉冲起始时刻垂线

    patch_x = [T_steady_start, T_steady_pulse_end, T_steady_pulse_end, T_steady_start];

    % ================= 左列：Scenario A (Holiday) =================
    subplot(2,2,1); hold on; grid on; box on;
    all_R_hol = [R_base; R_hol_temp; R_hol_early; R_hol_steady];
    y_lim_R = [0, max(all_R_hol)*1.1];
    ylim(y_lim_R);
    fill(patch_x, [y_lim_R(1), y_lim_R(1), y_lim_R(2), y_lim_R(2)], ...
        col_shadow, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    xline(T_early_step,   '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    xline(T_steady_start, '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    p1 = plot(t_series, R_base,       'Color', col_base,       'LineWidth', 2);
    p2 = plot(t_series, R_hol_temp,   '--',    'Color', col_temp,       'LineWidth', 2);
    p3 = plot(t_series, R_hol_early,  '-.',    'Color', col_perm_early, 'LineWidth', 2);
    p4 = plot(t_series, R_hol_steady, ':',     'Color', col_perm_steady,'LineWidth', 3);
    ylabel('Infection \rho(t)', 'FontSize', 12);
    title('Scenario A: Holiday Gathering (Increased \kappa)', 'FontSize', 12, 'FontWeight', 'bold');
    legend([p1,p2,p3,p4], {'Baseline', 'Temp Pulse (Steady)', 'Perm Step (Early)', 'Perm Step (Steady)'}, ...
        'Location', 'best', 'FontSize', 9);
    set(gca, 'Layer', 'top');

    subplot(2,2,3); hold on; grid on; box on;
    all_X_hol = [X_base; X_hol_temp; X_hol_early; X_hol_steady];
    y_lim_X = [0, max(all_X_hol)*1.1];
    ylim(y_lim_X);
    fill(patch_x, [y_lim_X(1), y_lim_X(1), y_lim_X(2), y_lim_X(2)], ...
        col_shadow, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    xline(T_early_step,   '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    xline(T_steady_start, '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    plot(t_series, X_base,       'Color', col_base,       'LineWidth', 2);
    plot(t_series, X_hol_temp,   '--',    'Color', col_temp,       'LineWidth', 2);
    plot(t_series, X_hol_early,  '-.',    'Color', col_perm_early, 'LineWidth', 2);
    plot(t_series, X_hol_steady, ':',     'Color', col_perm_steady,'LineWidth', 3);
    ylabel('Adoption x(t)', 'FontSize', 12);
    xlabel('Time', 'FontSize', 12);
    set(gca, 'Layer', 'top');

    % ================= 右列：Scenario B (Fatigue) =================
    subplot(2,2,2); hold on; grid on; box on;
    all_R_fat = [R_base; R_fat_temp; R_fat_early; R_fat_steady];
    y_lim_R_fat = [0, max(all_R_fat)*1.1];
    ylim(y_lim_R_fat);
    fill(patch_x, [y_lim_R_fat(1), y_lim_R_fat(1), y_lim_R_fat(2), y_lim_R_fat(2)], ...
        col_shadow, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    xline(T_early_step,   '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    xline(T_steady_start, '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    plot(t_series, R_base,       'Color', col_base,       'LineWidth', 2);
    plot(t_series, R_fat_temp,   '--',    'Color', col_temp,       'LineWidth', 2);
    plot(t_series, R_fat_early,  '-.',    'Color', col_perm_early, 'LineWidth', 2);
    plot(t_series, R_fat_steady, ':',     'Color', col_perm_steady,'LineWidth', 3);
    ylabel('Infection \rho(t)', 'FontSize', 12);
    title('Scenario B: Protection Fatigue (Increased Cost c)', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'Layer', 'top');

    subplot(2,2,4); hold on; grid on; box on;
    all_X_fat = [X_base; X_fat_temp; X_fat_early; X_fat_steady];
    y_lim_X_fat = [0, max(all_X_fat)*1.1];
    ylim(y_lim_X_fat);
    fill(patch_x, [y_lim_X_fat(1), y_lim_X_fat(1), y_lim_X_fat(2), y_lim_X_fat(2)], ...
        col_shadow, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    xline(T_early_step,   '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    xline(T_steady_start, '--', 'Color', col_pulse_line, 'LineWidth', 1.6);
    plot(t_series, X_base,       'Color', col_base,       'LineWidth', 2);
    plot(t_series, X_fat_temp,   '--',    'Color', col_temp,       'LineWidth', 2);
    plot(t_series, X_fat_early,  '-.',    'Color', col_perm_early, 'LineWidth', 2);
    plot(t_series, X_fat_steady, ':',     'Color', col_perm_steady,'LineWidth', 3);
    ylabel('Adoption x(t)', 'FontSize', 12);
    xlabel('Time', 'FontSize', 12);
    set(gca, 'Layer', 'top');

    sgtitle(['System Resilience vs. Regime Shift (' CurrentPM.name ')'], ...
        'FontSize', 16, 'FontWeight', 'bold');

    saveas(gcf, ['PulseDisturbance_PM' num2str(pm_idx) '_aligned_to_theory_code1.fig']);
    saveas(gcf, ['PulseDisturbance_PM' num2str(pm_idx) '_aligned_to_theory_code1.jpg']);
end

end

%% ===================== 对齐 theory代码.txt 的理论求解器 =====================
function [rho_res, x_res] = solve_theory_scenario_aligned(P, init_rho, init_p, ...
    t_mod_start, t_mod_end, ...
    kB_A, kB_N, c_base, ...
    kP_A, kP_N, c_pulse)

    % ---- Unpack ----
    dt = P.dt; Tmax = P.Tmax; N = P.N; deg = P.deg; Adj = P.Adj;
    aR = P.aR; bR = P.bR; aI = P.aI; betaA = P.betaA; betaN = P.betaN;
    chiA = P.chiA; chiN = P.chiN; gamma_mem = P.gamma_mem;
    uNN = P.uNN; uNA = P.uNA; uAA = P.uAA; behavior_dt = P.behavior_dt;
    epsTail = P.epsTail;

    nSteps = ceil(Tmax / dt);

    % ---- 感染龄截断与核函数（对齐 theory代码.txt） ----
    tauI_max = bR * (log(1 / epsTail))^(1 / aR);
    L_I = ceil(min(Tmax, tauI_max) / dt) + 1;

    p_rec = weibull_condprob(L_I, dt, aR, bR);
    sRec  = 1 - p_rec(:)';

    eta_rate_A = renewal_attempt_kernel(L_I, dt, aI, betaA);
    eta_rate_N = renewal_attempt_kernel(L_I, dt, aI, betaN);
    etaA = eta_rate_A(:);
    etaN = eta_rate_N(:);

    % ---- 记忆核离散系数（对齐 theory代码.txt） ----
    if gamma_mem > 0
        decayH = exp(-gamma_mem * dt);
        addH   = 1 - decayH;
    else
        decayH = 0;
        addH   = 1;
    end

    % ---- 初始化 ----
    PA = zeros(N,1);
    PA(randperm(N, max(1, round(init_p * N)))) = 1.0;

    IA_age = zeros(N, L_I);
    IN_age = zeros(N, L_I);
    init_inf_idx = randperm(N, max(1, round(init_rho * N)));
    IA_age(init_inf_idx,1) = PA(init_inf_idx);
    IN_age(init_inf_idx,1) = 1 - PA(init_inf_idx);

    H = zeros(N,1);
    next_update_time = behavior_dt * rand(N,1);

    rho_res = zeros(nSteps + 1, 1);
    x_res   = zeros(nSteps + 1, 1);
    rho_res(1) = mean(sum(IA_age,2) + sum(IN_age,2));
    x_res(1)   = mean(PA);

    t_now = 0;

    for step = 1:nSteps
        t_end = t_now + dt;

        % ---- 动态参数切换 ----
        if (t_now >= t_mod_start) && (t_now < t_mod_end)
            cur_kA = kP_A;
            cur_kN = kP_N;
            cur_c  = c_pulse;
        else
            cur_kA = kB_A;
            cur_kN = kB_N;
            cur_c  = c_base;
        end

        % ---------------- (1) 当前感染质量 ----------------
        IA_before = sum(IA_age, 2);
        IN_before = sum(IN_age, 2);
        I_before  = IA_before + IN_before;

        % ---------------- (2) 源侧感染压力 Phi ----------------
        phi_A = cur_kA * (IA_age * etaA);
        phi_N = cur_kN * (IN_age * etaN);
        Phi   = phi_A + phi_N;

        % ---------------- (3) 空间聚合 Lambda ----------------
        Lambda = Adj * Phi;

        % ---------------- (4) 记忆更新 H ----------------
        if gamma_mem > 0
            H = decayH * H + addH * Lambda;
        else
            H = Lambda;
        end
        H = max(H, 0);

        % ---------------- (5) 感染更新 ----------------
        SA = max(0, PA - IA_before);
        SN = max(0, (1 - PA) - IN_before);

        p_inf_A = 1 - exp(-chiA * H * dt);
        p_inf_N = 1 - exp(-chiN * H * dt);
        p_inf_A = min(max(p_inf_A, 0), 1);
        p_inf_N = min(max(p_inf_N, 0), 1);

        newA = SA .* p_inf_A;
        newN = SN .* p_inf_N;

        % ---------------- (6) 恢复 + 感染龄推进 ----------------
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

        I_after = sum(IA_age,2) + sum(IN_age,2);

        % ---------------- (7) 博弈更新 ----------------
        due = (next_update_time <= t_end + 1e-12);
        if any(due)
            neigh_A = Adj * PA;
            neigh_N = deg - neigh_A;
            rho_global = mean(I_after);
            F_risk = 1 / max(1e-5, 1 - rho_global);

            pi_N = uNN .* neigh_N + uNA .* neigh_A;
            pi_A = (uNA - cur_c) .* neigh_N + uAA .* neigh_A;
            z_N  = pi_N;
            z_A  = pi_A .* F_risk;

            max_z  = max(z_N, z_A);
            prob_A = exp(z_A - max_z) ./ (exp(z_N - max_z) + exp(z_A - max_z) + 1e-20);

            idx_due = find(due);
            prob_due = prob_A(idx_due);
            PA(idx_due) = prob_due;

            % 对齐 theory代码.txt：按当前策略概率重新分配感染龄质量
            I_age_total = IA_age(idx_due,:) + IN_age(idx_due,:);
            IA_age(idx_due,:) = I_age_total .* prob_due;
            IN_age(idx_due,:) = I_age_total .* (1 - prob_due);

            next_update_time(idx_due) = next_update_time(idx_due) + behavior_dt;
        end

        % ---------------- (8) 记录 ----------------
        rho_res(step + 1) = mean(sum(IA_age,2) + sum(IN_age,2));
        x_res(step + 1)   = mean(PA);
        t_now = t_end;
    end
end

%% ===================== 辅助函数 =====================
function pcond = weibull_condprob(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;
    dH = (tau1 ./ beta).^alpha - (tau0 ./ beta).^alpha;
    pcond = -expm1(-dH);
    pcond = min(max(pcond, 0), 1)';
end

function eta_rate = renewal_attempt_kernel(L, dt, alpha, beta)
% 离散 renewal 版本的尝试核
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;

    surv0 = exp(-(tau0 ./ beta).^alpha);
    surv1 = exp(-(tau1 ./ beta).^alpha);
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
    A = sprand(N, N, k_avg / N) > 0;
    A = triu(A, 1);
    A = A + A';
    A = sparse(A);
end
