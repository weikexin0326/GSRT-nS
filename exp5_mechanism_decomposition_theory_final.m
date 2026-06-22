function [results, experiments, P, t_series, Adj_Fixed] = ...
    exp5_mechanism_decomposition_aligned_final_theory()
% =========================================================================
% 机制贡献分解实验（与最终 theory 代码底层逻辑对齐版）
% -------------------------------------------------------------------------
% 对齐要点：
% 1) 移除二阶 Jensen 修正（不再使用 H_var，也不再使用二阶近似指数项）；
% 2) 不再显式跟踪边龄质量 EA / EN；
% 3) 源端压力直接由感染年龄密度 IA_age / IN_age 与 eta(tau_I|X) 卷积得到；
% 4) 记忆变量只保留 H 的均值演化；
% 5) 感染概率统一为 P_inf = 1 - exp(-chi_X * H * dt)；
% 6) 行为更新后，对感染年龄质量按当前策略状态重新分配，保持与最终 theory 代码一致。
% =========================================================================

clc; clear; close all;

%% ------------------ 1. 基础参数设置 ------------------
dt   = 0.01;
Tmax = 12;
t_series = (0:dt:Tmax)';

P = struct();
P.N = 1000;
P.k_avg = 10;

% --- Weibull 参数 ---
P.aR = 2.0;
P.bR = 0.5;
P.aI = 0.8;            % 基准 alpha_I
P.betaA = 1.2;
P.betaN = 1.0;

% --- 双重调制 ---
P.kappaN = 1.0;
P.kappa_ratio = 0.5;   % kappaA = kappaN * kappa_ratio = 0.5
P.chiA = 0.5;
P.chiN = 1.0;

% --- 记忆与博弈 ---
P.gamma_mem = 1.0;     % g(u)=exp(-u) 对应 gamma_mem = 1
P.uNN = 0.1;
P.uNA = 0.05;
P.uAA = 0.01;
P.c   = 0.02;
P.behavior_dt = 1.0;

% --- 初始条件 ---
P.init_rho = 0.05;
P.init_p   = 0.05;

%% ------------------ 2. 定义实验场景 ------------------
experiments = {};

% 1. Full Model（基准）
experiments{end+1} = struct('name', 'Full Model', 'params_override', struct());

% 2. No Game（固定行为）
experiments{end+1} = struct('name', 'No Game (Fixed Behavior)', ...
    'params_override', struct('no_game', true));

% 3. Markovian Spread（alpha_I = 1，保持平均等待时间一致）
mean_tau_A = P.betaA * gamma(1 + 1 / P.aI);
mean_tau_N = P.betaN * gamma(1 + 1 / P.aI);
experiments{end+1} = struct('name', 'Markovian Spread ($\alpha_{I}=1$)', ...
    'params_override', struct('aI', 1.0, 'betaA', mean_tau_A, 'betaN', mean_tau_N));

% 4. Instantaneous Risk（无记忆极限）
experiments{end+1} = struct('name', 'Instantaneous Risk ($\gamma \to \infty$)', ...
    'params_override', struct('gamma_mem', 1e9));

% 5. Self-interest Only（仅利己：A 不削弱对他人的传播）
experiments{end+1} = struct('name', 'Self-interest Only', ...
    'params_override', struct('kappa_ratio', 1.0, 'betaA', P.betaN));

% 6. Altruism Only（仅利他：A 不降低自身易感性）
experiments{end+1} = struct('name', 'Altruism Only', ...
    'params_override', struct('chiA', P.chiN));

%% ------------------ 3. 求解所有场景 ------------------
results = struct();
num_exps = numel(experiments);

% 预生成固定 ER 网络，避免拓扑波动影响分解比较
rng(1);
Adj_Fixed = ER_network(P.N, P.k_avg);

for i = 1:num_exps
    exp_name = experiments{i}.name;
    fprintf('Solving (%d/%d): %s ...\n', i, num_exps, exp_name);

    CurrentP = P;
    override_fields = fieldnames(experiments{i}.params_override);
    for j = 1:numel(override_fields)
        f = override_fields{j};
        CurrentP.(f) = experiments{i}.params_override.(f);
    end

    [rho, x] = solve_theory_ablation_aligned(dt, Tmax, CurrentP, Adj_Fixed);

    vname = matlab.lang.makeValidName(exp_name);
    results.(vname).rho = rho;
    results.(vname).x   = x;
end

%% ------------------ 4. 绘图 ------------------
fprintf('Plotting results...\n');
figure('Color','w', 'Position', [100 100 1000 900]);

colors = lines(num_exps);
linestyles = {'-', '--', ':', '-.', '-', ':'};
linewidths = 2.5 * ones(1, num_exps);

% --- 上图：rho(t) ---
ax1 = subplot(2,1,1); hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
all_rho = [];
for i = 1:num_exps
    vname = matlab.lang.makeValidName(experiments{i}.name);
    rr = results.(vname).rho(:);
    all_rho = [all_rho; rr]; %#ok<AGROW>
    plot(ax1, t_series, rr, ...
        'DisplayName', experiments{i}.name, ...
        'Color', colors(i,:), 'LineStyle', linestyles{i}, 'LineWidth', linewidths(i));
end

title(ax1, 'Mechanism Ablation Study', 'FontSize', 16, 'Interpreter', 'latex');
ylabel(ax1, 'Infection Density $\rho(t)$', 'FontSize', 14, 'Interpreter', 'latex');
legend(ax1, 'Location', 'southeast', 'FontSize', 10, 'Interpreter', 'latex', 'NumColumns', 2);
xlim(ax1, [0, Tmax]);
ylim(ax1, [0, min(1, 1.15 * max(all_rho))]);
set(ax1, 'FontSize', 12, 'TickLabelInterpreter', 'latex');

% --- 下图：x(t) ---
ax2 = subplot(2,1,2); hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
all_x = [];
for i = 1:num_exps
    vname = matlab.lang.makeValidName(experiments{i}.name);
    xx = results.(vname).x(:);
    all_x = [all_x; xx]; %#ok<AGROW>
    plot(ax2, t_series, xx, ...
        'DisplayName', experiments{i}.name, ...
        'Color', colors(i,:), 'LineStyle', linestyles{i}, 'LineWidth', linewidths(i));
end

ylabel(ax2, 'Protection Fraction $x(t)$', 'FontSize', 14, 'Interpreter', 'latex');
xlabel(ax2, 'Time $t$', 'FontSize', 14, 'Interpreter', 'latex');
xlim(ax2, [0, Tmax]);
ylim(ax2, [0, min(1, 1.15 * max(all_x))]);
set(ax2, 'FontSize', 12, 'TickLabelInterpreter', 'latex');

saveas(gcf,['四月实验结果图/机制分解-理论-final2',],'fig');
saveas(gcf,['四月实验结果图/机制分解-理论-final2',],'jpg');

% ===== 保存所有关键变量到 workspace =====
assignin('base','results',results);
assignin('base','experiments',experiments);
assignin('base','P',P);
assignin('base','t_series',t_series);
assignin('base','Adj_Fixed',Adj_Fixed);

fprintf('\nAll variables have been exported to workspace.\n');
end

%% ========================================================================
%  核心求解函数：与最终 theory 代码底层逻辑对齐
%% ========================================================================
function [rho_series, x_series] = solve_theory_ablation_aligned(dt, Tmax, P, Adj)

    rng(1);  % 保证不同场景初始感染节点一致

    nSteps = ceil(Tmax / dt);
    rho_series = zeros(nSteps+1, 1);
    x_series   = zeros(nSteps+1, 1);

    kappaA = P.kappaN * P.kappa_ratio;
    kappaN = P.kappaN;
    game_on = ~(isfield(P, 'no_game') && P.no_game);

    deg = full(sum(Adj,2));
    deg = max(deg, 1e-20);

    % --- Infection-age truncation from recovery survival ---
    epsTail = 1e-8;
    tauI_max = P.bR * (log(1/epsTail))^(1 / P.aR);
    L_I = ceil(min(Tmax, tauI_max) / dt) + 1;

    % Recovery conditional probability per dt interval
    p_rec = weibull_condprob(L_I, dt, P.aR, P.bR);
    pRec = p_rec(:)';
    sRec = 1 - pRec;

    % Strategy-dependent infection-age kernel eta(tau_I | X)
    eta_rate_A = renewal_attempt_kernel(L_I, dt, P.aI, P.betaA);
    eta_rate_N = renewal_attempt_kernel(L_I, dt, P.aI, P.betaN);
    etaA = eta_rate_A(:);
    etaN = eta_rate_N(:);

    % --- Initialization ---
    PA = zeros(P.N,1);
    PA(randperm(P.N, max(1, round(P.init_p * P.N)))) = 1.0;

    IA_age = zeros(P.N, L_I);
    IN_age = zeros(P.N, L_I);
    init_inf_idx = randperm(P.N, max(1, round(P.init_rho * P.N)));
    IA_age(init_inf_idx,1) = PA(init_inf_idx);
    IN_age(init_inf_idx,1) = 1 - PA(init_inf_idx);

    % Memory state H only (no second-order correction)
    H = zeros(P.N,1);
    next_update_time = P.behavior_dt * rand(P.N,1);

    rho_series(1) = mean(sum(IA_age,2) + sum(IN_age,2));
    x_series(1)   = mean(PA);

    if P.gamma_mem > 1e6
        decayH = 0;
        addH   = 1;
    elseif P.gamma_mem > 0
        decayH = exp(-P.gamma_mem * dt);
        addH   = 1 - decayH;
    else
        decayH = 0;
        addH   = 1;
    end

    t_now = 0;
    for step = 1:nSteps
        t_end = t_now + dt;

        % ---------------- (1) Current infected masses ----------------
        IA_before = sum(IA_age, 2);
        IN_before = sum(IN_age, 2);
        I_before  = IA_before + IN_before;

        % ---------------- (2) Source-side pressure (aligned with final theory) ---
        % Phi_j(t) = int kappa_X * eta(tau_I|X) * I_j(tau_I;t) d tau_I
        phi_A = kappaA * (IA_age * etaA);
        phi_N = kappaN * (IN_age * etaN);
        Phi   = phi_A + phi_N;

        % ---------------- (3) Spatial aggregation -------------------------------
        Lambda = Adj * Phi;

        % ---------------- (4) Memory evolution (mean only) ----------------------
        if P.gamma_mem > 1e6 || P.gamma_mem <= 0
            H = Lambda;
        else
            H = decayH * H + addH * Lambda;
        end
        H = max(H, 0);

        % ---------------- (5) Infection: NO Jensen correction -------------------
        SA = max(0, PA - IA_before);
        SN = max(0, (1 - PA) - IN_before);

        p_inf_A = 1 - exp(-P.chiA * H * dt);
        p_inf_N = 1 - exp(-P.chiN * H * dt);
        p_inf_A = min(max(p_inf_A, 0), 1);
        p_inf_N = min(max(p_inf_N, 0), 1);

        newA = SA .* p_inf_A;
        newN = SN .* p_inf_N;

        % ---------------- (6) Recovery + infection-age shift --------------------
        IA_surv = IA_age .* sRec;
        IN_surv = IN_age .* sRec;

        IA_next = zeros(P.N, L_I);
        IN_next = zeros(P.N, L_I);
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

        % ---------------- (7) Game update (Eq. 3) ------------------------------
        due = (next_update_time <= t_end + 1e-12);
        if any(due)
            if game_on
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

                idx_due  = find(due);
                prob_due = prob_A(idx_due);
                PA(idx_due) = prob_due;

                % Repartition infection-age mass by CURRENT strategy state
                I_age_total = IA_age(idx_due,:) + IN_age(idx_due,:);
                IA_age(idx_due,:) = I_age_total .* prob_due;
                IN_age(idx_due,:) = I_age_total .* (1 - prob_due);
            end

            next_update_time(due) = next_update_time(due) + P.behavior_dt;
        end

        % ---------------- (8) Record -------------------------------------------
        rho_series(step+1) = mean(sum(IA_age,2) + sum(IN_age,2));
        x_series(step+1)   = mean(PA);
        t_now = t_end;
    end
end

%% ========================================================================
%  Helper functions
%% ========================================================================
function pcond = weibull_condprob(L, dt, alpha, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;
    dH = (tau1 ./ beta).^alpha - (tau0 ./ beta).^alpha;
    pcond = -expm1(-dH);
    pcond = min(max(pcond,0),1)';
end

function eta_rate = renewal_attempt_kernel(L, dt, alpha, beta)
% ------------------------------------------------------------
% Discrete renewal version of Eq. (10)
%   eta(t) = psi(t) + integral_0^t eta(s) psi(t-s) ds
%
% q_mass(n) = P(T in ((n-1)dt, n dt]) for Weibull waiting-time T
% eta_mass(n) = expected number of attempt events in nth time bin
% eta_rate = eta_mass / dt
% ------------------------------------------------------------
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
    A = sprand(N, N, k_avg/(N-1));
    A = spones(triu(A,1));
    A = A + A';
end
