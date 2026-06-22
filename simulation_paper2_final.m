function sim_out = simulation_paper2_final()
% ============================================================
% 包含特性：
% 1. [关键修正] S->I 时，重置该节点所有出边的传播尝试年龄 (tau_e = 0)。
% 2. [并行计算] 使用 parfor 加速 run_reps 循环。
% 3. [双重调控] 完整的非马尔可夫传播 + 演化博弈双向耦合。
% 4. [高效实现] 向量化计算 + 邻接表预索引。
% τ_a (代码中的 tau_e): 不重置，继续累积。
% τ_a 代表的是"自上次传播尝试以来经过的时间”。这是一个物理事实。一个感染者改变策略，并不会改变他上一次传播尝试的时间。因此，τ_a 的计时器应该继续走。
% β_X (代码中的 beta_eff): 立即更新。
% β_X 代表的是当前策略下的传播尝试时间尺度。当一个感染者改变策略，他产生下一次传播尝试的"频率"或“节奏”会立刻改变。因此，在计算下一个时间步的尝试发生概率时，必须使用新策略对应的 β_X 值。

% Behavior update:异步 (asynchronous staggered phases, same period Tb)
% ------------------------------------------------------------
% Each node i has a phase phi_i ~ U(0,behavior_dt). It updates at times:
%   t_i(m) = phi_i + m*behavior_dt,  m=0,1,2,...
% Within each small dt step, we collect nodes whose next update time <= t_end,
% and update those nodes simultaneously using the snapshot at t_end.
%
% Outputs:
%   sim_out.t_series   : time grid (Tn x 1)
%   sim_out.aI_list    : list of alpha_I values
%   sim_out.rho_mean   : mean infection ratio over run_reps (Tn x numAlpha)
%   sim_out.x_mean     : mean protection ratio over run_reps (Tn x numAlpha)
%   sim_out.params     : params struct
%
% Strategy vector name:
%   X_celue : N x 1 (0=N, 1=A)
% Time series name:
%   t_series : 1 x Tn (stored as column in output)
%
% Key modeling detail:
%   Maintain edge age tau_e (since last attempt).
%   Each dt, attempt occurs with conditional prob:
%     p = 1-exp( -[( (tau+dt)/beta )^aI - (tau/beta)^aI ] )
%   事件在 tau 时刻还未发生的情况下，在接下来的 [tau, tau + dt] 时间段内发生的概率[weibull分布]
%   beta depends on CURRENT source strategy, but tau does NOT reset when strategy changes.
% ============================================================

    % ---------------- 1. 参数设置 (Parameters) ----------------
    % 时间相关
    dt   = 0.01;
    Tmax = 12; 
    t_series = 0:dt:Tmax;
    Tn = numel(t_series);

    % 网络相关
    N = 1000;
    k_avg = 10;
    p_edge = k_avg/(N-1); 

    % 恢复过程 (Weibull Recovery)
    aR = 2.0; bR = 0.5;

    % 传播过程 (Weibull Attempt)
    % 可以在此处修改列表，例如 [0.8 1.2 1.6]
    aI_list = [0.6, 0.8 1.0 1.2]; 
    
    % 策略依赖的传播参数 (Dual Modulation)
    % Beta: 影响时间尺度 (尝试频率)
    betaA = 1.2;    betaN = 1.0;  
    % Kappa: 影响感染强度 (每次尝试的剂量)
    kappaA = 0.5;   kappaN = 1.0; 
    % Chi: 影响易感性 (接收端的防御)
    chiA   = 0.4;   chiN   = 1.0; 

    % 记忆核 (指数核)
    gamma_mem = 1.0;

    % 博弈参数 (Game Payoff - Prisoner's Dilemma)
    c = 0.02; 
    uNN = 0.1; uNA = 0.05; uAA = 0.01;
    behavior_dt = 1.0; % 行为更新周期

    % 初始条件
    init_rho = 0.05;
    init_p   = 0.05;

    % 运行次数 (parfor 并行下建议设置较大，如 50-100 以获得平滑曲线)
    run_reps = 50; 
    base_seed = 1;

    fprintf('SIMULATION START: N=%d, <k>=%.1f, dt=%.3g, Reps=%d\n', N, k_avg, dt, run_reps);

    % ---------------- 2. 构建网络 (Fixed Structure) ----------------
    % 为了控制变量，所有 rep 使用相同的网络拓扑，但初始感染源不同
    rng(base_seed);
    Adj = sprand(N,N,p_edge);
    Adj = spones(triu(Adj,1));
    Adj = Adj + Adj';
    Adj = Adj - diag(diag(Adj)); % 确保无自环

    % 预处理网络结构：构建“出边索引”，用于快速重置 tau
    [src, dst] = find(Adj);   % 所有有向边的源和目
    E = numel(src);           % 总有向边数
    
    % out_edges{i} 存储节点 i 所有出边在全局边列表 E 中的索引
    out_edges = cell(N,1);
    for e = 1:E
        u = src(e);
        out_edges{u} = [out_edges{u}; e];
    end
    
    % 计算度 (用于博弈归一化)
    deg = full(sum(Adj,2));

    % ---------------- 3. 主循环 (Loop over alpha_I) ----------------
    numAlpha = numel(aI_list);
    rho_mean_mat = zeros(Tn, numAlpha);
    x_mean_mat   = zeros(Tn, numAlpha);

    for ai_idx = 1:numAlpha
        aI = aI_list(ai_idx);
        fprintf('  Processing alpha_I = %.2f ...\n', aI);

        % 使用临时变量存储并行结果的累加值
        rho_sum_total = zeros(Tn,1);
        x_sum_total   = zeros(Tn,1);

        % ==== 并行循环开始 ====
        parfor rep = 1:run_reps
            % 注意：在 parfor 中不要固定 rng 种子，否则所有 worker 会产生相同随机数
            
            % --- 状态初始化 ---
            state = zeros(N,1);       % 0:S, 1:I
            X_celue = zeros(N,1);     % 0:N, 1:A (策略)

            % 随机初始策略
            X_init_indices = randperm(N, round(init_p*N));
            X_celue(X_init_indices) = 1;

            % 随机初始感染
            initI = randperm(N, round(init_rho*N));
            state(initI) = 1;
            
            % 初始化感染年龄 (Node Infection Age)
            inf_age = zeros(N,1); 

            % 初始化边的传播尝试年龄 (Edge Attempt Age, tau)
            % 物理含义：自上次尝试传播以来经过的时间
            % 初始时假设均为 0
            tau_e = zeros(E,1); 
            
            % 初始化记忆压力 (Memory H)
            H = zeros(N,1); 

            % 异步更新的时间表
            next_update_time = behavior_dt * rand(N,1);
            
            % 单次运行的记录容器
            rho_run = zeros(Tn,1);
            x_run   = zeros(Tn,1);
            
            % 记录 t=0
            rho_run(1) = mean(state);
            x_run(1)   = mean(X_celue);

            t_now = 0;

            % ---- 时间步进循环 (Time Evolution) ----
            for ti = 2:Tn
                t_end = t_now + dt;
                
                % 1. 识别当前感染者及其相关边
                infected_nodes_mask = (state == 1);
                
                % 只有源头是 I 的边才会计时并尝试传播
                active_e = infected_nodes_mask(src); 
                
                Lambda_count = zeros(N,1);

                if any(active_e)
                    % --- 核心：非马尔可夫传播尝试 (Edge-based) ---
                    tau_old = tau_e(active_e);
                    tau_new = tau_old + dt;
                    
                    % 获取源节点的策略，决定 beta 和 kappa
                    Xs = X_celue(src(active_e)); 
                    beta_eff  = betaN + (betaA - betaN).*Xs; 
                    kappa_eff = kappaN + (kappaA - kappaN).*Xs;

                    % Weibull 风险计算: P(event in dt | survive to tau)
                    % hazard rate 积分 = H(t+dt) - H(t)
                    dH = (tau_new ./ beta_eff).^aI - (tau_old ./ beta_eff).^aI;
                    p_att = -expm1(-dH); % 等价于 1 - exp(-dH)，精度更高
                    p_att = min(max(p_att,0),1);

                    % 蒙特卡洛抽样
                    u_rand = rand(sum(active_e),1);
                    did_att = (u_rand < p_att);
                    
                    % 更新 tau: 
                    % 1. 发生了尝试的边 -> 重置为 0
                    % 2. 未发生尝试的边 -> 继续累积 (tau_new)
                    tau_updated = tau_new;
                    tau_updated(did_att) = 0;
                    tau_e(active_e) = tau_updated; % 写回全局数组

                    % 统计成功的传播压力
                    if any(did_att)
                        % 找出真正激发的边在全局列表中的索引
                        act_idx_global = find(active_e);
                        edge_fired = act_idx_global(did_att);
                        
                        target_nodes = dst(edge_fired);
                        weights      = kappa_eff(did_att);
                        
                        % 将压力累加到目标节点
                        Lambda_count = accumarray(target_nodes, weights, [N,1], @sum, 0);
                    end
                end

                % 2. 记忆演化 (Memory Evolution: Integral of Lambda)
                Lambda_rate = Lambda_count ./ dt;
                if gamma_mem > 0
                    decay = exp(-gamma_mem*dt);
                    H = H*decay + Lambda_rate * ((1 - decay)/gamma_mem);
                else
                    H = Lambda_rate;
                end

                % 3. 感染事件 (S -> I)
                chi_vec = chiN*ones(N,1);
                chi_vec(X_celue==1) = chiA; % 接收端策略调节易感性

                sus_mask = (state == 0);
                new_infected = [];
                
                if any(sus_mask)
                    % 瞬时感染率 lambda_inf = chi * H
                    lam = chi_vec(sus_mask) .* H(sus_mask);
                    p_inf = -expm1(-lam*dt);
                    p_inf = min(max(p_inf,0),1);
                    
                    u2 = rand(sum(sus_mask),1);
                    sus_indices = find(sus_mask);
                    new_infected = sus_indices(u2 < p_inf);
                end
                
                % ========================================================
                % [关键修正 Patch] 更新状态并重置 Tau
                % ========================================================
                if ~isempty(new_infected)
                    state(new_infected) = 1;
                    inf_age(new_infected) = 0;
                    
                    % 遍历所有新感染节点，重置其所有出边的 tau_e
                    % 因为对于这些边来说，传播过程刚刚开始
                    for k = 1:numel(new_infected)
                        v_node = new_infected(k);
                        e_ids = out_edges{v_node}; % 利用预处理的索引，O(1)查找
                        if ~isempty(e_ids)
                            tau_e(e_ids) = 0; 
                        end
                    end
                end
                % ========================================================

                % 4. 恢复事件 (I -> S)
                if any(infected_nodes_mask)
                    idxI = find(infected_nodes_mask);
                    age0 = inf_age(idxI);
                    age1 = age0 + dt;
                    
                    % Weibull 恢复风险
                    dR = (age1./bR).^aR - (age0./bR).^aR;
                    p_rec = -expm1(-dR);
                    
                    u3 = rand(numel(idxI),1);
                    rec_nodes = idxI(u3 < p_rec);
                    
                    % 更新未恢复者的年龄
                    inf_age(idxI) = age1;
                    
                    if ~isempty(rec_nodes)
                        state(rec_nodes) = 0;
                        inf_age(rec_nodes) = 0;
                        % 注意：恢复者的出边 tau_e 不需要在这里重置
                        % 它们会保留数值，但在 active_e 判定时变为 false 从而停止计时
                        % 等到下次该节点再次变 I 时，上面的 Patch 会将其重置为 0
                    end
                end

                % 5. 行为更新 (Asynchronous Game Update)
                due_mask = (next_update_time <= t_end + 1e-12);
                if any(due_mask)
                    rho_global = mean(state);
                    F_risk = 1 / (max(1e-5, 1 - rho_global)); % 风险感知因子
                    
                    idx_due = find(due_mask);
                    
                    % --- 计算局部收益 (Local Payoff) ---
                    % 为了 parfor 效率，这里直接用矩阵运算提取邻居信息
                    % (在循环内访问 slice 变量是允许的)
                    
                    % 1. 获取当前所有节点的策略
                    curr_strat = X_celue;
                    
                    % 2. 统计每个 due 节点的邻居策略
                    % 注意：在 parfor 中 Adj 是广播变量，可以直接使用
                    neigh_A_count = Adj(idx_due, :) * (curr_strat == 1);
                    neigh_total   = deg(idx_due);
                    neigh_N_count = neigh_total - neigh_A_count;
                    
                    % 3. 计算收益
                    % 如果我是 N，收益为:
                    pi_N = uNN * neigh_N_count + uNA * neigh_A_count;
                    % 如果我是 A，收益为:
                    pi_A = (uNA - c) * neigh_N_count + uAA * neigh_A_count;
                    
                    % 4. 费米更新规则 (Fermi-like / Softmax)
                    % 比较两个虚拟收益： U_N vs U_A * F
                    z_N = pi_N; 
                    z_A = pi_A * F_risk; 
                    
                    % 计算选择 A 的概率 P(A) = exp(z_A) / (exp(z_N) + exp(z_A))
                    % 使用 max shift 防止溢出
                    max_z = max(z_N, z_A);
                    exp_N = exp(z_N - max_z);
                    exp_A = exp(z_A - max_z);
                    prob_choose_A = exp_A ./ (exp_N + exp_A + 1e-20);
                    
                    % 5. 执行策略更新
                    r_behav = rand(numel(idx_due), 1);
                    new_strat = double(r_behav < prob_choose_A);
                    
                    X_celue(idx_due) = new_strat;
                    
                    % 6. 安排下一次更新
                    next_update_time(due_mask) = next_update_time(due_mask) + behavior_dt;
                end

                % 记录当前步数据
                rho_run(ti) = mean(state);
                x_run(ti)   = mean(X_celue);
                
                t_now = t_end;
            end
            
            % 累加到总和 (Reduction variable)
            rho_sum_total = rho_sum_total + rho_run;
            x_sum_total   = x_sum_total   + x_run;
        end
        % ==== 并行循环结束 ====
        
        rho_mean_mat(:,ai_idx) = rho_sum_total / run_reps;
        x_mean_mat(:,ai_idx)   = x_sum_total   / run_reps;
    end

    % ---------------- 4. 输出打包 ----------------
    sim_out = struct();
    sim_out.t_series = t_series(:);
    sim_out.aI_list  = aI_list;
    sim_out.rho_mean = rho_mean_mat;
    sim_out.x_mean   = x_mean_mat;
    
    % 保存参数以便复查
    sim_out.params = struct('N',N, 'k_avg',k_avg, 'dt',dt, ...
                            'aR',aR, 'bR',bR, ...
                            'betaA',betaA, 'betaN',betaN, ...
                            'kappaA',kappaA, 'kappaN',kappaN, ...
                            'chiA',chiA, 'chiN',chiN, ...
                            'gamma',gamma_mem, 'reps',run_reps);
end