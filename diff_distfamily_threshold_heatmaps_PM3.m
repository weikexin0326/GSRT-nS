%% diff_distfamily_threshold_heatmaps_PM3.m
% PM3: threshold-distance heatmaps in the (kappa_1, zeta_1) plane
% for three transmission-attempt waiting-time families under the
% median-matched scheme.
%
% Families:
%   1) Weibull        (alpha_W  = 0.8)
%   2) Log-logistic   (alpha_LL = 2.0)
%   3) Lomax          (a_L      = 2.0)
%
% Median-matched rule:
%   The adopter / non-adopter medians are matched to the baseline Weibull
%   setting used in the paper-consistent calculation:
%       alpha_W = 0.8, beta_A = 1.2, beta_N = 1.0
%
% Output:
%   四月实验结果图/diff_distfamily_threshold_heatmaps_PM3.fig
%   四月实验结果图/diff_distfamily_threshold_heatmaps_PM3.jpg

clc; clear; close all;

%% ---------------- 1. Global baseline parameters ----------------
dt    = 0.01;
Tmax  = 20;
k_avg = 10;

% Recovery Weibull (unchanged)
aR = 2.0;
bR = 0.5;

% Baseline attempt family for defining target medians
alphaW_base = 0.8;
betaA_base  = 1.2;   % adopter
betaN_base  = 1.0;   % non-adopter

% Source / receiver baseline for non-adopters (unchanged)
kappa0 = 1.0;
zeta0  = 1.0;

% Baseline point to mark (unchanged)
baseline_kappa1 = 0.5;
baseline_zeta1  = 0.4;

% Scan ranges (unchanged)
kappa1_vec = linspace(0, 1, 121);
zeta1_vec  = linspace(0, 1, 121);

% Fixed contour levels (unchanged)
fixed_levels = [1, 1.5, 2, 3, 4];

% Tail truncation
epsTail = 1e-8;

% Recovery support length
max_tau_rec = bR * (log(1/epsTail))^(1/aR);
L_R = ceil(min(Tmax, max_tau_rec) / dt) + 1;
tauR = (0:L_R-1)' * dt;
Psi_rec = exp(-(tauR ./ bR).^aR);

%% ---------------- 2. PM3 settings (paper-consistent) -----------
PM3.name = 'PM3';
PM3.uAA  = 0.10;
PM3.uNA  = 0.05;
PM3.uNN  = 0.01;
PM3.c    = 0.01;

%% ---------------- 3. Median-matched family settings ------------
% Target medians inherited from the baseline Weibull setting
mA_target = weibull_median(alphaW_base, betaA_base);
mN_target = weibull_median(alphaW_base, betaN_base);

families = struct([]);

% (1) Weibull
families(1).name        = 'Weibull';
families(1).shape       = 0.8;
families(1).betaA       = median_to_beta('weibull',     families(1).shape, mA_target);
families(1).betaN       = median_to_beta('weibull',     families(1).shape, mN_target);
families(1).title_str   = sprintf('Weibull (\\alpha=%.1f)', families(1).shape);
families(1).family_key  = 'weibull';

% (2) Log-logistic
families(2).name        = 'Log-logistic';
families(2).shape       = 2.0;
families(2).betaA       = median_to_beta('loglogistic', families(2).shape, mA_target);
families(2).betaN       = median_to_beta('loglogistic', families(2).shape, mN_target);
families(2).title_str   = sprintf('Log-logistic (\\alpha=%.1f)', families(2).shape);
families(2).family_key  = 'loglogistic';

% (3) Lomax
families(3).name        = 'Lomax';
families(3).shape       = 2.0;
families(3).betaA       = median_to_beta('lomax',       families(3).shape, mA_target);
families(3).betaN       = median_to_beta('lomax',       families(3).shape, mN_target);
families(3).title_str   = sprintf('Lomax (\\alpha=%.1f)', families(3).shape);
families(3).family_key  = 'lomax';

numFam = numel(families);

%% ---------------- 4. Solve x_DFE for PM3 -----------------------
xDFE = solve_x_DFE(k_avg, PM3.uAA, PM3.uNA, PM3.uNN, PM3.c);

%% ---------------- 5. Precompute family-specific kernels --------
[K1, Z1] = meshgrid(kappa1_vec, zeta1_vec);

D_maps = zeros(numel(zeta1_vec), numel(kappa1_vec), numFam);
baseline_D = zeros(numFam,1);
zero_contour_exists = false(numFam,1);
LambdaTildeA = zeros(numFam,1);
LambdaTildeN = zeros(numFam,1);

for f = 1:numFam
    fam = families(f);

    max_tau_att = estimate_tau_max(fam.family_key, fam.shape, max(fam.betaA, fam.betaN), epsTail, Tmax);
    L_I = ceil(min(Tmax, max_tau_att) / dt) + 1;

    etaA = renewal_attempt_kernel_rate_family(L_I, dt, fam.family_key, fam.shape, fam.betaA);
    etaN = renewal_attempt_kernel_rate_family(L_I, dt, fam.family_key, fam.shape, fam.betaN);

    LR_use = min(L_I, L_R);
    LambdaTildeA(f) = sum(etaA(1:LR_use) .* Psi_rec(1:LR_use)) * dt;
    LambdaTildeN(f) = sum(etaN(1:LR_use) .* Psi_rec(1:LR_use)) * dt;

    zeta_mix   = xDFE .* Z1 + (1 - xDFE) .* zeta0;
    Lambda_mix = xDFE .* (K1 .* LambdaTildeA(f)) + (1 - xDFE) .* (kappa0 .* LambdaTildeN(f));

    Dth = zeta_mix .* Lambda_mix .* k_avg - 1;
    D_maps(:,:,f) = Dth;

    baseline_D(f) = (xDFE*baseline_zeta1 + (1-xDFE)*zeta0) * ...
                    (xDFE*(baseline_kappa1*LambdaTildeA(f)) + (1-xDFE)*(kappa0*LambdaTildeN(f))) * ...
                    k_avg - 1;

    zero_contour_exists(f) = (min(Dth(:)) <= 0) && (max(Dth(:)) >= 0);
end

%% ---------------- 6. Shared color settings ---------------------
global_min = min(D_maps(:));
global_max = max(D_maps(:));
shared_cmin = floor(global_min);
shared_cmax = ceil(global_max);
if shared_cmin == shared_cmax
    shared_cmax = shared_cmax + 1;
end
shared_ticks = shared_cmin:1:shared_cmax;
shared_cmap = graywhitered_zeroanchored(256, shared_cmin, shared_cmax);

%% ---------------- 7. Plotting ----------------------------------
fig = figure('Color','w','Position',[80 80 1550 470]);
ax = gobjects(numFam,1);

sgtitle(['PM3: threshold-distance heatmaps in the (\kappa_1,\zeta_1) plane ', ...
         'under median-matched transmission-attempt waiting-time families'], ...
        'FontWeight', 'bold', 'FontSize', 16, 'Interpreter', 'tex');

for f = 1:numFam
    ax(f) = subplot(1,3,f);
    Dth = D_maps(:,:,f);

    imagesc(kappa1_vec, zeta1_vec, Dth);
    set(gca, 'YDir', 'normal');
    hold on;
    caxis([shared_cmin, shared_cmax]);
    colormap(gca, shared_cmap);

    valid_fixed_levels = fixed_levels(fixed_levels > min(Dth(:)) & fixed_levels < max(Dth(:)));
    if ~isempty(valid_fixed_levels)
        [Cfix, hfix] = contour(kappa1_vec, zeta1_vec, Dth, valid_fixed_levels, ...
                               'k--', 'LineWidth', 1.2);
        clabel(Cfix, hfix, ...
               'Color', 'k', ...
               'FontSize', 9, ...
               'BackgroundColor', 'w', ...
               'Margin', 2, ...
               'LabelSpacing', 350, ...
               'Interpreter', 'tex');
    end

    if zero_contour_exists(f)
        contour(kappa1_vec, zeta1_vec, Dth, [0 0], 'k-', 'LineWidth', 2.0);
        txt_head = 'D_{th}=0 contour shown';
    else
        txt_head = 'No D_{th}=0 contour in scanned range';
    end

    plot(baseline_kappa1, baseline_zeta1, 'p', 'MarkerSize', 10, ...
         'MarkerFaceColor', [1 0.9 0], 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);

    annotation_lines = { ...
        txt_head, ...
        sprintf('x_{DFE}=%.4f', xDFE), ...
        sprintf('\\beta_{A}=%.4f, \\beta_{N}=%.4f', families(f).betaA, families(f).betaN), ...
        sprintf('min =%.3f, max =%.3f', min(Dth(:)), max(Dth(:))) ...
    };
    text(0.04, 0.95, annotation_lines, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
         'FontSize', 10, 'BackgroundColor', [1 1 1], 'Margin', 5, ...
         'Interpreter', 'tex');

    text(baseline_kappa1 + 0.03, baseline_zeta1 + 0.03, ...
         { 'baseline', sprintf('D_{th}=%.3f', baseline_D(f)) }, ...
         'FontSize', 10, 'BackgroundColor', [1 1 1], 'Margin', 3, ...
         'Interpreter', 'tex');

    title(families(f).title_str, 'FontWeight', 'normal', 'FontSize', 14, 'Interpreter', 'tex');
    xlabel('\kappa_1', 'Interpreter', 'tex');
    ylabel('\zeta_1', 'Interpreter', 'tex');
    set(gca, 'TickLabelInterpreter', 'tex');
    grid on;
    box on;
end

% Compatible colorbar placement for different MATLAB versions
cb = colorbar;
try
    cb.Position = [0.92 0.16 0.015 0.68];
catch
end
cb.Ticks = shared_ticks;
cb.FontSize = 11;
ylabel(cb, 'D_{th}=\zeta(x_{DFE})\Lambda(x_{DFE}\mid\kappa_1)\langle k\rangle-1', ...
       'Interpreter', 'tex', 'FontSize', 13);

%% ---------------- 8. Save outputs ------------------------------
outdir = '四月实验结果图';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

saveas(fig, fullfile(outdir, 'diff_distfamily_threshold_heatmaps_PM3.fig'));
try
    exportgraphics(fig, fullfile(outdir, 'diff_distfamily_threshold_heatmaps_PM3.jpg'), 'Resolution', 300);
catch
    saveas(fig, fullfile(outdir, 'diff_distfamily_threshold_heatmaps_PM3.jpg'));
end

%% ---------------- 9. Workspace results -------------------------
threshold_PM3_distfamily_results = struct();
threshold_PM3_distfamily_results.dt = dt;
threshold_PM3_distfamily_results.Tmax = Tmax;
threshold_PM3_distfamily_results.k_avg = k_avg;
threshold_PM3_distfamily_results.aR = aR;
threshold_PM3_distfamily_results.bR = bR;
threshold_PM3_distfamily_results.xDFE = xDFE;
threshold_PM3_distfamily_results.PM3 = PM3;
threshold_PM3_distfamily_results.kappa0 = kappa0;
threshold_PM3_distfamily_results.zeta0 = zeta0;
threshold_PM3_distfamily_results.kappa1_vec = kappa1_vec;
threshold_PM3_distfamily_results.zeta1_vec = zeta1_vec;
threshold_PM3_distfamily_results.families = families;
threshold_PM3_distfamily_results.LambdaTildeA = LambdaTildeA;
threshold_PM3_distfamily_results.LambdaTildeN = LambdaTildeN;
threshold_PM3_distfamily_results.D_maps = D_maps;
threshold_PM3_distfamily_results.baseline_point = [baseline_kappa1, baseline_zeta1];
threshold_PM3_distfamily_results.baseline_D = baseline_D;
threshold_PM3_distfamily_results.fixed_levels = fixed_levels;
threshold_PM3_distfamily_results.shared_cmin = shared_cmin;
threshold_PM3_distfamily_results.shared_cmax = shared_cmax;
threshold_PM3_distfamily_results.shared_ticks = shared_ticks;

disp('Done. Figures saved:');
disp('  四月实验结果图/diff_distfamily_threshold_heatmaps_PM3.fig');
disp('  四月实验结果图/diff_distfamily_threshold_heatmaps_PM3.jpg');

%% ==================== Local functions ==========================
function x = solve_x_DFE(k_avg, uAA, uNA, uNN, c)
    Gamma0 = (uNA - c) - uNN;
    Gamma1 = (uAA - uNA + c) - (uNA - uNN);
    fun = @(x) log(x./(1-x)) - k_avg .* (Gamma0 + Gamma1 .* x);

    lo = 1e-8;
    hi = 1 - 1e-8;
    flo = fun(lo);
    fhi = fun(hi);

    if sign(flo) ~= sign(fhi)
        x = fzero(fun, [lo, hi]);
        x = min(max(x, lo), hi);
        return;
    end

    obj = @(x) abs(fun(x));
    x = fminbnd(obj, lo, hi);
    x = min(max(x, lo), hi);
end

function med = weibull_median(alpha, beta)
    med = beta * (log(2))^(1/alpha);
end

function beta = median_to_beta(family_key, shape, med)
    switch lower(family_key)
        case 'weibull'
            beta = med / (log(2))^(1/shape);
        case 'loglogistic'
            beta = med;
        case 'lomax'
            beta = med / (2^(1/shape) - 1);
        otherwise
            error('Unknown family: %s', family_key);
    end
end

function tau_max = estimate_tau_max(family_key, shape, beta, epsTail, Tmax)
    switch lower(family_key)
        case 'weibull'
            tau_max = beta * (log(1/epsTail))^(1/shape);
        case 'loglogistic'
            tau_max = beta * ((1/epsTail - 1))^(1/shape);
        case 'lomax'
            tau_max = beta * (epsTail^(-1/shape) - 1);
        otherwise
            error('Unknown family: %s', family_key);
    end
    tau_max = min(tau_max, Tmax);
end

function eta_rate = renewal_attempt_kernel_rate_family(L, dt, family_key, shape, beta)
    tau0 = (0:L-1)' * dt;
    tau1 = tau0 + dt;

    surv0 = survival_family(tau0, family_key, shape, beta);
    surv1 = survival_family(tau1, family_key, shape, beta);
    q_mass = max(surv0 - surv1, 0);

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

function S = survival_family(tau, family_key, shape, beta)
    switch lower(family_key)
        case 'weibull'
            S = exp(- (tau./beta).^shape );
        case 'loglogistic'
            S = 1 ./ (1 + (tau./beta).^shape);
        case 'lomax'
            S = (1 + tau./beta).^(-shape);
        otherwise
            error('Unknown family: %s', family_key);
    end
end

function cmap = graywhitered_zeroanchored(m, vmin, vmax)
    if nargin < 1, m = 256; end

    if vmax <= 0
        cmap = graywhite_local(m);
        return;
    elseif vmin >= 0
        cmap = whitered_local(m);
        return;
    end

    frac_zero = abs(vmin) / (vmax - vmin);
    nneg = max(2, round(m * frac_zero));
    npos = max(2, m - nneg + 1);

    gray_part = [linspace(0.72,1,nneg)', linspace(0.72,1,nneg)', linspace(0.72,1,nneg)'];
    red_part  = [ones(npos,1), linspace(1,0,npos)', linspace(1,0,npos)'];
    cmap = [gray_part; red_part(2:end,:)];

    if size(cmap,1) > m
        cmap = cmap(1:m,:);
    elseif size(cmap,1) < m
        cmap = [cmap; repmat(cmap(end,:), m-size(cmap,1), 1)];
    end
end

function cmap = whitered_local(m)
    if nargin < 1, m = 256; end
    cmap = [ones(m,1), linspace(1,0,m)', linspace(1,0,m)'];
end

function cmap = graywhite_local(m)
    if nargin < 1, m = 256; end
    cmap = [linspace(0.72,1,m)', linspace(0.72,1,m)', linspace(0.72,1,m)'];
end