clear; close all; clc;

filename = 'recording_8ch_nc.wav';
info = audioinfo(filename);
fs = info.SampleRate;

% ---- rough time where you think the clap is ----
startT = 1.8;
endT = 1.9;

searchStartSample = floor(startT * fs) + 1;
searchEndSample   = floor(endT * fs);

[xSearch, ~] = audioread(filename, [searchStartSample, searchEndSample]);

M = 6; % using ch1..ch6
xSearch = xSearch(:,1:M);

% ============================================================
% Find clap peak index in this chunk
% env(n) = max abs amplitude across channels at sample n
% ============================================================
env = max(abs(xSearch), [], 2);
[~, kPeak] = max(env);          % clap peak

% ---- choose a frame centered on the clap peak ----
frameLen = 512;  % samples
frameLen = min(frameLen, size(xSearch,1));  % safety

startK = kPeak - floor(frameLen/2);
startK = max(1, startK);
endK   = min(size(xSearch,1), startK + frameLen - 1);

% if we hit the end, shift start to keep frameLen if possible
startK = max(1, endK - frameLen + 1);

x = xSearch(startK:endK, :);

% --------------------------
% Pair distances (meters)
% --------------------------
d = containers.Map;

d("1-2") = 35e-3;
d("1-3") = 60e-3;
d("1-4") = 69.462e-3;
d("1-5") = 60e-3;
d("1-6") = 35e-3;

d("2-3") = 35e-3;
d("2-4") = 60e-3;
d("2-5") = 69.462e-3;
d("2-6") = 60e-3;

d("3-4") = 35e-3;
d("3-5") = 60e-3;
d("3-6") = 69.462e-3;

d("4-5") = 35e-3;
d("4-6") = 60e-3;

d("5-6") = 35e-3;

c = 343;  % speed of sound (m/s)

% --------------------------
% Pre-process (important)
% --------------------------
% Remove DC offset per channel + apply Hann window to reduce edge artifacts
x = x - mean(x,1);
x = x .* hann(size(x,1));

% --------------------------
% GCC-PHAT for all pairs
% --------------------------
pairs = nchoosek(1:M, 2);
numPairs = size(pairs,1);

pairStr = strings(numPairs,1);
dist_mm = zeros(numPairs,1);
maxLag_samp = zeros(numPairs,1);
tau_s = zeros(numPairs,1);
tau_samp = zeros(numPairs,1);
peakRatio = zeros(numPairs,1);  % reliability metric

for k = 1:numPairs
    i = pairs(k,1);
    j = pairs(k,2);

    key = sprintf("%d-%d", i, j);
    dij = d(key);  % meters

    dist_mm(k) = dij * 1e3;

    maxLag = ceil((dij / c) * fs);    % samples
    maxLag_samp(k) = maxLag;

    % GCC-PHAT correlation
    [~, R, lag] = gccphat(x(:,i), x(:,j), fs);  % lag in seconds
    lagSamp = lag * fs;

    % Restrict search to physically possible lags
    mask = abs(lagSamp) <= maxLag;
    A = abs(R);

    % pick best peak within allowed region
    [pk, idxLocal] = max(A(mask));
    lagAllowed = lag(mask);
    tauBest = lagAllowed(idxLocal);

    tau_s(k) = tauBest;
    tau_samp(k) = tauBest * fs;

    % simple reliability: peak / mean inside allowed region
    peakRatio(k) = pk / (mean(A(mask)) + 1e-12);

    pairStr(k) = sprintf("ch%d-ch%d", i, j);
end

T = table(pairStr, dist_mm, maxLag_samp, tau_s, tau_samp, peakRatio, ...
    'VariableNames', {'Pair','Distance_mm','MaxLag_samp','Tau_s','Tau_samples','PeakRatio'});

T = sortrows(T, 'PeakRatio', 'descend');
disp(T);


% --- mic geometry (regular hexagon) ---
% Use your "opposite" distance to set radius. For a hexagon:
% opposite distance ≈ 2R  -> R ≈ 69.462mm / 2
dia = sqrt((60e-3)^2+(35e-3)^2);
R = dia / 2;  % meters

ang = deg2rad([0 60 120 180 240 300]);   % ch1..ch6 around the circle
micPos = [R*cos(ang(:)), R*sin(ang(:))]; % 6x2

% Build per-pair delta position matrix A, and measured tau vector
K = size(pairs,1);
A = zeros(K,2);
for k = 1:K
    i = pairs(k,1); j = pairs(k,2);
    A(k,:) = micPos(i,:) - micPos(j,:); % dp = p_i - p_j
end

tau_meas = tau_s(:);   % seconds, from your GCC-PHAT peak (already lag-limited)
w = peakRatio(:);      % reliability weights (bigger = more confident)
w = w / max(w);        % normalize

% --- scan angles and pick best ---
thetaGrid = 0:0.5:359.5;          % deg (0.5° step)
err = zeros(size(thetaGrid));

for t = 1:numel(thetaGrid)
    th = deg2rad(thetaGrid(t));
    u = [cos(th); sin(th)];       % 2x1
    tau_pred = (A*u) / c;         % Kx1
    e = tau_meas - tau_pred;
    err(t) = sum( (w .* e).^2 );  % weighted SSE
end

[errMin, idxBest] = min(err);
thetaBest = thetaGrid(idxBest);
thetaBest = mod(thetaBest - 150, 360);
% "range" around the best angle: pick angles close to min error
% (You can tighten/loosen this threshold)
thresh = errMin * 1.20;  % within 20% of best error
ok = err <= thresh;

% Find contiguous region(s) around the best angle. Handle wrap-around crudely:
thetaOk = thetaGrid(ok);

fprintf("Estimated DOA (scan) = %.1f deg\n", thetaBest);

figure;
plot(thetaGrid, err); grid on;
xlabel("Angle (deg)"); ylabel("Weighted error");
title(sprintf("DOA = %.1f deg (lower is better)", thetaBest));
