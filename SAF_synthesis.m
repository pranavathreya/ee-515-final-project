function y = SAF_synthesis(subbands, lpf, K, N_orig)

[K_check, L_sub] = size(subbands);
assert(K_check == K, 'K mismatch between analysis and synthesis.');

lpf = lpf(:);
Lh  = length(lpf);

% Use the *centered* index like analysis
n = (0:Lh-1).' - (Lh-1)/2;

% We'll upsample into an N_up grid matching original length
N_up   = N_orig;
y_acc  = zeros(N_up + Lh - 1, 1);  % for full convolution sums

for k = 0:K-1
    % Synthesis filter: conjugate of analysis filter (approx PR)
    hk = lpf .* exp(-1j * 2*pi*(k/K).*n);   % analysis filter
    gk = conj(hk);                           % synthesis filter

    % Upsample this subband into the full-rate grid
    up = zeros(N_up, 1);
    idx = k+1 : K : (k+1 + (L_sub-1)*K);   % where to place subband samples
    idx(idx > N_up) = [];                  % clip to available length
    up(idx) = subbands(k+1, 1:numel(idx)).';

    % Filter and accumulate
    y_acc = y_acc + conv(up, gk, 'full');
end

% Extract center N_orig samples (to undo FIR delay)
start = floor((Lh-1)/2) + 1;
y_rec = y_acc(start : start + N_orig - 1);

% Return real part (imag should be numerical noise)
y = real(y_rec);

end

