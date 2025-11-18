function [subbands, lpf] = SAF_subband(signal_in, Fs, K)
% SAF_SUBBAND  Decompose a single audio signal into K uniform subbands.
%
% Inputs:
%   signal_in  - input time-domain signal (vector)
%   Fs         - sampling frequency (Hz)
%   K          - number of subbands (e.g., 16, 32, 64)
%
% Outputs:
%   subbands   - K x ~N/K complex subband signals
%   lpf        - prototype low-pass filter used for the filterbank
%
% Notes:
%   - signal_in is assumed to be 1-D (mono)
%   - subbands(k,:) represents band k (0 = lowest freq band)
%   - perfect for SAF, LMS, and reconstruction pipelines

%% 1. Ensure column vector input
signal_in = signal_in(:);
N = length(signal_in);

fprintf('SAF: Analyzing %d samples at %d Hz into %d subbands...\n', N, Fs, K);

%% 2. Design prototype filter (length 4K recommendation)
L = 64*K;
cutoff = 1/K;
lpf = fir1(L, cutoff, hamming(L+1));       % normalized cutoff = 1/K
lpf = lpf(:);

Lh = length(lpf);
n  = (0:Lh-1).' - (Lh-1)/2;   % centered time index for modulation

 %% 3. Preallocate output matrix
 subb_length  = ceil((N + Lh) / K);    % safe upper bound
 subbands = zeros(K, subb_length);

 %% 4. Analysis filterbank loop
 for k = 0:K-1

     % Modulated analysis filter for subband k
     hk = lpf .* exp(-1j * 2*pi*(k/K)*n);
     hk = hk(:);

     % Filter input with subband filter
     y_full = conv(signal_in, hk, 'full');

     % Extract central N samples to align with input
    start = floor((Lh-1)/2) + 1;
    y     = y_full(Lh : Lh+N-1);  % length N

    % Polyphase-consistent decimation: phase = k+1
    idx   = k+1 : K : N;        % indices for this subband
    sb    = y(idx);             % decimated sequence

    % Store into row k+1, pad with zeros if needed
    Lk = numel(sb);
    subbands(k+1,1:Lk) = sb;
    if Lk < subb_length
        subbands(k+1,Lk+1:end) = 0;
    end

 end


fprintf('SAF complete. Each subband length ≈ %d samples.\n', subb_length);

end
