%% Simple tune + noise + adaptive filtering (LMS)
% Clean, noisy, and filtered playback included.

clear; clc; close all;
rng(42);                    % Reproducibility

%% Global audio settings
fs   = 44100;              % Sample rate (Hz)
bpm  = 120;                % Tempo
qDur = 60/bpm;             % Quarter-note duration (s)

% Small fade to avoid clicks at note boundaries
fade_ms = 10; 
function y = tone(freq, dur, fs, fade_ms)
    t = (0:round(dur*fs)-1)/fs;
    y = sin(2*pi*freq*t);                       % Sine tone
    % Simple ADSR-style fade-in/out
    Nfade = round(fade_ms/1000*fs);
    env   = ones(size(y));
    if Nfade>0 && 2*Nfade < numel(y)
        env(1:Nfade)              = linspace(0,1,Nfade);
        env(end-Nfade+1:end)      = linspace(1,0,Nfade);
    end
    y = y .* env;
end

%% Note frequencies (C major, 4th octave)
C4=261.63; D4=293.66; E4=329.63; F4=349.23; G4=392.00; A4=440.00; B4=493.88;

% "Mary Had a Little Lamb" (common version, key of C)
% Phrase:  E D C D | E E E (half) | D D D (half) | E G G (half) |
%          E D C D | E E E (E half) | D D E D | C (whole)
notes = [E4 D4 C4 D4  E4 E4 E4   D4 D4 D4    E4 G4 G4 ...
         E4 D4 C4 D4  E4 E4 E4             D4 D4 E4 D4  C4];

% Matching durations (in quarter-note units)
% quarters by default; halves = 2, whole = 4
dursQ = [1  1  1  1   1  1  2    1  1  2     1  1  2 ...
         1  1  1  1   1  1  2               1  1  1  1  4];

%% Synthesize the melody
melody = [];
for k = 1:numel(notes)
    dur = dursQ(k)*qDur;
    melody = [melody, tone(notes(k), dur, fs, fade_ms)]; %#ok<AGROW>
end
melody = melody(:);

%% Create a two-tone noise signal
% We'll simulate a realistic scenario:
%   - A reference noise signal v (two sinusoids combined)
%   - Room/propagation shapes it (unknown FIR b), creating n = filter(b,1,v)
%   - The listener's "primary mic" hears 'd = melody + n'
% The adaptive filter gets reference v to cancel n from d.

N = numel(melody);
t = (0:N-1)'/fs;

% Define two noise tones (you can experiment with these frequencies)
f1 = 500;   % Hz
f2 = 800;   % Hz

% Reference noise: sum of two pure tones
v = sin(2*pi*f1*t);% + sin(2*pi*f2*t);

% Optional: small random phase offsets for realism
% v = sin(2*pi*f1*t + rand*2*pi) + sin(2*pi*f2*t + rand*2*pi);

% Normalize to prevent clipping
v = v / max(abs(v));

% Unknown path from noise source to primary mic
b_path  = fir1(32, 0.35);    % Arbitrary FIR "acoustic path"
n = filter(b_path, 1, v);

% Primary mic (target): clean melody + scaled noise
d = melody + 0.3*n;

% For listening/demo purposes, also keep an isolated noise track
noise_alone = 0.3*n;

%% Playbacks (turn your volume down just in case)
disp('Playing clean melody...');         soundsc(melody, fs);      pause(numel(melody)/fs + 0.5);
disp('Playing noise only...');           soundsc(noise_alone, fs); pause(numel(noise_alone)/fs + 0.5);
disp('Playing noisy mixture (melody+noise)...'); 
soundsc(d, fs);                          pause(numel(d)/fs + 0.5);

%% Adaptive noise canceller (LMS)
% Goal: estimate the noise portion inside d using v_col as the reference,
% then subtract the estimate from d.

M  = 32;                 % Adaptive filter length (tune as needed)
mu = 0.01;               % Step size (smaller -> slower, safer)
w  = zeros(M,1);         % Adaptive filter taps

u = v;               % Reference input to the adaptive filter
y = zeros(N,1);          % Estimated noise
e = zeros(N,1);          % Error (should approach the clean melody)

% LMS loop
for nIdx = M:N
    uvec = flipud(u(nIdx-M+1:nIdx));     % Reference window
    y(nIdx) = w.'*uvec;                  % Noise estimate
    e(nIdx) = d(nIdx) - y(nIdx);         % Error (desired - estimate)
    % Normalized LMS (optional): uncomment two lines & comment plain update
    % norm_u = (uvec.'*uvec) + 1e-6;
    % w = w + (mu/norm_u)*uvec*e(nIdx);

    % Plain LMS update:
    w = w + mu*uvec*e(nIdx);
end

clean_est = e;   % After convergence, e ≈ melody

%% Play filtered output
disp('Playing LMS-filtered output (noise reduced)...');
soundsc(clean_est, fs);   % Should sound closer to the original melody

%% (Optional) Quick visual check
t = (0:N-1)/fs;
figure('Color','w');
subplot(3,1,1); plot(t, melody); title('Clean Melody'); xlabel('Time (s)'); ylabel('Amp');
subplot(3,1,2); plot(t, d);      title('Noisy Mixture (Melody + Noise)'); xlabel('Time (s)'); ylabel('Amp');
subplot(3,1,3); plot(t, clean_est); title('Adaptive LMS Output (Noise Reduced)'); xlabel('Time (s)'); ylabel('Amp');

% You can also listen back-to-back for comparison:
% soundsc([melody; zeros(fs/2,1); d; zeros(fs/2,1); clean_est], fs);

%% Frequency-domain analysis and visualization
Nfft = 2^nextpow2(N);
f = fs*(0:(Nfft/2))/Nfft;

% Compute single-sided magnitude spectra
MEL_spec = abs(fft(melody, Nfft)/N);
NOISY_spec = abs(fft(d, Nfft)/N);
CLEAN_spec = abs(fft(clean_est, Nfft)/N);
NOISE_spec = abs(fft(noise_alone, Nfft)/N);

MEL_spec = MEL_spec(1:Nfft/2+1);
NOISY_spec = NOISY_spec(1:Nfft/2+1);
CLEAN_spec = CLEAN_spec(1:Nfft/2+1);
NOISE_spec = NOISE_spec(1:Nfft/2+1);

figure('Color','w');
plot(f,20*log10(MEL_spec+eps),'b','LineWidth',1.2); hold on;
plot(f,20*log10(NOISY_spec+eps),'r','LineWidth',1.2);
plot(f,20*log10(CLEAN_spec+eps),'g','LineWidth',1.2);
plot(f,20*log10(NOISE_spec+eps),'y','LineWidth',1.2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('Clean Melody','Noisy Mixture','Filtered Output','Noise');
title('Frequency Spectra Comparison');
xlim([0 5000]);  % The melody energy is mostly below 5 kHz

