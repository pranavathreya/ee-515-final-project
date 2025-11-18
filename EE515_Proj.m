clear; clc; close all;

%% FFmpeg tester
ensureFFmpegReady;

%% Audio / Video exporter
function [audioFile, videoFile, fs, videoInfo] = splitAudioVideo(inputMov, audioOutName)
% splitAudioVideo.m
% ---------------------------
% Uses FFmpeg to extract the audio from a video file, saves it as an MP3,
% and creates an audio-less version of the video.
%
% INPUTS:
%   inputMov     - string, path to the input video file (.mov, .mp4, etc.)
%   audioOutName - string, desired output audio name (e.g., 'clean_audio.mp3')
%
% OUTPUTS:
%   audioFile    - name of the exported mp3 file
%   videoFile    - name of the silent video file
%   fs           - audio sampling rate (Hz)
%   videoInfo    - structure with video properties
%
% Example:
%   [aFile, vFile, fs, vInfo] = splitAudioVideo('input.mov', 'original_audio.mp3');

    % --- Check inputs ---
    if nargin < 2
        error('Usage: splitAudioVideo(inputMov, audioOutName)');
    end

    % --- Verify FFmpeg availability ---
    [status, ~] = system('ffmpeg -version');
    if status ~= 0
        error(['FFmpeg not found on system path. ' ...
               'Please install FFmpeg and ensure it''s accessible via command line.']);
    end

    % --- Derive base name for silent video ---
    [pathStr, baseName, ~] = fileparts(inputMov);
    videoFile = fullfile(pathStr, [baseName '_SilentVid.mp4']);

    % --- Extract Audio to MP3 ---
    fprintf('🎧 Extracting audio using FFmpeg...\n');
    cmdAudio = sprintf('ffmpeg -y -i "%s" -q:a 0 -map a "%s"', inputMov, audioOutName);
    [statusA, msgA] = system(cmdAudio);
    if statusA ~= 0
        error('Audio extraction failed:\n%s', msgA);
    end
    audioFile = audioOutName;

    % --- Extract Silent Video ---
    fprintf('🎥 Creating silent video...\n');
    cmdVideo = sprintf('ffmpeg -y -i "%s" -an -c:v copy "%s"', inputMov, videoFile);
    [statusV, msgV] = system(cmdVideo);
    if statusV ~= 0
        error('Silent video extraction failed:\n%s', msgV);
    end

    % --- Read Audio Metadata ---
    try
        infoAudio = audioinfo(audioOutName);
        fs = infoAudio.SampleRate;
    catch
        warning('Unable to read audio sampling frequency; setting fs = NaN.');
        fs = NaN;
    end

    % --- Read Video Metadata (optional) ---
    try
        v = VideoReader(inputMov);
        videoInfo = struct('FrameRate', v.FrameRate, ...
                           'Duration', v.Duration, ...
                           'Width', v.Width, ...
                           'Height', v.Height);
    catch
        % If VideoReader fails (e.g., codec not supported), return minimal info
        warning('VideoReader could not open file; video metadata limited.');
        videoInfo = struct('FrameRate', NaN, 'Duration', NaN, ...
                           'Width', NaN, 'Height', NaN);
    end

    % --- Print Summary ---
    fprintf('✅ Audio saved as %s (%.1f kHz)\n', audioOutName, fs/1000);
    fprintf('✅ Silent video saved as %s\n', videoFile);
    fprintf('Done.\n');
end


%% === USER INPUTS ===

windyVid = 'C:\Users\Precision\Desktop\EE 515\Project\Images\IMG_0220.MOV';
% WindyAudio = 'windyAudio.mp3';
% 
% % Call the function
% % [audioFile, videoFile, fs, vidInfo] = splitAudioVideo(windyVid, WindyAudio);
% 
% % Rename silent video to your preferred name
% movefile(videoFile, 'SilentVid.mp4');
% 
% disp('All files created successfully!');

%% Generating Audio signal + Noise
rng(42);                    % Reproducibility

% Global audio settings
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

% Note frequencies (C major, 4th octave)
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

% Synthesize the melody
melody = [];
for k = 1:numel(notes)
    dur = dursQ(k)*qDur;
    melody = [melody, tone(notes(k), dur, fs, fade_ms)]; %#ok<AGROW>
end
melody = melody(:);

% Create a two-tone noise signal
N = numel(melody);
t = (0:N-1)'/fs;

% Define two noise tones (you can experiment with these frequencies)
f1 = 600;   % Hz
f2 = 800;   % Hz

% Reference noise: sum of two pure tones
v = sin(2*pi*f1*t) + sin(2*pi*f2*t);

% Optional: small random phase offsets for realism
% v = sin(2*pi*f1*t + rand*2*pi) + sin(2*pi*f2*t + rand*2*pi);

% Normalize to prevent clipping
v = v / max(abs(v));

% Unknown path from noise source to primary mic
b_path  = fir1(32, 0.35);    % Arbitrary FIR "acoustic path"
n = filter(b_path, 1, v);

% Primary mic (target): clean melody + scaled noise
d = melody + 0.5*v;

% For listening/demo purposes, also keep an isolated noise track
noise_alone = 0.5*n;


% % Simpler Signal + noise for filtering:
% fs = 8000;            % lower Fs for easy plotting
% N  = 8000;            % 1 second
% t  = (0:N-1)'/fs;
% 
% % "Signal": low-frequency content only (<= 400 Hz)
% sig = 0.7*sin(2*pi*150*t) + 0.5*sin(2*pi*300*t);
% melody = sig;
% 
% % "Noise": a single high-frequency tone (e.g., 1800 Hz)
% f_noise = 800;
% v       = sin(2*pi*f_noise*t);   % reference noise
% 
% % Primary (noisy) signal
% d = sig + 0.5*v;


%% Spectrum of input signals
fmax = 2000; Nfft = 4096;

[Pd, f] = pwelch(d, hanning(1024), 512, Nfft, fs);
[Pv, ~] = pwelch(v, hanning(1024), 512, Nfft, fs);

figure; hold on; grid on;
plot(f, 10*log10(Pd+eps), 'LineWidth', 1.3);
plot(f, 10*log10(Pv+eps), 'LineWidth', 1.3);
xlim([0 fmax])

xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
title('PSD of Signal(s) using Welch');
legend('Input Signal + Noise', 'Noise')

%% Processing videos and audio files
NoisySong = 'C:\Users\Precision\Desktop\EE 515\Project\Recordings\NoisySong.MOV';
NoiseVid = 'C:\Users\Precision\Desktop\EE 515\Project\Recordings\NoiseVid.MOV';

[Song, Fs_song] = audioread(NoisySong);
Song = mean(Song, 2);
[Noise, Fs_noise] = audioread(NoiseVid);
Noise = mean(Noise, 2);
[WindyV, Fs_wind] = audioread(windyVid);
WindyV = mean(WindyV, 2);

[Song, Noise, delay] = AlignSignals(Song,Noise);

%% Subband Algorithm pt 1

K = 4*4;
% Split signals into subbands
% [subbands_Noisy, lpf] = SAF_subband(Song, Fs_song, K);
% subbands_Noise = SAF_subband(Noise, Fs_song, K);

[subbands_Primary, lpf_primary] = SAF_subband(d, fs, K);
subbands_n = SAF_subband(v, fs, K);
subbands_melody = SAF_subband(melody, fs, K);


Fs_sub = fs / K;
figure;
tiledlayout(ceil(K/2), 2);

for k = 1:K
    nexttile; hold on; grid on;

    if k == 1
        [Pdk, fk] = pwelch(subbands_Primary(k,:), ...
            hanning(1024), 512, 2048, Fs_sub, 'onesided');

        [Pvk, ~] = pwelch(subbands_n(k,:), ...
            hanning(1024), 512, 2048, Fs_sub, 'onesided');

        fk_full = fk;   % stays in 0–Fs_sub/2

    else
        [Pdk_full, fk_fullband] = pwelch(subbands_Primary(k,:), hanning(1024), 512, 2048, Fs_sub, 'twosided');

        [Pvk_full, ~] = pwelch(subbands_n(k,:), hanning(1024), 512, 2048, Fs_sub, 'twosided');

        half_idx = floor(length(fk_fullband)/2)+1;
        fk = fk_fullband(half_idx:end);       % positive freqs only
        Pdk = Pdk_full(half_idx:end);
        Pvk = Pvk_full(half_idx:end);

        fk_full = fk + (k-2)*Fs_sub/2;
    end

    % max(fk_full), min(fk_full)
    plot(fk_full, 10*log10(Pdk+eps),'b','LineWidth',1.3);
    plot(fk_full, 10*log10(Pvk+eps),'r','LineWidth',1.3);

    title(sprintf('Subband %d: Primary vs Noise', k));
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB/Hz)');
    ylim([-180 0]);
    legend('Primary','Noise');
end

%% Adaptive Filtering
% Energy of noise mainly in 1st subband -> Need to filter subband 1
n = 1;
M = 64*n; mu = 0.01/n/2;

% % Real recording example
% [e, y, w] = adaptiveLMS_complex(subbands_Noise(1,:).', subbands_Noisy(1,:).', mu, M);

% subbands_proc = subbands_Noisy;
% subbands_proc(1,:) = e;
% clean_audio = SAF_synthesis(subbands_proc, lpf, K, length(Song));


% Recreating melody with frozen coefficients
[d1, x1] = deal(subbands_Primary(1,:).', subbands_n(1,:).');
[e_train, y_train, w_final, W_train] = adaptiveLMS_complex(x1, d1, mu, M);

% Apply the learned FIR filter with frozen coefficients
y_fixed = filter(flipud(w_final), 1, x1);  % noise estimate using fixed, optimal weights
e_fixed = d1 - y_fixed;            % CLEANED subband (no transient / no jitter)


% Using NLMS Filter
% M = 64; mu = 0.8; delta = 1e-3;
% [e_train2, y_train, W] = adaptiveNLMS_complex(x1, d1, mu, M, delta);
% 
% N0   = 5*M;           % throw away ~5 filter lengths of transient
% e_ss = e_train2;
% 
% if N0 < length(e_ss)
%     e_ss(1:N0) = e_ss(N0);    % hold value, or do a short fade-in if you prefer
% end
% 
% subbands_prim = subbands_Primary;
% subbands_prim(1, :) = e_ss.';    % replace only that subband with steady-state output

% 
% subbands_prim = subbands_Primary;
% subbands_prim(1,1:length(e_train2)) = e_train2.';  % replace only first subband

% clean_melody = SAF_synthesis(subbands_prim, lpf_primary, K, length(melody));



[e_prim, y_prim, w_prim] = adaptiveLMS_complex(subbands_n(1,:).',subbands_Primary(1,:).', mu, M);
subbands_prim = subbands_Primary;
subbands_prim(1,:) = e_prim;
clean_melody = SAF_synthesis(subbands_prim, lpf_primary, K, length(melody));

% soundsc(Song, Fs_song)
% pause(length(Song)/Fs_song + 0.1);
% soundsc(clean_audio, Fs_song);

%% Playing Audio Recordings of signals
cleansig = melody;
noisysig = d;
clean_melody = clean_melody * max(cleansig) / max(clean_melody);

T = 10;                   % seconds
Nsel = T*fs;              % sample count for 10 seconds

cleansig_s     = cleansig(1:min(Nsel,length(cleansig)));
noisysig_s     = noisysig(1:min(Nsel,length(noisysig)))/max(noisysig);
clean_melody_s = clean_melody(1:min(Nsel,length(clean_melody)));

% soundsc(cleansig, fs);
% pause(length(cleansig)/fs + 0.1)
% soundsc(d, fs)
% pause(length(melody)/fs + 0.1);
% soundsc(clean_melody, fs)

silence = zeros(round(0.5*fs),1);  % 0.5 sec pause
T = 10; 

combo = [cleansig_s; silence; noisysig_s; silence; clean_melody];

audiowrite('comparison_all.mp3', combo, fs);

%% FUll Adaptive Filtering

M_full  = M;          % filter length (can match SAF M or be larger)
mu_full = mu;        % step size (tune so it converges but doesn’t blow up)

% Traditional full-band ANC: x = reference noise (v), d = primary (sig + noise)
[e_full, y_full, w_full] = adaptiveLMS_complex(v, d, mu_full, M_full);

% e_full is your "cleaned" **full-band** signal from traditional LMS
clean_full = e_full;   % rename for clarity

%% Analysis of Filtered and Un-Filtered Results
d1 = subbands_Primary(1,:);   % Noisy subband before filtering
x1 = subbands_n(1,:);         % Noise subband
e1 = subbands_prim(1,:);      % "Cleaned" subband after filtering
mel1 = subbands_melody(1,:);  % Original subband of melody

Fs_sub = fs / K;

% 1st subband spectrum plots
figure;
sgtitle('1st Subband Signals Before/After SAF')
subplot(2,2,1);
pwelch(mel1, hanning(1024), 512, 2048, Fs_sub);
title('Clean Signal');
ylabel('PSD (dB/Hz)');
ylim([-120 0])

subplot(2,2,2);
pwelch(x1, hanning(1024), 512, 2048, Fs_sub);
title('Reference Noise');
ylabel('PSD (dB/Hz)');
ylim([-120 0])

subplot(2,2,3);
pwelch(d1, hanning(1024), 512, 2048, Fs_sub);
title('Noisy Signal, (Before SAF)');
ylabel('PSD (dB/Hz)');
ylim([-120 0])

subplot(2,2,4);
pwelch(e1, hanning(1024), 512, 2048, Fs_sub);
title('Cleaned Signal, (After SAF)');
ylabel('PSD (dB/Hz)');
xlabel('Frequency (Hz)');
ylim([-120 0])

% 1st subband before vs after
[Px,f] = pwelch(d1, hanning(1024), 512, 2048, Fs_sub);
[Pe,~] = pwelch(e1, hanning(1024), 512, 2048, Fs_sub);
[Pn,~] = pwelch(x1, hanning(1024), 512, 2048, Fs_sub);

figure; hold on; grid on;
plot(f, 10*log10(Px+eps), 'LineWidth', 1.2);
plot(f, 10*log10(Pe+eps), 'LineWidth', 1.2);
% plot(f, 10*log10(Pn+eps), 'LineWidth', 1.2);
legend('Before LMS (d1)', 'After LMS (e1)', 'Noise (x1)');
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
title('Subband 1: Before vs After LMS');

E_d  = sum(abs(d1).^2);
E_x  = sum(abs(x1).^2);
% crude “noise power” proxy:
E_noise_like = abs(sum(conj(x1).*d1))^2 / (sum(abs(x1).^2) + eps);

% Full spectrum reg Adap vs SAF comp
Nfft = 4096;

[Pd_noisy, f_spec]         = pwelch(d,            hanning(1024), 512, Nfft, fs);
[Pfull_clean, ~]      = pwelch(clean_full,   hanning(1024), 512, Nfft, fs);
[Psaf_clean, ~]       = pwelch(clean_melody, hanning(1024), 512, Nfft, fs);

%%
figure; hold on; grid on;
% plot(f_spec, 10*log10(Psig+eps),        'LineWidth',1.3);  % reference clean
plot(f_spec, 10*log10(Pd_noisy+eps),    '-k','LineWidth', 1);  % noisy input
plot(f_spec, 10*log10(Pfull_clean+eps), '-m','LineWidth', 1.5);  % LMS
plot(f_spec, 10*log10(Psaf_clean+eps),  '-r','LineWidth', 1.5);  % SAF
xlim([0 2000]);   % or [0 fmax] as you like
xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
% title('Full-band PSD: Noisy vs LMS vs SAF');
legend('Input Signal', 'Full-band LMS', 'Subband LMS');
%%

% MSE/ Converegence comp

% Make sure lengths match (small trim if needed due to filter delay)
Lmin = min([length(d), length(clean_full), length(clean_melody)]);
sig_ref      = d(1:Lmin);
clean_full   = clean_full(1:Lmin);
clean_saf    = clean_melody(1:Lmin);

err_full = sig_ref - clean_full;
err_saf  = sig_ref - clean_saf;

% Use a moving-window MSE to smooth the curves
win = 200;   % samples
MSE_full = movmean(err_full.^2, win);
MSE_SAF  = movmean(err_saf.^2,  win);

figure; hold on; grid on;
plot(MSE_full, 'LineWidth',1.3);
plot(MSE_SAF,  'LineWidth',1.3);
xlabel('Sample index');
ylabel('Local MSE');
title(sprintf('MSE vs Time (window = %d samples)', win));
legend('Full-band LMS', 'SAF (subband)');

%%
% [b_hp, a_hp] = butter(4, 250/(fs/2), 'high');
% v_hp = filter(b_hp, a_hp, v);
% [e_full, y_full, w_full] = adaptiveLMS_complex(v_hp, d, mu, M);
% 
% disp("norm(w_full) = " + norm(w_full))
% disp("max(abs(w_full)) = " + max(abs(w_full)))
% disp("rms(e_full) = " + rms(e_full))
% disp("rms(d) = " + rms(d))
% disp("rms(y_full) = " + rms(y_full))

Nfft = 8192;

% True fullband noise, estimated noise from fullband LMS, and cleaned signal
[ Pv,  f ] = pwelch(v,       hanning(4096), 2048, Nfft, fs);
[ Py,  ~ ] = pwelch(y_full,  hanning(4096), 2048, Nfft, fs);
[ PeF, ~ ] = pwelch(e_full,  hanning(4096), 2048, Nfft, fs);  % cleaned (fullband LMS)

figure; hold on; grid on;
plot(f, 10*log10(Pv+eps),  'LineWidth', 1.2);
plot(f, 10*log10(Py+eps),  'LineWidth', 1.2);
plot(f, 10*log10(PeF+eps), 'LineWidth', 1.2);
xlim([0 2e3])
legend('True noise v(n)', 'Estimated noise y\_full(n)', 'Cleaned e\_full(n)');
xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
title('Fullband LMS: Noise vs Estimated Noise vs Cleaned Signal');

%%
% Generate power of each subband
E_noisy = sum(abs(subbands_Noisy).^2, 2);   % Kx1 – energy per subband
E_noise = sum(abs(subbands_Noise).^2, 2);  % Kx1

ratio = E_noise ./ (E_noisy + eps);   % Avoid divide-by-zero
[ratio_sorted, idx] = sort(ratio, 'descend');

% disp(table((1:K).', E_noisy, E_noise, ratio, ...
    % 'VariableNames', {'Band','E_noisy','E_noise','NoiseToTotalRatio'}));

% frameLen = 2048;
% energy   = movmean(Song.^2, frameLen);
% idx      = energy < 0.1 * max(energy);   % “low-signal” frames
% 
% Song_quiet  = Song(idx);
% Noise_quiet = Noise(idx);
% 
% rho_quiet = corrcoef(Noise_quiet, Song_quiet);
% rho_quiet = rho_quiet(1,2)
% %%
% 
% % Adaptive Filtering
% M = 64;
% mu = 0.05;
% 
% [y, e, w, MSE, W] = adaptiveLMSFilter(Noise, Song, mu, M);
% 
% e2 = e.^2;
% 
% i = 5e5;
% figure; grid on
% plot(1:i, MSE(1:i));
% ylabel('Mean Squared Error');
% xlabel('Samples (n)')
% 
% figure; grid on; hold on
% plot(1:i, Song(1:i), '--')
% plot(1:i, e(1:i), '-')
% plot(1:i, Noise(1:i), 'o')

%%
% player = audioplayer(e, Fs_song);
% play(player)
% pause(10);
% stop(player)
% 
% player = audioplayer(Song, Fs_song);
% play(player)
% pause(10)
% stop(player)
%%
% %% === LOAD AUDIO FILES ===
% [clean, fsC] = audioread(cleanFile);
% [windy, fsW] = audioread(windyFile);
% 
% % Convert to mono
% clean = mean(clean,2);
% windy = mean(windy,2);
% 
% % Resample if needed
% if fsC ~= targetFs
%     clean = resample(clean, targetFs, fsC);
% end
% if fsW ~= targetFs
%     windy = resample(windy, targetFs, fsW);
% end
% fs = targetFs;
% 
% %% === ALIGN CLEAN SEGMENT TO WINDY USING CROSS-CORRELATION ===
% disp('Aligning clean track to windy snippet...');
% [corrVals, lags] = xcorr(clean, windy);
% [~, idxMax] = max(abs(corrVals));
% bestLag = lags(idxMax);
% 
% Nw = length(windy);
% startIdx = bestLag + 1;
% if startIdx < 1
%     startIdx = 1;
% end
% endIdx = min(startIdx + Nw - 1, length(clean));
% cleanSegment = clean(startIdx:endIdx);
% 
% % Pad if windy longer than clean segment near edges
% if length(cleanSegment) < Nw
%     cleanSegment = [cleanSegment; zeros(Nw - length(cleanSegment),1)];
% end
% 
% fprintf('Alignment complete. Extracted %d samples (~%.2f s)\n', Nw, Nw/fs);
% 
% %% === OPTIONAL PREPROCESSING ===
% % Remove DC / low-frequency drift
% hpFilt = designfilt('highpassiir','FilterOrder',4, ...
%                     'HalfPowerFrequency',20, ...
%                     'SampleRate',fs,'DesignMethod','butter');
% cleanSegment = filtfilt(hpFilt, cleanSegment);
% windy        = filtfilt(hpFilt, windy);
% 
% % Optional pre-emphasis (for speech)
% if applyPreEmphasis
%     cleanSegment = filter([1 -0.95], 1, cleanSegment);
%     windy        = filter([1 -0.95], 1, windy);
% end
% 
% % Normalize both to same peak
% peakVal = max([max(abs(cleanSegment)), max(abs(windy))]);
% cleanSegment = cleanSegment / (peakVal + eps);
% windy        = windy / (peakVal + eps);
% 
% %% === OPTIONAL FRAMING (for block/adaptive processing) ===
% frameMs = 32;
% hopMs   = 16;
% L = round(frameMs * 1e-3 * fs);
% H = round(hopMs * 1e-3 * fs);
% win = hann(L, 'periodic');
% 
% numFrames = 1 + floor((length(windy)-L)/H);
% X_frames = zeros(L, numFrames);
% D_frames = zeros(L, numFrames);
% 
% idx = 1;
% for n = 1:numFrames
%     X_frames(:,n) = windy(idx:idx+L-1) .* win;
%     D_frames(:,n) = cleanSegment(idx:idx+L-1) .* win;
%     idx = idx + H;
% end
% 
% disp('Framing complete. Signals ready for subband adaptive filter.');
% 
% %% === SUMMARY ===
% fprintf('\n--- SUMMARY ---\n');
% fprintf('Sample Rate: %d Hz\n', fs);
% fprintf('Num Frames:  %d\n', numFrames);
% fprintf('Frame Length: %d samples (%.2f ms)\n', L, frameMs);
% fprintf('Hop: %d samples (%.2f ms)\n', H, hopMs);
% fprintf('Duration: %.2f s\n', length(windy)/fs);
% 
% %% === OPTIONAL PLOTS ===
% t = (0:length(windy)-1)/fs;
% figure;
% subplot(2,1,1); plot(t, windy); title('Windy (Corrupted) Audio'); xlabel('Time (s)');
% subplot(2,1,2); plot(t, cleanSegment); title('Aligned Clean Segment'); xlabel('Time (s)');
% % sgtitle('Aligned Audio Signals');