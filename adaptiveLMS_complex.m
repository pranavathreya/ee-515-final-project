function [e, y, w, W] = adaptiveLMS_complex(x, d, mu, M)
% Complex LMS (for subband adaptive filtering)
% x(n) - reference noise
% d(n) - primary (signal + noise)
% mu   - step size
% M    - filter length

x = x(:);
d = d(:);
N = length(d);

w = zeros(M,1);   % complex weights
y = zeros(N,1);
e = zeros(N,1);

for n = M:N
    u = x(n:-1:n-M+1);     % reference input vector
    y(n) = w' * u;         % noise estimate
    e(n) = d(n) - y(n);    % cleaned signal
    w = w + mu * u * conj(e(n));
    W(:,n) = w;
end

end