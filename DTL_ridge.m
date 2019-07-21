function [A1, A2, B1, B2, W1, W2, T, M1, M2, COR] = DTL_ridge(X1, X2, y, R, varargin)

pnames = {'CV', 'CONVERGE', 'MAX_ITERATION', 'DFmax', 'DISP', 'ALPHA', 'GAMMA', 'RHO'};
dflts  = {'resubstitution', 1e-3, 5000, Inf, false, 1, 1, 0.8};
[CV, CONVERGE, MAX_ITERATION, DFMAX, DISP, ALPHA, GAMMA, RHO] = internal.stats.parseArgs(pnames, dflts, varargin{:});

n = size(X1, 1);
p1 = size(X1, 2);
p2 = size(X2, 2);
b = sign(y);

A1 = zeros(p1, R);
A2 = zeros(p2, R);
M1 = zeros(n, R);
M2 = zeros(n, R);
COR = zeros(R, 1);

OX1 = X1;
OX2 = X2;

for r = 1 : R

    a1 = ones(p1, 1);
    a2 = ones(p2, 1);

    m1 = zeros(n, 1);
    m2 = zeros(n, 1);

    prev_total_error = 0;
    for iter = 1 : MAX_ITERATION
        drawnow;
        a1 = ((1 + GAMMA) * (X1' * X1) + ALPHA * eye(p1)) \ (X1' * y + X1' * (b .* m1) + GAMMA * X1' * X2 * a2);
        a2 = ((1 + GAMMA) * (X2' * X2) + ALPHA * eye(p2)) \ (X2' * y + X2' * (b .* m2) + GAMMA * X2' * X1 * a1);

        m1 = max((X1 * a1 - y) .* b, 0);
        m2 = max((X2 * a2 - y) .* b, 0);

        total_error = norm(X1 * a1 - y - b .* m1) + norm(X2 * a2 - y - b .* m2) ...
            + GAMMA * norm(X1 * a1 - X2 * a2) + ALPHA * norm(a1, 2) + ALPHA * norm(a2, 2);
        
        if DISP
            fprintf('Iter %d: NZ1: %d, NZ2: %d, TE: %f ', iter, ...
                length(find(a1)), length(find(a2)), total_error);
            fprintf('[1]%.2f, [2]%.2f, [3]%.2f, cor: %.4f\n', norm(X1 * a1 - y - b .* m1), ...
                norm(X2 * a2 - y - b .* m2), ...
                norm(X1 * a1 - X2 * a2), ...
                corr(X1 * a1, X2 * a2));
        end

        if abs(prev_total_error - total_error) < CONVERGE
            if DISP, fprintf('Converged at %d\n', iter); end
            break;
        end
        prev_total_error = total_error;
    end
    if iter == MAX_ITERATION, warning('max iteration'); end
    cor = abs(corr(X1 * a1, X2 * a2));
    if r > 1 && (isempty(a1) || isempty(a2) || cor < RHO)
        A1 = A1(:, 1:r - 1);
        A2 = A2(:, 1:r - 1);
        B1 = B1(:, 1:r - 1);
        B2 = B2(:, 1:r - 1);
        M1 = M1(:, 1:r - 1);
        M2 = M2(:, 1:r - 1);
        COR = COR(1:r-1);
        break;
    end
    
    l1 = X1 * a1;
    l2 = X2 * a2;

    b1 = (l1' * l1) \ (X1' * l1);
    b2 = (l2' * l2) \ (X2' * l2); 
    
    X1 = X1 - l1 * b1';
    X2 = X2 - l2 * b2';
    
    A1(:, r) = a1;
    A2(:, r) = a2;
    B1(:, r) = b1;
    B2(:, r) = b2;
    M1(:, r) = m1;
    M2(:, r) = m2;
    COR(r) = cor;
end

W1 = regress(y, [ones(n, 1) OX1 * A1]);
W2 = regress(y, [ones(n, 1) OX2 * A2]);

L1 = OX1 * A1;
L2 = OX2 * A2;
T = (L2' * L2) \ (L1' * L2);
