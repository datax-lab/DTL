function [ACCU1, RESULT, accu1] = test_by_DTL_LRR(DATA, TEST, alphas, gammas)%, PRE_PROC)

SOURCE_BASE = [DATA.OSBX; DATA.OSX];
SOURCE_BASE_LABEL = normalize([DATA.OSBY; DATA.OSY]);
SOURCE_BASE_LABEL(sign(SOURCE_BASE_LABEL) >= 0) = 1;
SOURCE_BASE_LABEL(sign(SOURCE_BASE_LABEL) < 0) = -1;


assert(length(unique(DATA.OSY)) == 2, 'TARGET_LABEL must be discritized into two values');

%% train 
mu = cell(2, 1);        % mean of original variable
el = cell(2, 1);        % Euclidean lengths for each original variable.
n = size(DATA.OSX, 1);

% Obtain normalization info with all available data
X1 = [DATA.OSBX; DATA.OSX];
X2 = [DATA.OTBX; DATA.OTX];

[~, IX1] = sort(sum(X1), 'descend');
[~, IX2] = sort(sum(X2), 'descend');

h1 = IX1(1:min(2000, length(IX1)));
h2 = IX2(1:min(2000, length(IX2)));
[~, mu{1}, el{1}] = normalize(X1(:, h1));
[~, mu{2}, el{2}] = normalize(X2(:, h2));

% prepare paired data
X1 = DATA.OSX(:, h1);
X2 = DATA.OTX(:, h2);
y = normalize(DATA.OSY);
y(sign(y) >= 0) = 1;
y(sign(y) < 0) = -1;

% Build classifiers on each language
nb = size(SOURCE_BASE, 1);
X = [ones(nb, 1) normalize(SOURCE_BASE(:, h1), mu{1}, el{1})];
SB1 = (X' * X + 0.1 * eye(size(h1, 2) + 1)) \ X' * SOURCE_BASE_LABEL;

X = [ones(n, 1) X2];
SB2 = (X' * X + 0.1 * eye(size(h2, 2) + 1)) \ X' * y;

amp = 2;
X1 = [datasample(X1(y == 1, :), n * amp); datasample(X1(y == -1, :), n * amp)];
X2 = [datasample(X2(y == 1, :), n * amp); datasample(X2(y == -1, :), n * amp)];
y = [repmat(y(y == 1), [amp * 2, 1]); repmat(y(y == -1), [amp * 2, 1])];

R = 5;

accu1 = zeros(length(alphas) * length(gammas), 1);
for i = 1 : length(gammas)
    
    gamma = gammas(i);
    for j = 1 : length(alphas)

        alpha = alphas(j);
        [A1, A2, B1, B2, W1, W2, T, M1, M2] = DTL_ridge(X1, X2, y, R, 'DISP', false, 'ALPHA', alpha, 'GAMMA', gamma);
        
        n = size(TEST.OTX, 1);
        NTEST = normalize(TEST.OTX(:, h2), mu{2}, el{2});
        NRX1 = NTEST * A2 * T * B1';
        
        class = zeros(n, 1);
        yfit1 = [ones(n, 1) NTEST] * SB2;
        yfit2 = [ones(n, 1) NRX1] * SB1;
        idx = abs(yfit1) > abs(yfit2);
        class(idx) = sign(yfit1(idx)) > 0;
        class(~idx) = sign(yfit2(~idx)) > 0;

        ACCU1 = sum(class == TEST.OTY) / length(TEST.OTY);
        accu1((i-1) * length(alphas)+ j) = ACCU1;
    end
end

[~, ix] = max(sum(accu1, 2));
ACCU1 = accu1(ix);


%% Output
RESULT.converge = true;
RESULT.prediction = class;
RESULT.train = [];
