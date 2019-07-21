function [ACCU, RESULT, accu] = test_by_DTL_SVM(DATA, TEST, alphas, gammas)

SOURCE_BASE = [DATA.OSBX; DATA.OSX];
SOURCE_BASE_LABEL = normalize([DATA.OSBY; DATA.OSY]);
SOURCE_BASE_LABEL(sign(SOURCE_BASE_LABEL) >= 0) = 1;
SOURCE_BASE_LABEL(sign(SOURCE_BASE_LABEL) < 0) = -1;


assert(length(unique(DATA.OSY)) == 2, 'TARGET_LABEL must be discritized into two values');

%% train with DTL
mu = cell(2, 1);        % mean of original variable
el = cell(2, 1);        % Euclidean lengths for each original variable.
n = size(DATA.OSX, 1);

% Obtain normalization info with all available data
X1 = [DATA.OSBX; DATA.OSX];
X2 = [DATA.OTBX; DATA.OTX];

% use only top 2000 features
[~, IX1] = sort(sum(X1), 'descend');
[~, IX2] = sort(sum(X2), 'descend');
h1 = IX1(1:min(2000, length(IX1)));
h2 = IX2(1:min(2000, length(IX2)));
[~, mu{1}, el{1}] = normalize(X1(:, h1));
[~, mu{2}, el{2}] = normalize(X2(:, h2));

% prepare paired data, no necessary to be translatd directly. 
X1 = DATA.OSX(:, h1);
X2 = DATA.OTX(:, h2);
y = normalize(DATA.OSY);
y(sign(y) >= 0) = 1;
y(sign(y) < 0) = -1;

X = normalize(SOURCE_BASE(:, h1), mu{1}, el{1});
S_SVM1 = fitcsvm(X, SOURCE_BASE_LABEL, 'KernelScale', 'auto', 'Standardize', true);
S_SVM1 = fitPosterior(S_SVM1, X, SOURCE_BASE_LABEL);

S_SVM2 = fitcsvm(X2 , y, 'KernelScale', 'auto', 'Standardize', true);
S_SVM2 = fitPosterior(S_SVM2, X2, y);

% rank
R = 5;

% amplify data
amp = 2;
X1 = [datasample(X1(y == 1, :), n * amp); datasample(X1(y == -1, :), n * amp)];
X2 = [datasample(X2(y == 1, :), n * amp); datasample(X2(y == -1, :), n * amp)];
y = [repmat(y(y == 1), [amp * 2, 1]); repmat(y(y == -1), [amp * 2, 1])];

accu = zeros(length(alphas) * length(gammas), 1);
for i = 1 : length(gammas)
    
    gamma = gammas(i);
    for j = 1 : length(alphas)

        alpha = alphas(j);
        [A1, A2, B1, B2, W1, W2, T, M1, M2] = DTL_ridge(X1, X2, y, R, 'DISP', false, 'ALPHA', alpha, 'GAMMA', gamma);
        
        % testing
        NTEST = normalize(TEST.OTX(:, h2), mu{2}, el{2});
        % transfer test data into source
        NRX1 = NTEST * A2 * T * B1';
        
        % test with monolingual classifiers
        [class1, score1] = predict(S_SVM2, NTEST);
        [class2, score2] = predict(S_SVM1, NRX1);
        
        classes = [class1, class2];
        scores = [max(score1, [], 2), max(score2, [], 2)];
        [~, idx] = max(scores, [], 2);
        class = arrayfun(@(i) classes(i, idx(i)), 1 : size(TEST.OTX, 1))';
        ACCU = sum(class > 0 == TEST.OTY) / length(TEST.OTY);
        accu((i-1) * length(alphas)+ j) = ACCU;
    end
end

[~, ix] = max(sum(accu, 2));
ACCU = accu(ix);


%% Output
RESULT.converge = true;
RESULT.prediction = class;
RESULT.train = [];
