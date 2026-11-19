function split = split_cv(n,seed,n_folds,test_fraction)
% Defaults: 80% development / 20% held-out test; five-fold CV within development.
% Optional n_folds/test_fraction configure CV and holdout size.
% Defaults retain the random operations/order of original_cv_split.m.
% No Statistics Toolbox needed. Does not modify the global random stream.
if nargin<3, n_folds = 5; end
if nargin<4, test_fraction = 0.20; end
validateattributes(n,{'numeric'},{'real','scalar','finite','integer','>=',10});
validateattributes(seed,{'numeric'}, ...
    {'real','scalar','finite','integer','>=',0,'<=',2^32-1});
validateattributes(n_folds,{'numeric'},{'real','scalar','finite','integer','>=',2});
validateattributes(test_fraction,{'numeric'},{'real','scalar','finite','>',0,'<',1});
n = double(n); n_folds = double(n_folds); test_fraction = double(test_fraction);
s = RandStream('mt19937ar','Seed',double(seed));
split.n_folds = n_folds;
order = randperm(s,n); nt = max(1,round(test_fraction*n));
split.test_rows = sort(order(1:nt));
split.development_rows = sort(order(nt+1:end));
nd = numel(split.development_rows);
assert(nd>split.n_folds && nd-ceil(nd/split.n_folds)>=2, ...
    'Insufficient development subjects for the requested CV folds and test fraction.');
perm = randperm(s,nd); fold_id = zeros(nd,1);
fold_id(perm) = mod((0:nd-1)',split.n_folds)+1;
split.inner_fold_id = fold_id;
end
