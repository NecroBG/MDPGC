function [ccc,U,C,B,S] = run_MDPGC(data)
% Usage: [ccc,U,C,B,S] = run_MDPGC(data);
% Input: data.X is n-by-p; data.Y contains three n-by-q matrices in a cell array.
% Outputs: ccc contains signed held-out test CCCs (1-by-3); mean(ccc) is their mean.
%          U is p-by-3; C, B and S are q-by-3; total imaging weights V=C+B+S.
%          Weights are refitted on the development set with the selected parameters
%          and apply to features standardized using development means and SDs.
%          Columns correspond to data.Y{1}, data.Y{2} and data.Y{3}, respectively.
% Set experiment parameters below; only data is required when calling this function.
% Requires only split_cv.m, MTMVSCCA.m and compute_CCC.m in this folder.

%% Search ranges: multiple values define candidates; a scalar fixes a parameter.
settings.search_values = {
    [1e-4,1,1e4]             % lambda_uf21: adjacent-feature group penalty on U
    [1e-4,1,1e4]             % lambda_ul21: row-group sparsity penalty on U
    [1e-4,1,1e4]             % lambda_u1: elementwise sparsity penalty on U
    [1e-4,1,1e4]             % lambda_v3: shared sparsity penalty on C
    [1e-4,1,1e4]             % lambda_v2: sparsity penalty on B
    [1e-4,1,1e4]             % lambda_v1: sparsity penalty on S
    [1e-4,1e-2,1,1e2,1e4]   % beta1: non-Gaussian gradient multiplier
    [1e-4,1e-2,1,1e2,1e4]   % beta2: cross-modal coupling multiplier
};

%% Data split and solver settings.
settings.n_folds = 5;          % Cross-validation folds within the development set
settings.test_fraction = 0.20; % Held-out test fraction; remaining subjects form the development set
settings.max_iter = 100;       % Maximum iterations per model fit
settings.tol = 1e-5;           % Maximum absolute weight change between consecutive iterations
settings.learning_rate = 1e-4; % Gradient step size for B; maps to solver opts.alpha
settings.seed = 1;            % Single random seed for data splitting and Gaussian references

% Defaults: 18,225 candidates, 91,125 five-fold CV fits, and one final refit.
% Select the parameters with the highest mean CCC across all folds and modalities.
% Score the test set only after the final refit; it is not used for selection.

%% Validate inputs and settings.
settings = validate_settings(settings);
assert(isstruct(data) && isscalar(data) && all(isfield(data,{'X','Y'})), ...
    'Supply data.X and data.Y.');
assert(iscell(data.Y) && numel(data.Y)==3,'Exactly three imaging modalities are required.');
data.Y = reshape(data.Y,1,3);
arrays = [{data.X},data.Y];
for k = 1:4
    a = arrays{k};
    assert(isa(a,'double') && isreal(a) && ismatrix(a) && ~issparse(a) && ...
        ~isempty(a) && all(isfinite(a(:))), ...
        'Inputs must be finite, full, real double matrices.');
end
[n,p] = size(data.X); q = size(data.Y{1},2);
assert(p>=2 && q>=2,'At least two SNPs and two imaging features are required.');
for k = 1:3
    assert(isequal(size(data.Y{k}),[n,q]),'All Y matrices must have the same n-by-q size.');
end
solver = functions(@MTMVSCCA);
assert(strcmpi(solver.file,fullfile(fileparts(mfilename('fullpath')),'MTMVSCCA.m')), ...
    'Use the MTMVSCCA.m in the same folder as run_MDPGC.m.');

%% Build the parameter grid and split the subjects.
names = {'lambda_uf21','lambda_ul21','lambda_u1', ...
    'lambda_v3','lambda_v2','lambda_v1','beta1','beta2'};
P = numel(names);
search_values = settings.search_values;
assert(iscell(search_values) && numel(search_values)==P, ...
    'Supply one search range per named coefficient in a cell array.');
for k = 1:P
    v = search_values{k};
    validateattributes(v,{'numeric'},{'real','vector','nonempty','finite','nonnegative'});
    assert(numel(unique(v))==numel(v),'Search values must be distinct.');
    if k<=6, assert(all(v>0),'Sparse coefficient search values must be positive.'); end
    search_values{k} = double(v(:)');
end
grid_axes = cell(1,P); [grid_axes{:}] = ndgrid(search_values{:});
grid = zeros(numel(grid_axes{1}),P);
for k = 1:P, grid(:,k) = grid_axes{k}(:); end
opts.alpha = settings.learning_rate;
opts.max_iter = settings.max_iter;
opts.tol = settings.tol;

split = split_cv(n,settings.seed,settings.n_folds,settings.test_fraction);
F = split.n_folds;
% Derive fold reference seeds internally, wrapping within the valid seed range.
% Every candidate in a given fold uses exactly the same Gaussian reference.
fold_seeds = mod(settings.seed+100*(1:F),2^32);

%% Fit preprocessing on each training fold, then transform its validation fold.
training = cell(F,1); validation = cell(F,1);
for f = 1:F
    tr = split.development_rows(split.inner_fold_id~=f);
    va = split.development_rows(split.inner_fold_id==f);
    [training{f},prep] = standardize(take_rows(data,tr));
    validation{f} = standardize(take_rows(data,va),prep);
end

%% Search for the highest mean validation CCC across folds and modalities.
N = size(grid,1);
scores = nan(N,1);
first_error = '';
fprintf('MDPGC: %d candidates x %d folds = %d fits.\n',N,F,N*F);
for j = 1:N
    for k = 1:P, opts.(names{k}) = grid(j,k); end
    fold_ccc = nan(F,3);
    for f = 1:F
        [m,ok,message] = fit_model(training{f},opts,fold_seeds(f));
        if ok
            c = compute_CCC(validation{f},m.U,m.V);
            fold_ccc(f,:) = c;
            if ~all(isfinite(c)), message = 'Degenerate validation projection.'; end
        end
        if isempty(first_error) && ~isempty(message), first_error = message; end
    end
    % Require valid scores for every fold and modality; keep their original signs.
    if all(isfinite(fold_ccc(:)))
        scores(j) = mean(mean(fold_ccc,2));
    end
    if mod(j,20)==0 || j==N, fprintf('Completed %d/%d candidates.\n',j,N); end
end
eligible = find(isfinite(scores));
if isempty(eligible)
    error('MDPGC:NoValidCandidate','No valid model was selected. First failure: %s',first_error);
end
[best_score,position] = max(scores(eligible)); % First grid candidate wins exact ties.
best = eligible(position);
for k = 1:P, opts.(names{k}) = grid(best,k); end

%% Refit on all development subjects and evaluate the held-out test set once.
[development,prep] = standardize(take_rows(data,split.development_rows));
test = standardize(take_rows(data,split.test_rows),prep);
[m,ok,message] = fit_model(development,opts,settings.seed);
if ~ok
    error('MDPGC:InvalidFinalModel','Final model fit failed: %s',message);
end
[ccc,mean_ccc] = compute_CCC(test,m.U,m.V);
if ~all(isfinite(ccc))
    error('MDPGC:InvalidFinalModel','Final test evaluation failed: degenerate test projection.');
end
U = m.U;
C = m.C;
B = m.B;
S = m.S;
fprintf('Selected candidate %d; CV mean CCC=%.9f; test mean CCC=%.9f.\n', ...
    best,best_score,mean_ccc);
% Finite weights do not establish convergence; the solver has no convergence diagnostics.
end

function settings = validate_settings(settings)
validateattributes(settings.n_folds,{'numeric'},{'real','scalar','finite','integer','>=',2});
validateattributes(settings.test_fraction,{'numeric'},{'real','scalar','finite','>',0,'<',1});
validateattributes(settings.max_iter,{'numeric'},{'real','scalar','finite','integer','>=',1});
validateattributes(settings.tol,{'numeric'},{'real','scalar','finite','nonnegative'});
validateattributes(settings.learning_rate,{'numeric'},{'real','scalar','finite','positive'});
validateattributes(settings.seed,{'numeric'}, ...
    {'real','scalar','finite','integer','>=',0,'<=',2^32-1});
names = {'n_folds','test_fraction','max_iter','tol','learning_rate','seed'};
for k = 1:numel(names), settings.(names{k}) = double(settings.(names{k})); end
end

function out = take_rows(data,rows)
out.X = data.X(rows,:); out.Y = cell(1,3);
for k = 1:3, out.Y{k} = data.Y{k}(rows,:); end
end

function [out,prep] = standardize(data,prep)
% Means and sample standard deviations are fitted on training subjects only.
if nargin<2
    prep.x_mean = mean(data.X,1); prep.x_sd = std(data.X,0,1);
    prep.x_sd(prep.x_sd<1e-12) = 1;
    for k = 1:3
        prep.y_mean{k} = mean(data.Y{k},1); prep.y_sd{k} = std(data.Y{k},0,1);
        prep.y_sd{k}(prep.y_sd{k}<1e-12) = 1;
    end
end
out.X = (data.X-prep.x_mean)./prep.x_sd; out.Y = cell(1,3);
for k = 1:3, out.Y{k} = (data.Y{k}-prep.y_mean{k})./prep.y_sd{k}; end
end

function [m,ok,message] = fit_model(training,opts,seed)
% Set a local reference seed for this fit and restore the caller's RNG afterward.
m = []; ok = false; message = '';
previous = RandStream.getGlobalStream;
restore_rng = onCleanup(@()RandStream.setGlobalStream(previous));
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed));
[old_message,old_id] = lastwarn;
restore_lastwarn = onCleanup(@()lastwarn(old_message,old_id));
ids = {'MATLAB:singularMatrix','MATLAB:nearlySingularMatrix','MATLAB:illConditionedMatrix'};
states = cellfun(@(id)warning('query',id),ids,'UniformOutput',false);
restore_warn = onCleanup(@()restore_warnings(states));
for k = 1:numel(ids), warning('off',ids{k}); end
try
    [U,C,B,S] = MTMVSCCA(training,opts);
    m = struct('U',U,'C',C,'B',B,'S',S,'V',C+B+S);
    if ~all(isfinite([U(:);C(:);B(:);S(:);m.V(:)]))
        message = 'Nonfinite solver weights.'; return;
    end
    c = compute_CCC(training,m.U,m.V);
    ok = all(isfinite(c));
    if ~ok, message = 'Degenerate training projection.'; end
catch ex
    message = ex.message;
end
end

function restore_warnings(states)
for k = 1:numel(states), warning(states{k}); end
end
