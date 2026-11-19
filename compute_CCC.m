function [ccc,mean_ccc] = compute_CCC(data,U,V)
% Signed canonical correlation coefficient (not Lin's concordance measure).
% V must be TOTAL imaging weights: V=V3+V2+V1=C+B+S.
% Use inputs standardized with TRAINING means/SDs, as in run_MDPGC.m.
% No additional centering, absolute values or validation-based sign changes.
% Invalid or zero-norm projections return NaN; mean does not omit NaNs.
assert(iscell(data.Y) && numel(data.Y)==3 && size(U,2)==3 && size(V,2)==3, ...
    'Expected three modalities and three weight columns.');
ccc = nan(1,3);
for k = 1:3
    x = data.X*U(:,k); y = data.Y{k}*V(:,k);
    nx = norm(x); ny = norm(y);
    if isreal(x) && isreal(y) && all(isfinite(x)) && all(isfinite(y)) && ...
            isfinite(nx) && isfinite(ny) && nx>0 && ny>0
        ccc(k) = max(-1,min(1,(x/nx)'*(y/ny)));
    end
end
mean_ccc = mean(ccc);
end
