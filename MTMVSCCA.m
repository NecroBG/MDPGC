function [U, V3, V2, V1] = MTMVSCCA(data, opts)
% Eight adjustable model coefficients: the six lambda fields in opts,
% plus opts.beta1 and opts.beta2 (both required, finite, nonnegative).
% Same beta convention as MTMVSCCA_beta.m: beta1 multiplies the ICA
% gradient; beta2 is the direct cross-modal coupling multiplier.
% Coupling penalty: (beta2/2)*sum_{k<j} ||Y{k}*S(:,k)-Y{j}*S(:,j)||^2.
% beta1=0.01, beta2=1 recover the original fixed coefficients.
% opts.alpha remains the learning rate; it is not one of these eight.
% Optional opts.max_iter (default 100), opts.tol (default 1e-5) set stopping.
% Original updates and scaling remain unchanged.
% --------------------------------------------------
% MTMVSCCA
% Date: 19-Jan-2022
% @ Northwestern Polytechnical University, School of Automation
% --------------------------------------------------
% data
X = data.X;
Y = data.Y;
r = size(Y{1}, 1);
p = size(X, 2);
q = size(Y{1}, 2);

w1 = randn(r,1);
w2 = randn(r,1);
w3 = randn(r,1);

w{1} = w1;
w{2} = w2;
w{3} = w3;

U  = ones(p, 3);
V1 = ones(q, 3);
V2 = ones(q, 3);
V3 = ones(q, 3);

% set parameters
alpha = opts.alpha ;
lambda_uf21 = opts.lambda_uf21;
lambda_ul21 = opts.lambda_ul21;
lambda_u1 = opts.lambda_u1;
lambda_v3 = opts.lambda_v3;
lambda_v2 = opts.lambda_v2;
lambda_v1 = opts.lambda_v1;
beta1 = opts.beta1;
beta2 = opts.beta2;
validateattributes(beta1,{'numeric'},{'real','scalar','finite','nonnegative'});
validateattributes(beta2,{'numeric'},{'real','scalar','finite','nonnegative'});

% set stopping criteria
max_Iter = 100;
t = 0;
tol = 1e-5;
if isfield(opts,'max_iter'), max_Iter = opts.max_iter; end
if isfield(opts,'tol'), tol = opts.tol; end
validateattributes(max_Iter,{'numeric'},{'real','scalar','finite','integer','>=',1});
validateattributes(tol,{'numeric'},{'real','scalar','finite','nonnegative'});
validateattributes(alpha,{'numeric'},{'real','scalar','finite','positive'});
max_Iter = double(max_Iter); tol = double(tol); alpha = double(alpha);
tu = inf;
tv1 = inf;
tv2 = inf;
tv3 = inf;

XX = X' * X;
for i = 1 : 3
    XY{i} = X' * Y{i};
    YY{i} = Y{i}' * Y{i};
    YX{i} = XY{i}';
end

% initial scale
for i = 1 : 3
    U_set = U(:, i) ;
    scale = sqrt(U_set' * XX * U_set);
    U(:, i) = U(:, i) ./ scale;
    
    V_set = V3(:, i) + V2(:, i) + V1(:, i);
    scale = sqrt(V_set' * YY{i} * V_set);
    V1(:, i) = V1(:, i) / scale;
    V2(:, i) = V2(:, i) / scale;
    V3(:, i) = V3(:, i) / scale;
end

while (t<max_Iter && (tu>tol|| tv3>tol ||tv2>tol || tv1>tol))

        t = t + 1;
        U_old = U;
        V3_old = V3;
        V2_old = V2;
        V1_old = V1;

        % update U^(i)
        fd1 = updateD(U(:, 1), U(:, 2), U(:, 3));
        fD1 = diag(fd1);
        d1 = updateD(U);
        D1 = diag(d1);
        for i = 1 : 3
            d1i = updateD(U(:, i));
            D1i = diag(d1i);
            F1 = XX + lambda_ul21 * D1+ lambda_uf21 * fD1 +lambda_u1*D1i ;
            b1 = XY{i} * (V3(:, i) +V2(:, i) + V1(:, i));
            U(:, i) = F1 \ b1;
        end

        for i = 1 : 3
            U_set = U(:, i) ;
            scale = sqrt(U_set' * XX * U_set);
             U(:, i) = U(:, i) ./ scale;
        end

        % update V^(i)
        % correlation-dependent
        d2i = updateD(V1(:, 1));
        D2i = diag(d2i);
        F1 =  (1+2*beta2)*YY{1} + lambda_v1 *D2i;
        b1 = YX{1} * U(:,1) - Y{1}' *Y{1}*(V2(:, 1)+V3(:, 1)) + beta2*( Y{1}' * Y{2} * V1(:, 2) +  Y{1}' * Y{3} *V1(:, 3));
        V1(:, 1) = F1 \ b1;

        d2i = updateD(V1(:, 2));
        D2i = diag(d2i);
        F1 =  (1+2*beta2)*YY{2} + lambda_v1 *D2i;
        b1 = YX{2} * U(:,2)- Y{2}' *Y{2}*(V2(:, 2)+V3(:, 2)) + beta2*(Y{2}' * Y{1} * V1(:, 1) +  Y{2}' * Y{3} * V1(:, 3));
        V1(:, 2) = F1 \ b1;

        d2i = updateD(V1(:, 3));
        D2i = diag(d2i);
        F1 =  (1+2*beta2)*YY{3} + lambda_v1 *D2i;
        b1 = YX{3} * U(:,3)- Y{3}' *Y{3}*(V2(:, 3)+V3(:, 3)) + beta2*( Y{3}' * Y{1} * V1(:, 1) +  Y{3}' * Y{2} * V1(:, 2));
        V1(:, 3) = F1 \ b1;

        % specific
        for i = 1 : 3
            d2i = updateD(V2(:, i));
            D2i = diag(d2i);
            YIter = Y{i};
            ww = w{i};
            T1sum = 0;
            for k = 1:r
                T1 = (log(cosh(YIter(k,:)*V2(:,i)))-log(cosh(ww(k,:))))*tanh(YIter(k,:)*V2(:,i))*(YIter(k,:))';
                T1sum = T1sum+T1;
            end
            g1 = YY{i}*V2(:, i) - YX{i}*U(:, i) + Y{i}'*Y{i}*(V1(:, i) + V3(:, i)) +lambda_v2*D2i*V2(:,i) - beta1*T1sum;
            V2(:, i)=V2(:, i) - alpha*g1;
        end

        % anatomical-consistent
        d3 = updateD(V3);
        D3 = diag(d3);
        for i = 1 : 3
            F3 = YY{i} + lambda_v3 * D3;
            b3 = YX{i} * U(:, i)-Y{i}'*Y{i}*(V1(:, i) + V2(:, i));
            V3(:, i) = F3 \ b3;
        end

        % stopping condition
        % ------------------------------
        tu = max(max(abs(U - U_old)));
        tv3 = max(max(abs(V3 - V3_old)));
        tv2 = max(max(abs(V2 - V2_old)));
        tv1 = max(max(abs(V1 - V1_old)));

    end
    
    % scale V^(i)
    for i = 1 : 3
        V_set = V3(:, i) + V2(:, i) + V1(:, i);
        scale = sqrt(V_set' * YY{i} * V_set);
        V1(:, i) = V1(:, i) ./ scale;
        V2(:, i) = V2(:, i) ./ scale;
        V3(:, i) = V3(:, i) ./ scale;
   
   end
   
 function D = updateD(u1, u2, u3)

% Update the diagnoal matrix
if nargin == 1
    W = u1;
    [~, n_tasks] = size(W);
    if n_tasks == 1
        % for L1-norm
        d = 1 ./ sqrt(W .^ 2 + eps);
    else
        % for L2,1-norm
        d = 1 ./ sqrt(sum(W .^ 2, 2) + eps);
    end
else
    % for FGL2,1-norm
    ulen = length(u1);
    for i = 1 : ulen
        if i == 1
            d(i) = 1 ./ sqrt(u1(i).^2 + u2(i).^2 + u3(i).^2 + u1(i+1).^2 + u2(i+1).^2 + u3(i+1).^2 + eps);
        elseif i == ulen
            d(i) = 1 ./ sqrt(u1(i-1).^2 + u2(i-1).^2 + u3(i-1).^2 + u1(i).^2 + u2(i).^2 + u3(i).^2 + eps);
        else
            d(i) = 1 ./ (sqrt(u1(i-1).^2 + u2(i-1).^2 + u3(i-1).^2 + u1(i).^2 + u2(i).^2 + u3(i).^2 + eps)) ...
                + 1 ./ (sqrt(u1(i).^2 + u2(i).^2 + u3(i).^2 + u1(i+1).^2 + u2(i+1).^2 + u3(i+1).^2 + eps));
        end
    end
end

D = 0.5 * d;
