function [B1, B2, W3, W4] = DAZSH(X1, X2, Y, A, options)
   
% label matrix N x c
if isvector(Y) 
    Y = sparse(1:length(Y), double(Y), 1); Y = full(Y);    Y=Y';
end
[n, c] = size(Y);
dA = size(X1,2);


sampleColumn = options.num_samples;
nbits = options.l;
alpha1 = options.alpha1;
alpha2 = options.alpha2;
beta1 = options.beta1; 
beta2 = options.beta2; 
mu1 = options.mu1;
mu2 = options.mu2;                                                                             
gamma = options.gamma;
lambda = options.lambda;


%% initial 
Z1 = rand(n,n);
Z2 = rand(n,n);
W1 = rand(c,nbits);
W2 = rand(c,nbits);
W3 = rand(dA,nbits);
W4 = rand(dA,nbits);
B1 = sign(randn(n,nbits));
B2 = sign(randn(n,nbits));


S = Y * Y';        
S(S>0) = 1;

i = 0;

X1AL = alpha1*X1*A'*Y';
X2AL = alpha2*X2*A'*Y';
LA = Y*A*A'*Y';
X1X1 = mu1*X1'*X1+gamma*eye(dA);
X2X2 = mu2*X2'*X2+gamma*eye(dA);

while i < options.maxItr    
    i=i+1;  
   %% sample Sc
    Sc = randperm(n, sampleColumn);
    % update B1
    SX = Y * Y(Sc, :)' > 0;    
    B1 = updateColumnV(B1, B2, SX, Sc, nbits, lambda, sampleColumn, beta1, mu1, Z1, Y, W1, X1, W3);

    % update B2
    SY = Y(Sc, :) * Y' > 0;
    B2 = updateColumnB(B2, B1, SY, Sc, nbits, lambda, sampleColumn,beta2, mu2, Z2, Y, W2, X2, W4); 

    %  Update Z1
    Z1 = (X1AL + beta1*B1*W1'*Y')*pinv(alpha1*LA + beta1*Y*W1*W1'*Y' + gamma*eye(n));
    
    %  Update Z2
    Z2 = (X2AL + beta2*B2*W2'*Y')*pinv(alpha2*LA + beta1*Y*W2*W2'*Y' + gamma*eye(n));

    %  Update W1
    W1 = pinv(beta1*Y'*Z1'*Z1*Y+gamma*eye(c))*(beta1*Y'*Z1'*B1);
    
    %  Update W2
    W2 = pinv(beta2*Y'*Z2'*Z2*Y+gamma*eye(c))*(beta2*Y'*Z2'*B2);
    
    %  Update W3
    W3 = pinv(X1X1)*(mu1*X1'*B1);
    
    %  Update W4
    W4 = pinv(X2X2)*(mu2*X2'*B2);

end
end

function B1 = updateColumnV(B1, B2, SX, Sc, bit, lambda, sampleColumn, beta1, mu1, Z1, Y, W1, X1, W3)
m = sampleColumn;
n = size(B1, 1);
for k = 1: bit
    TX = lambda * B1 * B2(Sc, :)' / bit;
    AX = 1 ./ (1 + exp(-TX));
    Bjk = B2(Sc, k)';
    aaa = (beta1 + mu1);
    p = lambda * ((SX - AX) .* repmat(Bjk, n, 1)) * ones(m, 1) / bit...
        + (m * lambda^2 + 8*bit^2 *aaa)* B1(:, k) / (4 * bit^2) + ...
        + 2*beta1*( B1(:,k)- Z1*Y*W1(:,k)) + 2*mu1*(B1(:,k)- X1*W3(:,k));
    U_opt = ones(n, 1);
    U_opt(p < 0) = -1;
    B1(:, k) = U_opt;
end
end


function B2 = updateColumnB(B2, B1, SY, Sc, bit, lambda, sampleColumn,beta2, mu2, Z2, Y, W2, X2, W4)
m = sampleColumn;
n = size(B1, 1);
for k = 1: bit
    TX1 = lambda * B1(Sc, :) * B2' / bit;
    AX1 = 1 ./ (1 + exp(-TX1));
    Ujk = B1(Sc, k)';  %1*8
    aaa = (beta2 + mu2);
    p = lambda * ((SY' - AX1') .* repmat(Ujk, n, 1)) * ones(m, 1)  / bit +...
        + (m * lambda^2 + 8*bit^2 *aaa)* B2(:, k) / (4 * bit^2) + ...
        + 2*beta2*( B2(:,k)- Z2*Y*W2(:,k)) + 2*mu2*(B2(:,k)- X2*W4(:,k));
    B_opt = ones(n, 1);
    B_opt(p < 0) = -1;
    B2(:, k) = B_opt;
end
end