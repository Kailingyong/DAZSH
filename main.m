clear; 
clc;
addpath data;
addpath utils;

%% load data
load './data/LabelmeZeroShot.mat'
fprintf('Labelme dataset loaded...\n');


%% centralization
fprintf('centralizing data...\n');
I_te = bsxfun(@minus, I_te, mean(I_tr, 1)); I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1)); T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

I_tr = NormalizeFea(I_tr); I_te = NormalizeFea(I_te);
T_tr = NormalizeFea(T_tr); T_te = NormalizeFea(T_te);


%% kernelization
fprintf('kernelizing...\n\n');
[I_tr1,I_te1]=Kernelize(I_tr,I_te); [T_tr1,T_te1]=Kernelize(T_tr,T_te);
I_te = bsxfun(@minus, I_te1, mean(I_tr1, 1)); I_tr = bsxfun(@minus, I_tr1, mean(I_tr1, 1));
T_te = bsxfun(@minus, T_te1, mean(T_tr1, 1)); T_tr = bsxfun(@minus, T_tr1, mean(T_tr1, 1));




run = 10;
map = zeros(run,2);
smap = zeros(run,2);
pre = zeros(run,2);
spre = zeros(run,2);

L_tr_Matrix = L_tr;
L_te_Matrix = L_te;

[a  L_tr] = max(L_tr');
[a  L_te] = max(L_te');

L_tr = L_tr';
L_te = L_te';

for i = 1 : run
    
    
classes = randperm(8);
seenClass = classes(3:end);
unseenClass = classes(1:2);
fprintf('Unseen classes:\n');
 unseenClass

temp = zeros(length(L_te),1);
for ii=1:2
    temp = temp + ismember(L_te,unseenClass(ii));
end
index_unseen_in_te = find(temp==1);
index_seen_in_te = [1:length(L_te)]';
index_seen_in_te(index_unseen_in_te) = [];
% ------------------------------------------
temp = zeros(length(L_tr),1);
for ii=1:2
    temp = temp + ismember(L_tr,unseenClass(ii));
end
index_unseen_in_tr = find(temp==1);
index_seen_in_tr = [1:length(L_tr)]';
index_seen_in_tr(index_unseen_in_tr) = [];

%%

% train data of seen class. same as retrieal data
X1_SR = I_tr(index_seen_in_tr,:);
X2_SR = T_tr(index_seen_in_tr,:);
L_SR = L_tr_Matrix(index_seen_in_tr,:);

X1_SQ = I_te(index_seen_in_te,:);
X2_SQ = T_te(index_seen_in_te,:);
L_SQ = L_te_Matrix(index_seen_in_te,:);

% data split of unseen data
X1_UR = I_tr(index_unseen_in_tr,:);
X2_UR = T_tr(index_unseen_in_tr,:);
L_UR = L_tr_Matrix(index_unseen_in_tr,:);

X1_UQ = I_te(index_unseen_in_te,:);
X2_UQ = T_te(index_unseen_in_te,:);
L_UQ = L_te_Matrix(index_unseen_in_te,:);


S = labelme_attributes(seenClass,:);
%% set parameters
    
options.l = 32;
options.maxItr = 10;
options.lambda = 4;
options.num_samples = 1.5 * options.l;

% unseen class retrieval parameters
options.alpha1  = 1e3;
options.alpha2   = 1e5;
options.beta1 = 1e5; 
options.beta2 = 1e6; 
options.mu1  = 1e5;
options.mu2  = 1e6;
options.gamma  = 1e6;

% % % seen class retrieval parameters
% % options.alpha1  = 1e-3;
% % options.alpha2   = 1e-5;
% % options.beta1 = 1e-5; 
% % options.beta2 = 1e-6; 
% % options.mu1  = 1e-6;
% % options.mu2  = 1e-6;



% % options.gamma  = 1e-8;

%% DAZSH   
    [B1, B2, P1, P2] = DAZSH(X1_SR, X2_SR, L_SR(:,seenClass), S, options);
 
%% Test case 1
% query    : from unseen classes
% retrieval: from unseen classes + seen classes
    
    rBX = [sign(X1_UR * P1);B1];
    qBX = sign(X1_UQ * P1);
    rBY = [sign(X2_UR * P2);B2];
    qBY = sign(X2_UQ * P2);

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nUnseen Text-to-Image Result:\n');
    [result.mAP, result.Precision, result.Recall, result.F1] = evaluation(rBX,qBY,[L_UR; L_SR] * L_UQ');
    map(i,1) = result.mAP;
    pre(i,1) = result.Precision;
    
    
    fprintf('Unseen Image-to-Text Result:\n');
    [result.mAP, result.Precision, result.Recall, result.F1] = evaluation(rBY,qBX,[L_UR; L_SR] * L_UQ');
    map(i,2) = result.mAP;
    pre(i,2) = result.Precision;       
    

% % %% Test case 2
% % % query    : from seen classes
% % % retrieval: from seen classes
% % 
% %     rBX = B1;
% %     qBX = sign(X1_SQ * P1);
% %     rBY = B2;
% %     qBY = sign(X2_SQ * P2);
% % 
% %     rBX = (rBX > 0);
% %     qBX = (qBX > 0);
% %     rBY = (rBY > 0);
% %     qBY = (qBY > 0);
% %     
% %     fprintf('\nSeen Text-to-Image Result:\n');
% %     [result.mAP, result.Precision, result.Recall, result.F1] = evaluation(rBX,qBY,L_SR * L_SQ');
% %     smap(i,1) = result.mAP;
% %     spre(i,1) = result.Precision;
% % 
% %     
% %     fprintf('Seen Image-to-Text Result:\n');
% %     [result.mAP, result.Precision, result.Recall, result.F1] = evaluation(rBY,qBX,L_SR *L_SQ');
% %     smap(i,2) = result.mAP;
% %     spre(i,2) = result.Precision;
end

%% unseen class retrieval
fprintf('Average Unseen Map & Pre Text-to-Image over %d runs: %.4f & %.4f\n', run, mean(map(:,1)),mean(pre(:,1)));
fprintf('Average Unseen Map & Pre Image-to-Text over %d runs: %.4f & %.4f\n', run, mean(map(:,2)),mean(pre(:,2)));
fprintf('-------------------------------------------------------------\n');

% % %% seen class retrieval
% % fprintf('Average Seen   Map & Pre Text-to-Image over %d runs: %.4f & %.4f\n', run, mean(smap(:,1)),mean(spre(:,1)));
% % fprintf('Average Seen   Map & Pre Image-to-Text over %d runs: %.4f & %.4f\n', run, mean(smap(:,2)),mean(spre(:,2)));
% % fprintf('-------------------------------------------------------------\n');



