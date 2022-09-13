function [KTrain, KTest] = Kernelize(Train,Test)
    [n,~]=size(Train);
    [nT,~]=size(Test);
%     n_anchor=1000;
    n_anchor=300;
    
    anchor=Train(randsample(n,n_anchor),:);
   
    KTrain = sqdist1(Train',anchor');
    sigma = mean(mean(KTrain,2));
    KTrain = exp(-KTrain/(2*sigma));  
    mvec = mean(KTrain);
    KTrain = KTrain-repmat(mvec,n,1);
    
    KTest = sqdist1(Test',anchor');
    KTest = exp(-KTest/(2*sigma));
    KTest = KTest-repmat(mvec,nT,1);
end