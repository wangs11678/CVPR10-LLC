fea = rand(7,10);
options=[];
options.ReducedDim=4;
[eigvector,eigvalue] = PCA(fea,4);
Y = fea*eigvector;