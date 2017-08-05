clc
clear

datapath = 'e:/prmldata/prnn/';
% datapath = '/Users/n0n/work/data/prmldata/prnn/';

fid = fopen([datapath 'PIMA.TR']);
names = textscan(fid,'%s %s %s %s %s %s %s %s',1);
data = textscan(fid,'%f %f %f %f %f %f %f %s');
fclose(fid);
X1 = [data{1} data{2} data{3} data{4} data{5} data{6} data{7}];
X1 = svmscale(X1,[-1,1],'range','s');
y1 = zeros(size(X1,1),1);
for i = 1:size(X1,1)
    if strcmp(data{8}{i},'Yes')
        y1(i) = 1;
    elseif strcmp(data{8}{i},'No')
        y1(i) = -1;
    end
end

fid = fopen([datapath 'PIMA.TE']);
names = textscan(fid,'%s %s %s %s %s %s %s %s',1);
data = textscan(fid,'%f %f %f %f %f %f %f %s');
fclose(fid);
X2 = [data{1} data{2} data{3} data{4} data{5} data{6} data{7}];
X2 = svmscale(X2,[-1,1],'range','r');
y2 = zeros(size(X2,1),1);
for i = 1:size(X2,1)
    if strcmp(data{8}{i},'Yes')
        y2(i) = 1;
    elseif strcmp(data{8}{i},'No')
        y2(i) = -1;
    end
end

opts.lambda = 0.001;
opts.ktype = 'rbf';
opts.args = [4];
opts.method = 'blassoprobit';
% model = skFLDA(y1,X1,opts);
model = rvmtrain(y1,X1,opts);
yP1 = kpredict(X1,model);
yP2 = kpredict(X2,model);

yP1 = sign(yP1);
yP2 = sign(yP2);

index11 = find(y1==1);
index12 = find(y1==-1);
index13 = find(yP1~=y1);
index21 = find(y2==1);
index22 = find(y2==-1);
index23 = find(yP2~=y2);


figure;

subplot(221);
hold on;
scatter(X1(index11,1), X1(index11,2), 'bx');
scatter(X1(index12,1), X1(index12,2), 'ro');
scatter(X1(index13,1), X1(index13,2), 100, 'ko');
legend('c1', 'c2');
title(['train error num = ' num2str(length(index13))]);

subplot(222);
hold on;
scatter(X2(index21,1), X2(index21,2), 'bx');
scatter(X2(index22,1), X2(index22,2), 'ro');
scatter(X2(index23,1), X2(index23,2), 100, 'ko');
legend('c1', 'c2');
title(['test error num = ' num2str(length(index23))]);

subplot(224);
hold on;
scatter(X1(index11,1), X1(index11,2), 'bx');
scatter(X1(index12,1), X1(index12,2), 'ro');
scatter(model.sv(:,1), model.sv(:,2), 100, 'ko');
legend('c1', 'c2');
title('support vectors');
