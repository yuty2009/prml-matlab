clc
clear

datapath = 'F:\bcicompetition\bci2005\II\';

% 6 by 6  matrix
matrix=['ABCDEF','GHIJKL','MNOPQR','STUVWX','YZ1234','56789_'];
          
subject = 'Subject_A';
fileTrain = [datapath subject '_Train.mat'];
fileTest = [datapath subject '_Test.mat'];
fileTrueLabel = [datapath 'true_labels_a.txt'];
disp(['Loading training dataset for ' subject]);

load(fileTrain);

fp = fopen(fileTrueLabel, 'rt');
line = fgetl(fp);
fclose(fp);
targetTrue = line;

fs = 240;
f1 = 0.1;
f2 = 20;
order = 20;
% [b,a] = butter(order,[f1*2/fs f2*2/fs]);
h  = fdesign.bandpass('N,F3dB1,F3dB2', order, f1, f2, fs);
Hd = design(h, 'butter');

% convert to double precision
Signal = double(Signal);
Flashing = double(Flashing);
StimulusCode = double(StimulusCode);
StimulusType = double(StimulusType);

numChars = 12;
numRepeats = 15;
numSamples = 240;
numTrials = size(Signal,1);
numChannels = size(Signal,3);

featureTrain = [];
labelTrain = zeros(numTrials*numChars,1);
for trial = 1:numTrials
    repeat = zeros(1,numChars);
    for n = 2:size(Signal,2)
        if Flashing(trial,n)==0 && Flashing(trial,n-1)==1
            event = StimulusCode(trial,n-1);
            repeat(event) = repeat(event) + 1;
            signalTrial(event,repeat(event),:,:) = Signal(trial,n-24:n+numSamples-25,:);
        end
    end
    
    featureTrial = zeros(numChars, numRepeats, numSamples, numChannels);
    for i = 1:numChars
        for j = 1:numRepeats
            signalEpoch = squeeze(signalTrial(i,j,:,:));
            % signalEpoch = detrend(signalEpoch);
            % signalFiltered = filter(Hd, signalEpoch);
            featureTrial(i,j,:,:) = reshape(signalEpoch,1,1,numSamples,numChannels);
        end
    end
    featureAveraged = squeeze(mean(featureTrial,2));
    featureTrain = cat(1,featureTrain,featureAveraged);
    
    targetIndex = strfind(matrix,TargetChar(trial));
    targetRow = floor((targetIndex-1)/6) + 1;
    targetCol = targetIndex - (targetRow-1)*6;
    labelTrain((trial-1)*numChars+[targetCol,targetRow+6]) = 1;
end

targetChannel = 11; % Cz
t = [0:numSamples-1].*1000/fs;

% P300 timecourse
indexTarget = find(labelTrain==1);
indexNontarget = find(labelTrain==0);
featureTarget = squeeze(mean(featureTrain(indexTarget,:,:),1));
featureNontarget = squeeze(mean(featureTrain(indexNontarget,:,:),1));

% P300 r square
featureChannel = squeeze(featureTrain(:,:,targetChannel));
Y = labelTrain;
Y(find(Y==0)) = -1;
rr = zeros(numSamples,1);
for i = 1:numSamples;
    % r square = square of correlation coeffient,
    % refer http://www.bci2000.org/wiki/index.php/Glossary#r-squared
    % rr(i) = rsquare(featureChannel(indexTarget,i), featureChannel(indexNontarget,i));
    temp = corrcoef(featureChannel(:,i), Y);
    rr(i) = temp(1,2)^2;
end

% ttest
featureChannel = squeeze(featureTrain(:,:,targetChannel));
index1 = find(Y==1);
index2 = find(Y==-1);
pp = zeros(numSamples,1);
for i = 1:numSamples;
    % r square = square of correlation coeffient,
    % refer http://www.bci2000.org/wiki/index.php/Glossary#r-squared
    % rr(i) = rsquare(featureChannel(indexTarget,i), featureChannel(indexNontarget,i));
    [h p] = ttest2(featureChannel(index1,i), featureChannel(index2,i));
    pp(i) = p;
end

disp('Plotting results');
figure;
subplot(311);
plot(t, featureTarget(:,targetChannel), 'r');
hold on;
plot(t, featureNontarget(:,targetChannel), 'b');
legend('oddball', 'standard');
xlabel('Time after stimulus (ms)');
ylabel('Signal Amplitude');

subplot(312);
plot(t, rr, 'b');
xlabel('Time after stimulus (ms)');
ylabel('r^2 for Standard vs. Oddball');

subplot(313);
plot(t, pp, 'b');
ylim([0 0.1]);
xlabel('Time after stimulus (ms)');
ylabel('p value for Standard vs. Oddball');