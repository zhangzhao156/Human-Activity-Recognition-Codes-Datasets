clear;

base='../../dataset/';
% Subject 1
% training
sadl1 = load([base 'S1-ADL1.dat']);
sadl2 = load([base 'S1-ADL2.dat']);
sadl3 = load([base 'S1-ADL3.dat']);
sdrill = load([base 'S1-Drill.dat']);
% test
sadl4 = load([base 'S1-ADL4.dat']);
sadl5 = load([base 'S1-ADL5.dat']);

[training1,test1]=tarrange(4,sadl1,sadl2,sadl3,sdrill,sadl4,sadl5);
selectedCol = [1:46 51:59 64:72 77:85 90:98 103:134 244 250];
training1 = training1(:,selectedCol);
test1 = test1(:,selectedCol);
%save('D:\work\matlab\benchmark\tejaswi (copy)\subject1\UCIbegin1.mat','training1','test1');
save('UCIbegin1.mat','training1','test1');
clear training1;
clear test1;


% Subject 2
% training
sadl1 = load([base 'S2-ADL1.dat']);
sadl2 = load([base 'S2-ADL2.dat']);
sadl3 = load([base 'S2-ADL3.dat']);
sdrill = load([base 'S2-Drill.dat']);
% test
sadl4 = load([base 'S2-ADL4.dat']);
sadl5 = load([base 'S2-ADL5.dat']);

[training2,test2]=tarrange(4,sadl1,sadl2,sadl3,sdrill,sadl4,sadl5);
training2 = training2(:,selectedCol);
test2 = test2(:,selectedCol);
save('UCIbegin2.mat','training2','test2');
clear training2;
clear test2;

% Subject 3
% training
sadl1 = load([base 'S3-ADL1.dat']);
sadl2 = load([base 'S3-ADL2.dat']);
sadl3 = load([base 'S3-ADL3.dat']);
sdrill = load([base 'S3-Drill.dat']);
% test
sadl4 = load([base 'S3-ADL4.dat']);
sadl5 = load([base 'S3-ADL5.dat']);

[training3,test3]=tarrange(4,sadl1,sadl2,sadl3,sdrill,sadl4,sadl5);
training3 = training3(:,selectedCol);
test3 = test3(:,selectedCol);
save('UCIbegin3.mat','training3','test3');
clear training3;
clear test3;

% Subject 4
% training
sadl1 = load([base 'S4-ADL1.dat']);
sadl2 = load([base 'S4-ADL2.dat']);
sadl3 = load([base 'S4-ADL3.dat']);
sdrill = load([base 'S4-Drill.dat']);
% test
sadl4 = load([base 'S4-ADL4.dat']);
sadl5 = load([base 'S4-ADL5.dat']);

[training4,test4]=tarrange(4,sadl1,sadl2,sadl3,sdrill,sadl4,sadl5);
selectedCol = [1 38:46 51:59 64:72 77:85 90:98 244 250]; % Motion Jacket only
training4 = training4(:,selectedCol);
test4 = test4(:,selectedCol);
save('UCIbegin4.mat','training4','test4');

clear;
load('UCIbegin1.mat');
load('UCIbegin2.mat');
load('UCIbegin3.mat');
load('UCIbegin4.mat');

