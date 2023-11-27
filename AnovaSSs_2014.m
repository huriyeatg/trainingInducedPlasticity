[~, ~, ~, table, ~, ~] = AnovaVowStimWithBinsArtifactRejectNEW (Data);

SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).SSRvaluesAll = a;
spikeFN(ind).SSRvalues_Anovatable = table;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt

[trainedAnova,untrainedAnova]=AnovaVowStimWithBinsArtifactRejectNEWtrainedVUntrainedEU(Data);


table=trainedAnova;
SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).EU = a;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt

table=untrainedAnova;
SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
 a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).AI = a;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt 
clear untrainedAnova trainedAnova
[trainedAnova2,untrainedAnova2]=AnovaVowStimWithBinsArtifactRejectNEWtrainedVUntrainedEA2(Data);


table=trainedAnova2;
SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
 a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).EA = a;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt

table=untrainedAnova2;
SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
 a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).UI = a;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt
clear untrainedAnova trainedAnova
[trainedAnova,untrainedAnova]=AnovaVowStimWithBinsArtifactRejectNEWtrainedVUntrainedEI(Data);


table=trainedAnova;
SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).EI = a;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt

table=untrainedAnova;
SSp=(table{5,2}-(table{9,5}*table{5,3}))/(table{10,2}-table{2,2});
SSt=(table{3,2}-(table{9,5}*table{3,3}))/(table{10,2}-table{2,2});
SSa=(table{4,2}-(table{9,5}*table{4,3}))/(table{10,2}-table{2,2});
SSat=(table{6,2}-(table{9,5}*table{6,3}))/(table{10,2}-table{2,2});
SSap=(table{7,2}-(table{9,5}*table{7,3}))/(table{10,2}-table{2,2});
SSpt=(table{8,2}-(table{9,5}*table{8,3}))/(table{10,2}-table{2,2});
totalStimSS=sum([SSa,SSp,SSt,SSat,SSap,SSpt]);
 a= [SSa,SSp,SSt,SSap,SSat,SSpt,totalStimSS];
spikeFN(ind).AU = a;
clear table SSa SSt SSp SSat SSpt SSap totalStimSS sumInt ratioInt
clear untrainedAnova trainedAnova