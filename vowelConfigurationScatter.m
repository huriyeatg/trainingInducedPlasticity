function  vowelConfiguration_scatter (save_path)

%% load extracted info
% load trained metrics
load trained_metrics
BFcorrection
CoordinateCorrection
load gpointsTrained

%load control metrics used in Bizley et al., 09
load control_metrics
load PropsCNT

%% Load the data
ind = [];
for kk=1:length(trained_metrics)
    if~isempty(trained_metrics(kk).EU) & ~isnan(trained_metrics(kk).EU)&...
            trained_metrics(kk).SSRvalues_Anovatable{2,7}<0.05
        ind = [ind;kk];
    end
end
trained_metrics = trained_metrics(ind);

% get the data
for dsa = 1:3
    if dsa==1 % for control
        DataMain = PropsCNT;
        vf =struct2cell(control_metrics);
        FF = struct('field',vf(12,:));
        An = struct('animal',vf(1,:));
        DataMain = catstruct(DataMain,FF,An);
    elseif dsa==2 % discrimination animals only
        clear DataMain vf ind
        vf = struct2cell(trained_metrics);
        ind = find( cell2mat(vf(2,:))==833 | cell2mat(vf(2,:))==943 | cell2mat(vf(2,:))==963)';
        DataMain = trained_metrics(ind);
    elseif  dsa==3 % detection animal only
        clear DataMain vf ind
        vf = struct2cell(trained_metrics);
        ind = find( cell2mat(vf(2,:))==848 | cell2mat(vf(2,:))==832)';
        DataMain = trained_metrics(ind);
    end
    
    datC=[]; dat=[]; t =1;
    for k=1:length(DataMain)
        if~isempty(DataMain(k).EU) & ~isnan(DataMain(k).EU)
            datC=[datC;DataMain(k).animal,DataMain(k).field]; %ideally would be animal, cortical field
            
            dat{t,1}=DataMain(k).EU(1)*100;
            dat{t,2}=DataMain(k).AI(1)*100;
            dat{t,3}=DataMain(k).EA(1)*100;
            dat{t,4}=DataMain(k).UI(1)*100;
            dat{t,5}=DataMain(k).EI(1)*100;
            dat{t,6}=DataMain(k).AU(1)*100;
            
            dat{t,7}=DataMain(k).EU(2)*100;
            dat{t,8}=DataMain(k).AI(2)*100;
            dat{t,9}=DataMain(k).EA(2)*100;
            dat{t,10}=DataMain(k).UI(2)*100;
            dat{t,11}=DataMain(k).EI(2)*100;
            dat{t,12}=DataMain(k).AU(2)*100;
            
            dat{t,13}=DataMain(k).EU(3)*100;
            dat{t,14}=DataMain(k).AI(3)*100;
            dat{t,15}=DataMain(k).EA(3)*100;
            dat{t,16}=DataMain(k).UI(3)*100;
            dat{t,17}=DataMain(k).EI(3)*100;
            dat{t,18}=DataMain(k).AU(3)*100;
            t = t+1;
        end
    end
    dd(dsa).dat= dat;
    dd(dsa).datC= datC;
end
%% Calculate the mean
for dsa=1:3
    
    dat = dd(dsa).dat;
    datC = dd(dsa).datC;
    dat=cell2mat(dat);
    
    xp1=[];
    xp1 = [dat(:,13),dat(:,14),dat(:,15),dat(:,16),dat(:,17),dat(:,18)];%for timbre
    
    MeanTM(dsa,:)= nanmean(xp1);
    
end

%% Get pairs
e = [2058, 730]; 
a = [1551, 936];
u = [1105, 460];
i = [2761, 437];

EU = abs(e-u);
AI = abs(a-i);
EA = abs(e-a);
UI = abs(u-i);
EI = abs(e-i);
AU = abs(a-u);

F2 = [EU(1),AI(1),EA(1),UI(1),EI(1),AU(1)];
F1 = [EU(2),AI(2),EA(2),UI(2),EI(2),AU(2)];

%% Plot the values
cc= colormap;
cc = cc(1:10:size(F1,2)*10,:);
figure
subplot(3,2,1)
scatter(F1,MeanTM(1,:),[],cc)
subplot(3,2,3)
scatter(F1,MeanTM(2,:),[],cc)
subplot(3,2,5)
scatter(F1,MeanTM(3,:),[],cc)

subplot(3,2,2)
scatter(F2,MeanTM(1,:),[],cc)
subplot(3,2,4)
scatter(F2,MeanTM(2,:),[],cc)
subplot(3,2,6)
scatter(F2,MeanTM(3,:),[],cc)
