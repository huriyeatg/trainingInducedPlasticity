% Convert the spikeDataControl to similar version of training spike Data
load('/Users/Huriye/Documents/Code/4x4training/info_data/spikeDataControl.mat')
load('/Users/Huriye/Documents/Code/4x4training/info_data/control_metrics.mat')

for ind = 1:size(spikeDataControl,2)
    stim = [];
    spikes = [];
    for k = 1: 64 
        spikes = [spikes; double(spikeDataControl{ind}(k).sweeps)];
        for kk = 1:size(spikeDataControl{ind}(k).sweeps,1)
            stim = [stim;spikeDataControl{ind}(k).stim([9, 1, 2,3],1)'];
        end
    end
    spikeData(ind).spikes   = spikes;
    spikeData(ind).stim     = stim;

    resp=sum(spikes(:,1:200),2);
    spon=sum(spikes(:,end-200),2);
    control_metrics(ind).resp = resp;
    control_metrics(ind).spon = spon;
    control_metrics(ind).stim = stim;
    control_metrics(ind).normalised = (resp-spon)./(resp+spon);
    control_metrics(ind).normalisedMean = (mean(spikes(:,1:200),2)-mean(spikes(:,end-200),2))./ (mean(spikes(:,1:200),2) + mean(spikes(:,end-200),2));

end

% Combine the vowel SSA values from PropsControl
load('C:\Users\Huriye\Documents\code\trainingInducedPlasticity\info_data\PropsControl.mat')
for ind = 1:size(control_metrics,2)
    control_metrics(ind).EA = PropsControl(ind).EA;
    control_metrics(ind).UI = PropsControl(ind).UI;
    control_metrics(ind).EU = PropsControl(ind).EU;
    control_metrics(ind).AI = PropsControl(ind).AI;
    control_metrics(ind).EI = PropsControl(ind).EI;
    control_metrics(ind).AU = PropsControl(ind).AU;
    control_metrics(ind).SSRvaluesAll = PropsControl(ind).All;
end


save('/Users/Huriye/Documents/Code/4x4training/info_data/control_metricsStatsMore','control_metrics');
save('/Users/Huriye/Documents/Code/4x4training/info_data/spikeData_control','spikeData','-v7.3')



