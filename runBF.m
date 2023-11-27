spikes=[];stim=[];
%first check the sweep lengths:
for ii=1:length(tDat);
    sL(ii)=round(tDat(ii).length_signal_ms);
end
for ii=1:length(tDat);
    m=zeros(size(tDat(ii).repeats,2),round(tDat(ii).length_signal_ms));
    % extract the indexes for the spikes;
    Vind=sub2ind(size(m),tDat(ii).spikes.repeat_id',round(tDat(ii).spikes.t)');
    m(Vind)=1;
    spikes=[spikes;m(:,1:min(sL))]; %incase there are different sweep lengths - just choose the shortest
    s=[tDat(ii).stim_params.level_dB,tDat(ii).stim_params.Freq_Hz];
    stim=[stim;repmat(s,size(m,1),1)];
end
SpikeDataTone{1,3}=spikes;
SpikeDataTone{1,4}=stim;
nlevels=unique(stim(:,1));
nfreqs=unique(stim(:,2));
clear nlevs
for nn=1:length(nfreqs)
    f=find(stim(:,2)==nfreqs(nn));
    nlevs(nn)=length(unique(stim(f,1)));
end
f1 = find(nlevs==round(median(nlevs))); %f1 = find(nlevs==6);
f1 = [1,f1];
freqs1=nfreqs(f1);
% f2 = find(nlevs==max(nlevs));
% f2 = [1,f2];
% freqs2=nfreqs(f2);

spikes=[];stim=[];
for ii=2:length(freqs1)
    f=find(SpikeDataTone{1,4}(:,2)==freqs1(ii));
    spikes=[spikes;SpikeDataTone{1,3}(f,:)];
    stim=[stim;SpikeDataTone{1,4}(f,:)];
end
f=find(SpikeDataTone{1,4}(:,2)==freqs1(1) & SpikeDataTone{1,4}(:,1)>=30);
spikes=[spikes;SpikeDataTone{1,3}(f,:)];
stim=[stim;SpikeDataTone{1,4}(f,:)];
% figure(1);clf;imagesc(spikes);xlim([0 400]);
% figure(2);clf; plot(sum(spikes)); title(pents)
nlevel=unique(stim(:,1));
nfreq=unique(stim(:,2));
spon1=sum(spikes(:,end-200:end),2);
resp1=sum(spikes(:,1:200),2);
dSpikes=[resp1,stim(:,2),stim(:,1)];
sSpikes=[spon1,stim(:,2),stim(:,1)];
[h,p]=ttest(resp1,spon1);
spikeFN(ind).ToneDriven = p;
[~,bf,Q10,Q30,~,~]=FRAanalyse(dSpikes,sSpikes);
% figure(4);clf;imagesc(fra); axis xy;colorbar; title(bf)
