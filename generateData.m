% To generate the SFNAll_2014
SingleUnits_2014
load('/Users/Huriye/Documents/Code/4x4training/info_data/pCoor_trained.mat')


spikeFN = struct;
spikeData = struct;
ind = 1;
for unt = 1:length(S_units)
    animal = S_units{unt,1};
    pents  = S_units{unt,2};
    
    for sh = 1:4
        if sh==1
            ax=3; ay =4; elseif sh==2 % pCoor column shows the shrank position
            ax=5; ay =6; elseif sh==3
            ax=7; ay =8; elseif sh==4
            ax=9; ay =10;
        end
        singleUnitIndex = S_units{unt,2+sh};
        if isnan(S_units{unt,2+sh})
            % no shrank
            disp(['No unit animal:',num2str(animal),' site:',num2str(pents)])
        else
            if sh==1
            path = ['/Users/Huriye/Documents/Code/trainingInducedPlasticity/Data/F0',num2str(animal),...
                '/P',sprintf('%02d',pents)];
            elseif sh==2
                path = ['/Users/Huriye/Documents/Code/trainingInducedPlasticity/Data/F0',num2str(animal),...
                '/P',sprintf('%02d',pents),'b'];
            elseif sh==3
                path = ['/Users/Huriye/Documents/Code/trainingInducedPlasticity/Data/F0',num2str(animal),...
                '/P',sprintf('%02d',pents),'c'];
            elseif sh==4
                path = ['/Users/Huriye/Documents/Code/trainingInducedPlasticity/Data/F0',num2str(animal),...
                '/P',sprintf('%02d',pents),'d'];
            end
            
            pt = dir(fullfile(path, 'clusters.2014*'));
            d = dir(fullfile(path,pt.name,'*data.mat'));
            
            for dd=1:length(d) % each unit/cluster
                %Note the info for each cluster/unit
                spikeFN(ind).fileName   = [path,'/',pt.name, '/', d(dd).name];
                spikeFN(ind).animal     = animal;
                spikeFN(ind).penetration= pents;
                spikeFN(ind).shrank= sh;
                spikeFN(ind).Totalshrank= pCoor(unt,12); % total 
                spikeFN(ind).cluster    = str2double(d(dd).name(9:10)); % the unit
                spikeFN(ind).single     = ismember(spikeFN(ind).cluster,singleUnitIndex);
                % add the depth data
                t =load ([path,'/',pt.name,'/',d(dd).name(1:10),'.event_shape_mean.mat']);
                [~, elect]=max(max(abs(t.event_shape_mean)));
                spikeFN(ind).depth      =elect;
                clear t
                % add the coordinates values
                spikeFN(ind).xcoor     = pCoor(unt,ax);
                spikeFN(ind).ycoor     = pCoor(unt,ay);
                spikeFN(ind).field     = pCoor(unt,11);
                
                spikeData(ind).index    = ind;%index in spikefn, should be same
                spikeData(ind).filename = spikeFN(ind).fileName;
                spikeData(ind).spikes   = NaN;
                spikeData(ind).stim     = NaN;

                % load data
                load([path,'/',pt.name,'/',d(dd).name]);
                if size(data.set,2)==1 && size(data.set(1).stim_params,2)<2
                    spikeFN(ind).Recording ='Only one stim info';
                    % Recording error!
                elseif size(data.set,2)==1 && size(data.set(1).stim_params,2)>1
                    spikeFN(ind).Recording =['StimNo', num2str(size(data.set(1).stim_params,2))];
                    % Recording error!
                elseif size(data.set,2)>2
                    % Get the tone stim order and data
                     stim=[];
                     for ii=1:length(data.set)
                         stim=[stim;data.set(ii).stim_params.Freq_Hz,...
                             data.set(ii).stim_params.Azim_deg];
                     end
                    tDat=data.set(find(stim(:,1)~=-1)); 
                    vDat=data.set(find(stim(:,2)~=-1));
                    %add the BF
                    if size(tDat,2)>1
                        runBF
                    
                    spikeFN(ind).BF    = bf;
                    spikeFN(ind).Q10   = Q10;
                    spikeFN(ind).Q30   = Q30;
                    else
                     spikeFN(ind).Recording = 'Only vowel data';   
                    spikeFN(ind).BF    = NaN;
                    spikeFN(ind).Q10   = NaN;
                    spikeFN(ind).Q30   = NaN;
                    end
                   
                    clear tDat 
                    % Get the vowel data stim order
                    spikes=[];stim=[];
                    %first check the sweep lengths
                    for ii=1:length(vDat)
                        sL(ii)=round(vDat(ii).length_signal_ms);
                    end
                    for ii=1:length(vDat)
                        m=zeros(size(vDat(ii).repeats,2),round(vDat(ii).length_signal_ms));
                        Vind=sub2ind(size(m),vDat(ii).spikes.repeat_id',round(vDat(ii).spikes.t)');
                        m(Vind)=1;
                        spikes=[spikes;m(:,1:min(sL))];
                        s=[vDat(ii).stim_params.Azim_deg,vDat(ii).stim_params.Fundamental_Hz,...
                            vDat(ii).stim_params.Formant_1_Hz,vDat(ii).stim_params.Formant_2_Hz];
                        stim=[stim;repmat(s,size(m,1),1)];
                        
                    end
                    [s,i]=sortrows(stim,[2,1,3,4]);%sort by stimulus type
                    Spikes=spikes(i,:);

                    % save event times and spikes in a different struct for future
                    % reference
                    spikeData(ind).index    = ind;%index in spikefn
                    spikeData(ind).filename = spikeFN(ind).fileName;
                    spikeData(ind).spikes   = Spikes;
                    spikeData(ind).stim     = stim;
                    
                    %Check whether it is driven, if so generate SSs values with ANOVA
                    resp=sum(Spikes(:,1:200),2);
                    spon=sum(Spikes(:,end-200),2);
                    spikeFN(ind).resp = resp;
                    spikeFN(ind).spon = spon;
                    spikeFN(ind).stim = stim;
                    spikeFN(ind).normalised = (resp-spon)./(resp+spon);
                    spikeFN(ind).normalisedMean = (mean(Spikes(:,1:200),2)-mean(Spikes(:,end-200),2))./ (mean(Spikes(:,1:200),2) + mean(Spikes(:,end-200),2));


                    %run a ttest now to see if the difference in rate is ~= zero
                    [~,p] = ttest([resp-spon]);
                    spikeFN(ind).driven = p;
                    spikeFN(ind).pvaluesDriven = p;
                    clear Data
                    if p<0.05
                        us=unique(stim,'rows');
                        for uu=1:length(us)
                            f=find(stim(:,1)==us(uu,1) & stim(:,2)==us(uu,2) & stim(:,3)==us(uu,3) & stim(:,4)==us(uu,4) );
                            Data(uu).stim =us(uu,:)';
                            Data(uu).sweeps=sparse(Spikes(f,:));
                            Data(uu).sweeplength=length(Spikes);
                        end
                        AnovaSSs_2014;
                    end
                    ind = ind+1;
                end
                
                disp(['Animal:',num2str(animal),' Site:',num2str(pents),...
                    ' Sh:',num2str(sh),' ',num2str(dd),'/',num2str(size(d,1))])
            end

        end
    end
end
trained_metrics = spikeFN;
save('/Users/Huriye/Documents/Code/4x4training/info_data/trained_metrics','trained_metrics');
save('/Users/Huriye/Documents/Code/4x4training/info_data/spikeData_trained','spikeData','-v7.3');
           