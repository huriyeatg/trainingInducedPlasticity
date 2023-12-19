function [boundary,bf,Q10,Q30,Th,data]=FRAanalyse(driven,spon)

%function to plot an FRA and then mark the response boundary onto it.
%returns the outputs best freq, peak freq, Q10 and Q30 and will plot the
%FRA with the bounds on.
%[bounds,bf,Q10,Q30,Th,spikes]=FRAanalyse(file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear spikes ss re r freqs level nfreqs nlevels spike boundary srate
fsz=10;
%load the data

r=driven;
ss=spon;
%make r into a 3 column vector with freqs in column 2, levels in 3
freqs=unique(r(:,2));
levels=unique(r(:,3));

nfreqs=length(freqs);
nlevels=length(levels);%number of freqs/levels


% we think that there might be multiple tone files merged together here
% which we need to separate.... usual tone files have ~18 frequencies.

% if nfreqs>18 %we suspect multiple file types
%     lets run for low sound levels first of all;
%     nlevels=6;
%     alllevels=levels;
%     levels=alllevels(3:8);
%     spikes=[];stim=[];
%     ind2use=[];
%     for ff=1:nfreqs
%         for ll=1:nlevels
%             f=find(r(:,2)==freqs(ff) & r(:,3)==levels(ll));
%             if ~isempty(f)
%                 ind2use=[ind2use,1];
%                 spikes=[spikes;mean(r(find(r(:,2)==freqs(ff) & r(:,3)==levels(ll)),1))];
%                 stim=[stim;freqs(ff),levels(ll)];
%             else
%                 ind2use=[ind2use,0];
%                 stim=[stim;freqs(ff),levels(ll)];
%             end
%         end
%     end
%     f=find(ind2use);
%     stim=stim(f,:);
%     
%     f2use=[];
%     for ff=1:nfreqs
%         f=find(stim(:,1)==freqs(ff));
%         if length(f)==6
%             f2use=[f2use;ff];
%         end
%     end
%     
%     freqs1=freqs(f2use);
%     stim1=[];spikes1=[];
%     for ff=1:length(freqs1)
%         f=find(stim(:,1)==freqs1(ff));
%         stim1=[stim1;stim(f,:)];
%         spikes1=[spikes1;spikes(f)];
%     end
%     
%     nfreqs1=length(freqs1);
%     
%     [rs,i]=sortrows(stim1,[1,2]);
%     spikes1=spikes1(i);
%     spike1=reshape(spikes1,nlevels,nfreqs1); %form a matrix of spike counts ordered by freq and level
%     
%     s=[0.25 0.5 0.25;0.5 1 0.5; 0.25 0.5 0.25];
%     s = [1,2,1;2,4,2;1,2,1]; %create a sliding window
%     s = s/sum(s(:)); %normalise
%     [m,n]=size(spike1);
%     p(:,1)=spike1(:,1);
%     p(:,2:n+1)=spike1;
%     p(:,n+2)=spike1(:,n);
%     p(2:m+1,:)=p(1:m,:);
%     p(1,:)=p(1,:);
%     p(m+2,:)=p(m,:);
%     spikes1=conv2(p,s,'same'); %smooth
%     
%     spikes1=spikes1(2:m+1,2:n+1);%remove the padding
%     
%     
%     
%     
%     NB Spikerate produces the total in that time period
%     
%     calculate the mean spontaneous rate and calculate a threshold based on spontaneus
%     rate+20% of max rate during stimulus-spontaneus rate.
%     
%     srate=mean(ss(:,1))+1/5*(max(max(spikes1)));%-mean(ss)));
%     
%     if srate<1;
%         srate=1;
%     end
%     spikes=spikes1;
%     nfreqs=nfreqs1;
%     
%     bounds=[];
%     for ii=1:nfreqs
%         for jj=1:nlevels
%             if spikes(jj,ii)<srate & jj <nlevels
%                 no response, move on to next level
%             elseif spikes(jj,ii)>=srate & jj==1
%                 response at lowest level, move onto next to see if real
%                 elseif jj<5 & spikes(jj+4,ii)<srate & spikes(jj+5)<srate
%             elseif spikes(jj,ii)>=srate & jj>1 & spikes(jj-1,ii)>=srate & jj<nlevels-2 & spikes(jj+1,ii)>=srate & spikes(jj+2,ii)>srate
%                 bounds(ii)=jj-1;
%                 break
%                 if the response is there at 4 (for finer sampling) consequetive levels, set bounds
%                 for the level it first appeared at
%             elseif spikes(jj,ii)>=srate & jj==nlevels-1 & spikes(jj-1,ii)>=srate & jj>nlevels-2 & spikes(jj+1,ii)>=srate
%                 bounds(ii)=jj-1;
%                 break
%             elseif spikes(jj,ii)>=srate & jj==nlevels & spikes(jj-1,ii)>=srate
%                 bounds(ii)=jj-1;
%                 break
%                 if this is the highest level and there is a response here and
%                 at level-1 set bounds to level first appeared at
%             elseif spikes(jj,ii)>=srate & jj==nlevels
%                 bounds(ii)=nlevels;
%                 if there is only a response at the highest level take this
%                 level
%             elseif jj==nlevels & spikes(jj,ii)<srate % & spikes(jj-1,ii)<srate
%                 bounds(ii)=nlevels+1;
%                 if there are no responses then bounds =nlevels+1;
%             end
%         end
%     end
%     
%     for ii=2:length(bounds)-1
%         if bounds(ii-1)==nlevels+1 & bounds(ii+1)==nlevels+1 & bounds(ii)<4
%             bounds(ii)=nlevels+1;
%         end
%     end
%     plot the FRA
%     figure
%     
%     imagesc(spikes);
%     axis xy
%     
%     set the axis values
%     tickvals=linspace(min(log(freqs)),max(log(freqs)),4); % this caculatesvalues for 6 ticks, logarithmically between the min and max xvars
%     ticks2use=linspace(1,nfreqs,4); % 6 ticks
%     freqs2use=exp(tickvals)/1000; %this is the exp of the logged xvars so that the ticks are real nos, in KHz.
%     set(gca,'XTick',ticks2use)
%     set(gca,'XTickLabel',num2str(round(freqs2use(:)),2))%num2string means can specify the no of sig figs (2) for the labels.
%     set(gca,'FontSize',fsz);%,'FontWeight','bold');
%     set(gca,'YTickLabel',num2str(unique(levels(:))));
%     la(1)=Xlabel('Freq (KHz)','FontSize',fsz);%'FontWeight','bold');
%     la(2)=Ylabel('Level (dB)','FontSize',fsz);%,'FontWeight','bold');
%     
%     fsz=16;
%     b1=colorbar;  c=get(b1,'ylim'); c(1)=0; set(b1,'ylim',c) ;
%     set(b1,'FontSize',fsz);
%     set(b1,'position',[ 0.81    0.14    0.03   0.755], 'fontsize',fsz);
%     t=get(b1,'ylabel');
%     set(t,'string','spikes per presentation','fontsize',fsz) ;
%     hold on
%     add the line
%     hold on
%     plot(bounds,'color','white','LineWidth',[2])
%     
%     try making an index here and summing accross freqs to see
%     
%     Th=(min(bounds)-1)*10;
%     [a]=find(bounds==min(bounds));
%     u=unique(freqs);
%     bfs=log(u(a));
%     bf=exp(mean(bfs));
%     data=spikes;
%     hold off
%     FRAproperties
%     boundary=bounds;
%     
    
    
    
    spikes=[];stim=[];
    for ff=1:nfreqs
        for ll=1:nlevels
            spikes=[spikes;mean(r(find(r(:,2)==freqs(ff) & r(:,3)==levels(ll)),1))];
            stim=[stim;freqs(ff),levels(ll)];
        end
    end
    
    [rs,i]=sortrows(stim,[1,2]);
    spikes=spikes(i);
    spike=reshape(spikes,nlevels,nfreqs); %form a matrix of spike counts ordered by freq and level
    
    s=[0.25 0.5 0.25;0.5 1 0.5; 0.25 0.5 0.25];
    %s = [1,2,1;2,4,2;1,2,1]; %create a sliding window
    s = s/sum(s(:)); %normalise
    [m,n]=size(spike);
    p(:,1)=spike(:,1);
    p(:,2:n+1)=spike;
    p(:,n+2)=spike(:,n);
    p(2:m+1,:)=p(1:m,:);
    p(1,:)=p(1,:);
    p(m+2,:)=p(m,:);
    spikes=conv2(p,s,'same'); %smooth
    
    spikes=spikes(2:m+1,2:n+1);%remove the padding
    
    
    
    
    %NB Spikerate produces the total in that time period
    
    %calculate the mean spontaneous rate and calculate a threshold based on spontaneus
    %rate+20% of max rate during stimulus-spontaneus rate.
    
    srate=mean(ss(:,1))+1/5*(max(max(spikes)));%-mean(ss)));
    
    % if srate<1;
    %     srate=1;
    % end
    
    bounds=[];
    for ii=1:nfreqs
        for jj=1:nlevels
            if spikes(jj,ii)<srate & jj <nlevels
                %no response, move on to next level
            elseif spikes(jj,ii)>=srate & jj==1
                %response at lowest level, move onto next to see if real
                % elseif jj<5 & spikes(jj+4,ii)<srate & spikes(jj+5)<srate
            elseif spikes(jj,ii)>=srate & jj>1 & spikes(jj-1,ii)>=srate & jj<nlevels-2 & spikes(jj+1,ii)>=srate & spikes(jj+2,ii)>srate
                bounds(ii)=jj-1;
                break
                %if the response is there at 4 (for finer sampling) consequetive levels, set bounds
                %for the level it first appeared at
            elseif spikes(jj,ii)>=srate & jj==nlevels-1 & spikes(jj-1,ii)>=srate & jj>nlevels-2 & spikes(jj+1,ii)>=srate
                bounds(ii)=jj-1;
                break
            elseif spikes(jj,ii)>=srate & jj==nlevels & spikes(jj-1,ii)>=srate
                bounds(ii)=jj-1;
                break
                %if this is the highest level and there is a response here and
                %at level-1 set bounds to level first appeared at
            elseif spikes(jj,ii)>=srate & jj==nlevels
                bounds(ii)=nlevels;
                %if there is only a response at the highest level take this
                %level
            elseif jj==nlevels & spikes(jj,ii)<srate % & spikes(jj-1,ii)<srate
                bounds(ii)=nlevels+1;
                %if there are no responses then bounds =nlevels+1;
            end
        end
    end
    
    for ii=2:length(bounds)-1
        if bounds(ii-1)==nlevels+1 & bounds(ii+1)==nlevels+1 & bounds(ii)<4
            bounds(ii)=nlevels+1;
        end
    end
    %plot the FRA
    %figure
%     
%     imagesc(spikes);
%     axis xy
%     
%     %set the axis values
    tickvals=linspace(min(log(freqs)),max(log(freqs)),4); % this caculatesvalues for 6 ticks, logarithmically between the min and max xvars
    ticks2use=linspace(1,nfreqs,4); % 6 ticks
    freqs2use=exp(tickvals)/1000; %this is the exp of the logged xvars so that the ticks are real nos, in KHz.
%     set(gca,'XTick',ticks2use)
%     set(gca,'XTickLabel',num2str(round(freqs2use(:)),2))%num2string means can specify the no of sig figs (2) for the labels.
%     set(gca,'FontSize',fsz);%,'FontWeight','bold');
%     set(gca,'YTickLabel',num2str(unique(levels(:))));
    %la(1)=Xlabel('Freq (KHz)','FontSize',fsz);%'FontWeight','bold');
    %la(2)=Ylabel('Level (dB)','FontSize',fsz);%,'FontWeight','bold');
    
%     fsz=16;
%     %b1=colorbar;  c=get(b1,'ylim'); c(1)=0; set(b1,'ylim',c) ;
%     %set(b1,'FontSize',fsz);
%     %set(b1,'position',[ 0.81    0.14    0.03   0.755], 'fontsize',fsz);
%     %t=get(b1,'ylabel');
%     %set(t,'strin g','spikes per presentation','fontsize',fsz) ;
%     hold on
%     %add the line
%     hold on
%     plot(bounds,'color','white','LineWidth',[2])
%     
    %try making an index here and summing accross freqs to see
    
    Th=(min(bounds)-1)*10;
    [a]=find(bounds==min(bounds));
    u=unique(freqs);
    bfs=log(u(a));
    bf=exp(mean(bfs));
    data=spikes;
%     hold off
    FRAproperties
    boundary=bounds;
    %figure;
    %plot(suspikes,'color','black','LineWidth',[2]);
    

