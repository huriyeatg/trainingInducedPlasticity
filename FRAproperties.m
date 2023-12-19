clear bf pf freqspace bfvals threshold m pfvals q I Q10 Q30 bandwidth bandwidth30

%extracting BF as defined by lowest freq to evoke a response.

[I]=find(bounds==(min(bounds)));% find the freqs with the lowest intensity.

freqspace=linspace(min(log(freqs)),max(log(freqs)),nfreqs*2);
bfvals=mean(freqspace(I*2));
bf=exp(bfvals)/1000;
u=unique(freqs);

% text(mean(I),min(bounds),'o','Fontweight','bold','fontsize',14)
%extract threshold as the minimum intensity to evoke a response
if min(bounds)<=length(levels) & min(bounds)>0
threshold=levels(min(bounds));

%extract BF as defined by freq with max response.

% m=find(suspikes==max(suspikes));
% pfvals=mean(freqspace(m*2));
% pf=exp(pfvals)/1000;
% text(mean(m),nlevels,'x','Fontweight','bold','fontsize',14)

%Q10 the width of the tuning curve 10dB above threshold
q=(min(bounds)+1);
[I]=find(bounds<=q);
index=1;
ind=1;
bw=[];

if length(I)==1
    bw=[I I];
else
    for ii=1:length(I)
        if ii==length(I)
            if I(end)-1~=I(end-1)
                bw=[bw;I(end) I(end)];
            elseif I(end)-1==I(end-1)
                bw=[bw;I(index) I(end)];
            end
            
        elseif I(ii+1)==I(ii)+1;
            
        elseif I(ii+1)~=I(ii)
            bw=[bw; I(index) I(ii)];
            index=ii+1;
            ind=ind+1;
        end
    end
end
bandwidth=[];
for ii=1:size(bw,1)
    
    if bw(ii,1)==bw(ii,2)
        if bw(ii,1)==nfreqs
            bandwidth=[bandwidth;exp(freqspace(bw(ii,1)*2))/1000-exp(freqspace(bw(ii,1)*2-2))/1000];
        else
            bandwidth=[bandwidth;exp(freqspace(bw(ii,1+1)*2))/1000-exp(freqspace(bw(ii,1)*2-1))/1000];
        end
    else
        bandwidth=[bandwidth;exp(freqspace(bw(ii,2)*2))/1000-exp(freqspace(bw(ii,1)*2))/1000];
    end
end  
bandwidth=sum(bandwidth);
Q10=bf/bandwidth;
%repeat for Q30

%Q30 the width of the tuning curve 10dB above threshold
q=(min(bounds)+3);
[I]=find(bounds<=q);
index=1;
ind=1;
bw=[];
if length(I)==1
    bw=[I I];
else
    for ii=1:length(I)
        if ii==length(I)
            if I(end)-1~=I(end-1)
                bw=[bw;I(end) I(end)];
            elseif I(end)-1==I(end-1)
                bw=[bw;I(index) I(end)];
            end
            
        elseif I(ii+1)==I(ii)+1;
            
        elseif I(ii+1)~=I(ii)
            bw=[bw; I(index) I(ii)];
            index=ii+1;
            ind=ind+1;
        end
    end
end
bandwidth30=[];
for ii=1:size(bw,1)
    if bw(ii,1)==bw(ii,2)
        if bw(ii,1)==nfreqs
            bandwidth=[bandwidth;exp(freqspace(bw(ii,1)*2))/1000-exp(freqspace(bw(ii,1)*2-2))/1000];
        else
            bandwidth=[bandwidth;exp(freqspace(bw(ii,1+1)*2))/1000-exp(freqspace(bw(ii,1)*2-1))/1000];
        end
    else
        bandwidth30=[bandwidth;exp(freqspace(bw(ii,2)*2))/1000-exp(freqspace(bw(ii,1)*2))/1000];
    end
end  
bandwidth30=sum(bandwidth30);
Q30=bf/bandwidth30;
else
    threshold=NaN;
    bandwidth30=NaN;bf=NaN;
    pf=NaN; bfvals=NaN; q=NaN; I=NaN; Q10=NaN; Q30=NaN; bandwidth=NaN;
end