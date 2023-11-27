%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%% Number of Cells Driven by Stimuli Features
% % 
% % load DataAll
% % 3 untrained, 4 trained
load AnovaResultTra
load SFN

Azi=[];
Pitch=[];
Timbre=[];
AP=[];
AT=[];
PT=[];
APT=[];
Total=[];
No=[];

data=zeros(803,10);
for k=1:length(AnovaResultTra) 
    table=AnovaResultTra{k,1};
    
    if ~isempty(table)   & table{4,7}<0.001 &  table{5,7}>0.001 & table{3,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001  %azimuth
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,1)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2
            data(k,1)=1;
         end
    end
    if ~isempty(table)  & table{5,7}<0.001 &  table{4,7}>0.001 & table{3,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001% %pitch
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,2)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2
            data(k,2)=1;
         end
    end
    
    if ~isempty(table)  & table{3,7}<0.001 & table{5,7}>0.001 & table{4,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001%timbre
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,3)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2 
            data(k,3)=1;
         end
    end
    if ~isempty(table)  & table{4,7}<0.001 &  table{5,7}<0.001 & table{3,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833) %& table{3,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001  %azimuth-pitch
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,4)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2
            data(k,4)=1;
         end
    end
    
    if ~isempty(table)   & table{4,7}<0.001 & table{3,7}<0.001 & table{5,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{5,7}>0.001 %& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001% %azimuth-timre
        if  SFN(k).single==0 & SFN(k).Q==3
            data(k,5)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2 
            data(k,5)=1;
        end
    end
    if ~isempty(table)  & table{3,7}<0.001 & table{5,7}<0.001 & table{4,7}>0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{4,7}>0.001 %& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001%timbre-pitch
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,6)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2
            data(k,6)=1;
         end
    end
    
    if ~isempty(table)    & table{3,7}<0.001 & table{4,7}<0.001 & table{5,7}<0.001 & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)%& table{6,7}>0.001 &  table{7,7}>0.001 & table{8,7}>0.001%timbre
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,7)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2
            data(k,7)=1;
         end
    end
    if ~isempty(table)   &( table{3,7}>0.001 & table{4,7}>0.001 & table{5,7}>0.001) & table{6,7}>0.001 & table{7,7}>0.001 & table{8,7}>0.001& (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)
         if  SFN(k).single==0 & SFN(k).Q==3
            data(k,8)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2 
            data(k,8)=1;
         end
    end
    if ~isempty(table)    & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)
        if  SFN(k).single==0 & SFN(k).Q==3
            data(k,9)=1;
        elseif SFN(k).single==0 & SFN(k).Q==2
            data(k,9)=1;
    end
    end
     if ~isempty(table)   &( table{6,7}<0.001 & table{7,7}<0.001 & table{8,7}<0.001)  & (SFN(k).animal==943 ||SFN(k).animal==963 || SFN(k).animal==833)
        if  SFN(k).single==0 & SFN(k).Q==3
            data(k,10)=1;
        elseif SFN(k).single==0 & SFN(k).Q==3 
            data(k,10)=1;
    end
    end
end

Total=Count (data(:,9),'==1')
Azi=Count (data(:,1),'==1')/Total*100;
Pitch=Count (data(:,2),'==1')/Total*100;
Timbre=Count (data(:,3),'==1')/Total*100;
AP=Count (data(:,4),'==1')/Total*100;
AT=Count (data(:,5),'==1')/Total*100;
PT=Count (data(:,6),'==1')/Total*100;
APT=Count (data(:,7),'==1')/Total*100;
No=Count (data(:,8),'==1')/Total*100;
OnlyMainNo=Count (data(:,10),'==1')/Total*100;



Single=(Azi+Pitch+Timbre);
ThreeStim= APT;
TwoStim=AP+AT+PT;
All=Single+ThreeStim+No+TwoStim

   AxM=[Total,Azi,Pitch,Timbre,AP,AT,PT,APT,No,Single,TwoStim,ThreeStim]
%    
%    %
%    Results for
% 
%             Trained  CNTRL  AI Vowels   EU vowels  CntrlAI     CntrlEU
%         Azi       18   7       21          20          8           17
%        Pitch      42  55       25          39          66          74
%        Timbre     30  88       53          59          72          74
%         AP        38  19       23          35          4           21
%         AT        28  40       35          31          11          19
%         PT        41  185%56   25          39          57          55
%         APT       205 140      54          48          21          22
%         Total     402  534     552         552         615         615
% Single Stimuli    %16   %23-24    %17         %21         %23         %26
%    Two Stimuli    %19   %36-40    %15         %19         %11         %15
%  Three Stimuli    %37   %29-23     %9          %8          %3          %3
%                   %27   %12-13
% 

