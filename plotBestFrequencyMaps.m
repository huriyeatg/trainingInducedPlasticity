function plotBestFrequencyMaps ( save_path)
% plots Best Frequency maps for trained as well as for control dataset

% H Atilgan

%% load extracted info
% load trained metrics
load trained_metrics
BFcorrection
CoordinateCorrection
load gpointsTrained

%load control metrics used in Bizley et al., 09
load control_metrics
%%  exclude nans, not significant units

% For trained metrics
BFtrained=[]; fieldtrained=[];
xtrained=[]; ytrained=[];  indextrained =[];
for ind=1:length(trained_metrics)
    if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF)
        BFtrained       = [BFtrained;trained_metrics(ind).BF];
        xtrained          = [xtrained ; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
        ytrained          = [ytrained ; trained_metrics(ind).ycoor];
        fieldtrained   = [fieldtrained;trained_metrics(ind).field];
        indextrained = [indextrained ; ind];
        
    end
end

% For control metrics
BFControl=[];fieldControl=[];
xControl=[]; yControl=[]; indexControl =[];
for ind=1:length(control_metrics)
    if   ~isempty(control_metrics(ind).bf) & control_metrics(ind).field<5 & ind~=88
        BFControl      = [BFControl;control_metrics(ind).bf(1)];
        xControl         = [xControl ; control_metrics(ind).mapx,control_metrics(ind).field];
        yControl         = [yControl ; control_metrics(ind).mapy];
        fieldControl   = [fieldControl ; control_metrics(ind).field];
        indexControl = [indexControl ; ind];
        
    end
end

%% Plot the BF map for auditory cortex

for k = 1:2 % BF maps for control & trained
    %%  Change trained or control to plot the dataset.
    if k ==1
        BF = BFtrained;
        field = fieldtrained;
        x = xtrained;
        y = ytrained;
        index = indextrained;
        titleSt = 'Trained';
        cAxis = [ 0.2 30];
    elseif k ==2
        BF = BFControl;
        field = fieldControl;
        x = xControl;
        y = yControl;
        index = indexControl;
        titleSt = 'Control';
        cAxis = [ 0.2 30];
    end
     
    %% plot every recording:
    uy=unique(y);
    xx=x(:,1);
    yy=y(:,1);
    for ii=1:length(uy)
        f=find(y==uy(ii));
        ux=unique(xx(f));
        if length(ux)>1%there are multiple points
            for pp=1:length(ux)
                fx=find(x(f)==ux(pp));
                ang=2*pi/length(fx);
                [a,i]=sort(x(f(fx),2));
                for ff=1:length(fx)
                    xx(f(fx(ff)))=x(f(fx(ff)),1)+cos(i(ff)*ang);
                    yy(f(fx(ff)))=y(f(fx(ff)))+sin(i(ff)*ang);
                end
            end
        else
            ang=2*pi/length(f);
            [a,i]=sort(x(f,2));
            for ff=1:length(f)
                xx(f(ff))=x(f(ff),1)+cos(i(ff)*ang);
                yy(f(ff))=y(f(ff))+sin(i(ff)*ang);
            end
        end
    end
    
    %% add in the extra points so that the voronoi tesselation is restricted to
    % the cortex rather than the size of the plot
    
    jj=ind;
    jj=length(xx);
    for ii=1:length(g)
        xx(jj)=g(ii,1);
        yy(jj)=g(ii,2);
        BF(jj)=NaN;
        field(jj)=NaN;
        index(jj)=NaN;
        jj=jj+1;
    end
    
    BF=log(BF);
    
    % create voronoin points
    clear vdata vy vx jj v c colormap jcmap
    vx=xx;
    vy=yy;
    [v c]=voronoin([vx(:),vy(:)]);
    
    %% plot map
    figure
    im=imread('backgroundImageForBFmaps.jpg');
    
    imagesc(im); axis image;
    colormap;%(gray);
    box off
    pos=get(gca,'pos');
    pos(3)=pos(3)+0.1;
    pos(4)=pos(4)+0.1;
    pos=set(gca,'pos');
    
    vdata(1:length(BF))=BF;
    title('BF');
    col=[log(cAxis(1)) log(cAxis(2))];
    
    for i=1:length(c)
        if isnan(vdata(i))
            p=patch(v(c{i},1),v(c{i},2),'k','EdgeColor','none','FaceColor','none');
            % no edge color means that the empty cells we have are not visible
        elseif isinf(vdata(i))
            p=patch(v(c{i},1),v(c{i},2),'k','EdgeColor','black','FaceColor','none');
        else
            p=patch(v(c{i},1),v(c{i},2),vdata(i),'EdgeColor', 'black','linewidth',0.1);
        end
    end
    axis ( [ 00 1250 0 1250])
    hold on
    
    % add colorbar
    pos=get(gca,'pos');
    caxis(col)
    colorbar
    axis off;
    jcmap=colormap(jet);
    colormap(jcmap);
    
    saveas(gcf, fullfile(save_path,  ['BFmaps_',titleSt]), 'fig');
    print(gcf, '-dsvg',   fullfile(save_path, ['BFmaps_',titleSt]));
    print(gcf, '-dpng',  fullfile(save_path, ['BFmaps_',titleSt]));
    
end

% % for stats
% X =[ BFtrained;BFControl];
% group = [ fieldtrained, ones(size(fieldtrained,1),1) ; field, ones(size(field,1),1)*2];
% anovan(X, group, 'full')
