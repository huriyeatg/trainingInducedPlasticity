function corticalMap(backgroundImage_path, save_path)
warning('off', 'all')
load control_metricsStatsMore
load PropsControl

load trained_metrics
CoordinateCorrection
load gPointsTrained

for dd =1:2 % control & trained
    
    Fpitch=[];
    Ftimbre=[];
    Faz=[];
    x=[]; y=[];
    
    if dd==1
        excludeADF = [88,480:520]; 
        for ind=1:length(PropsControl)
            if ~isempty(PropsControl(ind).All) & ~isnan(PropsControl(ind).All) &...
                       control_metrics(ind).field<5 & ~ismember (ind,excludeADF)
                Fpitch    = [Fpitch;     PropsControl(ind).All(5)*100];
                Ftimbre = [Ftimbre;  PropsControl(ind).All(6)*100];
                Faz        = [Faz;         PropsControl(ind).All(4)*100];
                
                x=[x;control_metrics(ind).mapx, control_metrics(ind).PE(2)];
                y=[y;control_metrics(ind).mapy];

            end
        end
        xt = x;
        yt = y;
        titleSt = 'Control';
        
    else     
        for ind=1:length(trained_metrics)
            if ~isempty(trained_metrics(ind).SSRvaluesAll) & ~isnan(trained_metrics(ind).SSRvaluesAll)
                Fpitch     = [Fpitch;   trained_metrics(ind).SSRvaluesAll(5)*100];
                Ftimbre  = [Ftimbre; trained_metrics(ind).SSRvaluesAll(6)*100];
                Faz         = [ Faz;       trained_metrics(ind).SSRvaluesAll(4)*100];
                
                x= [x; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
                y= [y; trained_metrics(ind).ycoor];
            end
        end
        titleSt = 'trained';
    end
    
    %% take mean & clean points
    x=x(:,1);
    xx=[];
    yy=[];
    Fp=[];Fa=[];Ft=[];
    uy=unique(y);
    ind=1;
    for ii=1:length(uy)
        f=find(y==uy(ii));
        ux=unique(x(f));
        if length(ux)>1%there are multiple points
            for pp=1:length(ux)
                fx=find(x(f)==ux(pp));
                Fp(ind)=nanmean(Fpitch(f(fx)));
                Ft(ind)=nanmean(Ftimbre(f(fx)));
                Fa(ind)=nanmean(Faz(f(fx)));
                xx(ind)=ux(pp);
                yy(ind)=uy(ii);
                ind=ind+1;
            end
        else
            Fp(ind)=nanmean(Fpitch(f));
            Ft(ind)=nanmean(Ftimbre(f));
            Fa(ind)=nanmean(Faz(f));
            xx(ind)=x(f(1));
            yy(ind)=uy(ii);
            ind=ind+1;
        end
    end
    
    x=xx;
    y=yy;
    
    jj = length(x);
    for ii=1:length(g)
        x(jj)=g(ii,1);
        y(jj)=g(ii,2);
        Ft(jj)=NaN;
        Fa(jj)=NaN;
        Fp(jj)=NaN; 
        jj=jj+1;
    end
    
    Faz       = Fa;
    Fpitch   = Fp;
    Ftimbre = Ft;
    
    %% Plot the figure
    figure
    im = imread(fullfile(backgroundImage_path,'backgroundImageForBFmaps.jpg'));
    
    clear vdata vy vx jj v c colormap jcmap
    vx=x;
    vy=y;
    
    [v, c] = voronoin([vx(:),vy(:)]);
    
    for ii=1:3
        subplot(1,3,ii)
        imagesc(im);axis image;
        box off
        pos=get(gca,'pos');
        pos(3)=pos(3)+0.1;
        pos(4)=pos(4)+0.1;
        pos=set(gca,'pos');
         if ii==1
            vdata(1:length(Ftimbre))=Faz;
            title('Azimuth');
            col = [log(0.5) log(10)];
        elseif ii==3
            vdata(1:length(Ftimbre))=Ftimbre;
            title('Timbre');
            col=[1.5 5];%[0.05 3];
        elseif ii==2
            vdata(1:length(Fpitch))=Fpitch;
            title('Pitch');
            col=[log(1.5) log(20)];
        end
             
        for i=1:length(c)
            if isnan(vdata(i))
                pc= patch(v(c{i},1),v(c{i},2),'k','EdgeColor','none','FaceColor','none');
            elseif isinf(vdata(i))
                p = patch(v(c{i},1),v(c{i},2),'k','EdgeColor','black','FaceColor','none');
            else
                p = patch(v(c{i},1),v(c{i},2),vdata(i),'EdgeColor', 'black','linewidth',0.1);
            end

        end
         axis ( [ 00 1250 0 1250])
        % add colorbar
        pos = get(gca,'pos');
        caxis(col)
        colorbar
        axis off;
        jcmap=colormap(hot);
        jcmap(1:5,:)=[]; % lower end
        jcmap(end-5:end,:)=[]; % top end
        colormap(jcmap);
    end
    
    %% save figures
    saveas(gcf, fullfile(save_path,  ['corticalMap_',titleSt]), 'fig');
    print(gcf, '-dsvg',   fullfile(save_path, ['corticalMap_',titleSt]));
    print(gcf, '-dpng',  fullfile(save_path, ['corticalMap_',titleSt]));
    
end