SingleUnits_2014
load('pCoor_trained.mat')

for kk  = 1: length(trained_metrics)
    if ~isempty(trained_metrics(kk).animal)
        animal = trained_metrics(kk).animal;
        pents  = trained_metrics(kk).penetration;
        sh     = trained_metrics(kk).shrank;
        
        ss  = cell2mat( S_units(:,1:2));
        ind = find(ss(:,1)==animal & ss(:,2)==pents);
        
        trained_metrics(kk).field     = pCoor(ind,11); 
        %trained_metrics(kk).Totalshrank= pCoor(ind,12); should be same- not changing!
        if sh==1
        trained_metrics(kk).xcoor     = pCoor(ind,3);
        trained_metrics(kk).ycoor     = pCoor(ind,4);
        elseif sh ==2 & any(pCoor(ind,5))
        trained_metrics(kk).xcoor     = pCoor(ind,5);
        trained_metrics(kk).ycoor     = pCoor(ind,6);
        elseif sh ==2 & ~any(pCoor(ind,5))
        trained_metrics(kk).xcoor     = pCoor(ind,3);
        trained_metrics(kk).ycoor     = pCoor(ind,4);
        elseif sh ==3 
        trained_metrics(kk).xcoor     = pCoor(ind,7);
        trained_metrics(kk).ycoor     = pCoor(ind,8);
        elseif sh ==4 
        trained_metrics(kk).xcoor     = pCoor(ind,9);
        trained_metrics(kk).ycoor     = pCoor(ind,10);
        end
    end
end
             