function plot_vowelFormants ( save_path)
figure
scatter(1551, 936,300,'s','filled','markerfacecolor',[0 0 0]); hold on
scatter(2058, 730,300,'filled','markerfacecolor',[0.4 0.4 0.4]); hold on
scatter(1105, 460,300,'filled','markerfacecolor',[0.4 0.4 0.4]); hold on
scatter(2761, 437,300,'s','filled','markerfacecolor',[0 0 0]); hold on
ylim([100 1200]);set(gca,'YTick',300:300:1200)
xlim([800 3000]);set(gca,'XTick',1000:500:3000)
   set(gca,'FontSize',24,'FontWeight','bold','linewidth',5)

  
print(gcf,'-dpng',fullfile(save_path ,'vowelFormantPlot'));
