%% Table 6
results_yahoo_mf = [0.7275 -0.589; 0.5692 -0.5099; 0.7295 -0.5814;...
    0.7218 -0.4511; 0.7333 -0.5713; 0.7352 -0.5532; 0.7426 -0.6583;...
    0.7275 -0.587; 0.7331 -0.567; 0.7315 -0.5685; 0.7329 -0.559;...
    0.7319 -0.5718; 0.7299 -0.5804];
results_yahoo_ae = [0.7186 -0.5907; 0.5397 -0.5214; 0.7213 -0.5858;...
    0.7318 -0.5666; 0.7307 -0.5606; 0.7283 -0.6207; 0.7249 -0.5826;...
    0.7255 -0.5817; 0.7221 -0.5856; 0.723 -0.5803; 0.7196 -0.5799;...
    0.7128 -0.5629; 0.7202 -0.5978; 0.7206 -0.5441];
results_product_mf = [0.6281 -0.3294; 0.6324 -0.4743; 0.6320 -0.3262;...
    0.5705 -0.6934; 0.6894 -0.3627; 0.6417 -0.4129; 0.6435 -0.3343;...
    0.6523 -0.3233; 0.6348 -0.5528; 0.6671 -0.3211; 0.6383 -0.3246;...
    0.6326 -0.4671; 0.6315 -0.4678];
results_product_ae = [0.6529 -0.5074; 0.6702 -0.3391; 0.6679 -0.4957;...
    0.6827 -0.4050; 0.7012 -0.3550; 0.6826 -0.3661; 0.6506 -0.3860;...
    0.6894 -0.3773; 0.6924 -0.4058; 0.6814 -0.4966; 0.6860 -0.3307;...
    0.6534 -0.4926; 0.6717 -0.7727; 0.6785 -0.7080];
imp1 = (results_yahoo_mf - results_yahoo_mf(1,:))./results_yahoo_mf(1,:);
imp2 = (results_yahoo_ae(1:end-3,:) - results_yahoo_ae(1,:))./results_yahoo_ae(1,:);
imp3 = (results_yahoo_ae(end-2:end,:) - results_yahoo_ae(end-2,:))./results_yahoo_ae(end-2,:);
imp4 = (results_product_mf - results_product_mf(1,:))./results_product_mf(1,:);
imp5 = (results_product_ae(1:end-3,:) - results_product_ae(1,:))./results_product_ae(1,:);
imp6 = (results_product_ae(end-2:end,:) - results_product_ae(end-2,:))./results_product_ae(end-2,:);

%% Table 8
result_yahoo_bridge = [0.7303 -0.5734 0.7259 -0.5621;...
    0.7333 -0.5713 0.7318 -0.5666;...
    0.7352 -0.5532 0.7307 -0.5605];
result_yahoo_refine = [0.7384 -0.6602 0.7224 -0.6253;...
    0.7426 -0.6583 0.7283 -0.6207];
result_yahoo_weightc = [0.7324 -0.55 0.7233 -0.5825;...
    0.7331 -0.567 0.7249 -0.5826];
result_product_bridge = [0.6830 -0.3690 0.6825 -0.3204;...
    0.6894 -0.3627 0.6827 -0.4050;...
    0.6417 -0.4129 0.7012 -0.3550];
result_product_refine = [0.6295 -0.3310 0.6697 -0.3770;...
    0.6435 -0.3343 0.6826 -0.3661];
result_product_weightc = [0.5447 -0.3225 0.6413 -0.4149;...
    0.6348 -0.5528 0.6506 -0.3860];
imp1 = (result_yahoo_bridge(2:end,:)-result_yahoo_bridge(1,:))./result_yahoo_bridge(1,:);
imp2 = (result_yahoo_refine(2,:)-result_yahoo_refine(1,:))./result_yahoo_refine(1,:);
imp3 = (result_yahoo_weightc(2,:)-result_yahoo_weightc(1,:))./result_yahoo_weightc(1,:);
imp4 = (result_product_bridge(2:end,:)-result_product_bridge(1,:))./result_product_bridge(1,:);
imp5 = (result_product_refine(2,:)-result_product_refine(1,:))./result_product_refine(1,:);
imp6 = (result_product_weightc(2,:)-result_product_weightc(1,:))./result_product_weightc(1,:);

%% Figure 4
PN_MF = [0.6648 0.6814 0.6774 0.6866 0.6789 0.6841 0.684 0.6842 0.6730 0.6743;...
    0.6747 0.7086 0.7085 0.7160 0.7037 0.7090 0.7080 0.7094 0.7080 0.7033;...
    0.6674 0.7179 0.7186 0.7243 0.7109 0.7168 0.7154 0.7174 0.7184 0.7142;...
    0.6591 0.7179 0.7153 0.7187 0.7062 0.7095 0.7112 0.7129 0.7183 0.7124;...
    0.6586 0.7113 0.7002 0.7096 0.7074 0.6900 0.7030 0.6981 0.7090 0.6979];
N = 10;
% C = linspecer(N);
% C = linspecer(N, 'qualitative');
C = linspecer(N, 'sequential');
for i = 1:N
    plot(PN_MF(:,i), '-*', 'color', C(i,:), 'linewidth', 2);
    hold on 
end
lg = legend({'IPS','Bridge-Var1','Bridge-Var2','Refine-Var','CausE','WeightS-local',...
    'WeightS-global','Delay','FeatE-alter','FeatE-concat'},'NumColumns',2);
set(lg,'Box','off','Position',[0.360952383107726 0.266269668162772 0.549999988877347 0.240476183735189])
xticks([1 2 3 4 5])
xticklabels({'10%','30%','50%','70%','90%'})
xlabel('Ratio of positive samples');
ylabel('AUC');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',13,'linewidth',2);
axis tight 
saveas(gcf,'PN_MF','epsc');

%% Figure 5
PN_AE = [0.6506 0.6215 0.6425 0.6649 0.6627 0.6528 0.6498 0.6425;...
    0.6875 0.6741 0.6782 0.6853 0.6845 0.6748 0.6782 0.6749;...
    0.6962 0.6875 0.6935 0.6917 0.6912 0.6846 0.6797 0.6771;...
    0.7146 0.6959 0.6932 0.6948 0.6943 0.6942 0.6894 0.6893;...
    0.7263 0.6926 0.7017 0.7059 0.7056 0.7050 0.6799 0.6906];
N = 8;
% C = linspecer(N);
% C = linspecer(N, 'qualitative');
C = linspecer(N, 'sequential');
for i = 1:N
    plot(PN_AE(:,i), '-*', 'color', C(i,:), 'linewidth', 2);
    hold on 
end
lg = legend({'Bridge-Var1','Bridge-Var2','Refine-Var','WeightS-local',...
    'WeightS-global','Delay','FeatE-alter','FeatE-concat'},'NumColumns',2);
set(lg,'Box','off','Position',[0.337202391356584 0.130555560380693 0.571428559667298 0.194047613654818])
xticks([1 2 3 4 5])
xticklabels({'10%','30%','50%','70%','90%'})
xlabel('Ratio of positive samples');
ylabel('AUC');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',13,'linewidth',2);
axis tight 
saveas(gcf,'PN_AE','epsc');

%% Figure 6
Scale_MF = [0.7184 0.7263 0.7257 0.7394 0.7271 0.7286 0.7276 0.7212 0.7247 0.7276;...
    0.7193 0.7292 0.7295 0.7414 0.7274 0.7296 0.7289 0.7257 0.7282 0.7283;...
    0.7194 0.7305 0.7310 0.7422 0.7279 0.7302 0.7289 0.7279 0.7292 0.7285;...
    0.7214 0.7318 0.7297 0.7427 0.7281 0.7283 0.7293 0.7306 0.7310 0.7296];
N = 10;
% C = linspecer(N);
% C = linspecer(N, 'qualitative');
C = linspecer(N, 'sequential');
for i = 1:N
    plot(Scale_MF(:,i), '-*', 'color', C(i,:), 'linewidth', 2);
    hold on 
end
lg = legend({'IPS','Bridge-Var1','Bridge-Var2','Refine-Var','CausE','WeightS-local',...
    'WeightS-global','Delay','FeatE-alter','FeatE-concat'},'NumColumns',2);
set(lg,'Box','off','Position',[0.361309533654933 0.603174609206026 0.549999988877348 0.240476183735189])
xticks([1 2 3 4])
xticklabels({'20%','40%','60%','80%'})
xlabel('Size');
ylabel('AUC');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',13,'linewidth',2);
axis tight 
saveas(gcf,'Scale_MF','epsc');

%% Figure 7
Scale_AE = [0.7205 0.7032 0.7253 0.7225 0.7224 0.7200 0.7241 0.7164;...
    0.7247 0.7238 0.7258 0.7234 0.7234 0.7201 0.7233 0.7123;...
    0.7275 0.7259 0.7268 0.7238 0.7239 0.7216 0.7220 0.7170;...
    0.7299 0.7288 0.7269 0.7243 0.7245 0.7211 0.7226 0.7204];
N = 8;
% C = linspecer(N);
% C = linspecer(N, 'qualitative');
C = linspecer(N, 'sequential');
for i = 1:N
    plot(Scale_AE(:,i), '-*', 'color', C(i,:), 'linewidth', 2);
    hold on 
end
lg = legend({'Bridge-Var1','Bridge-Var2','Refine-Var','WeightS-local',...
    'WeightS-global','Delay','FeatE-alter','FeatE-concat'},'NumColumns',2);
set(lg,'Box','off','Position',[0.328273819928012 0.14722222704736 0.571428559667298 0.194047613654818])
xticks([1 2 3 4])
xticklabels({'20%','40%','60%','80%'})
xlabel('Size');
ylabel('AUC');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',13,'linewidth',2);
axis tight 
saveas(gcf,'Scale_AE','epsc');

%% Figure 8
Refine_MF = [0.7389 0.7402 0.7404 0.7402 0.7407 0.7418 0.7425 0.7426 0.7425...
    0.7418 0.7428 0.7427 0.7428];
Refine_AE = [0.7239 0.7261 0.7268 0.7265 0.726 0.7273 0.7269 0.7272 0.7271...
    0.7271 0.7271 0.7273 0.7285];
N = 2;
C = linspecer(N, 'sequential');
hold on 
plot(200:200:2600,Refine_MF, '-*', 'color', C(1,:), 'linewidth', 2);
plot(200:200:2600,Refine_AE, '-*', 'color', C(2,:), 'linewidth', 2);

line([200 2600],[0.7384 0.7384], 'linestyle','--', 'Color', C(1,:), 'LineWidth', 2);
line([200 2600],[0.7224 0.7224], 'linestyle','--', 'Color', C(2,:), 'LineWidth', 2);

lg = legend('Low Rank','Neural Nets');
set(lg,'Box','off','Position',[0.662202383959222 0.487698310365972 0.251785709496055 0.101190473494076])
xlabel('Number of samples');
ylabel('AUC');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',13,'linewidth',2);
xlim([200 2600])
saveas(gcf,'Refine_sample','epsc');