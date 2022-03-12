clc
clear

load data;

%% one way anova
% [p,table,stats] = anova1(accs_bayes_5,methods);
% [c,m,h,names] = multcompare(stats,'ctype','bonferroni');

%% repeated measure anova
[p,table,stats] = anova_rm(accs_bayes_5);
[c,m,h,names] = multcompare(stats,'ctype','bonferroni');

%% repeated measure anova with BBCIToolbox
% [p,t,stats,terms,arg] = stat_rmanova(accs_bayes_5','design','repeated-measures');
% [c,m,h,names] = multcompare(stats,'ctype','bonferroni');