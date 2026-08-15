clear all
close all
clc
addpath('..'); % Aggiunge la cartella genitore al path per poter usare AMSVisualizer

%% =========================================================
%  RL_CURRICULUM_TRAIN.M
%  Script autonomo per Curriculum Learning Adattivo
%  NON tocca nulla fuori dalla cartella Curriculum/.
%  Addestra l'agente partendo da 2 pacchi, con livello
%  di difficoltà (numero pacchi) che aumenta automaticamente
%  grazie al Self-Paced Learning in RL_curriculum_env.m
% ==========================================================

fprintf('=============================================\n');
fprintf('   CURRICULUM LEARNING ADATTIVO (Continuous)\n');
fprintf('=============================================\n\n');

% --- Configurazione Globale ---
RUN_NAME    = 'Adaptive_Curriculum_Run_15_Strict_Continuous';
learnRate   = 3e-4;  
entropy     = 0.02;  
maxEpisodes = 25000;
stopValue   = 9700; % Threshold finale per 10 pacchi

%% =========================================================
%  CREAZIONE AMBIENTE
% ==========================================================
clear RL_curriculum_env;
env = RL_curriculum_env();
% Inizialmente env.curriculum_level = 2 di default.

actInfo = getActionInfo(env);
obsInfo = getObservationInfo(env);
numObs  = prod(obsInfo.Dimension);  % 100

%% =========================================================
%  ARCHITETTURA RETE NEURALE
% ==========================================================
criticLayerSizes = [512 256 128];
actorLayerSizes  = [512 256 128];

fprintf('Creazione agente PPO...\n');

% Critic
criticNetwork = [
    featureInputLayer(numObs)
    fullyConnectedLayer(criticLayerSizes(1), ...
        Weights=sqrt(2/numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(1),1))
    reluLayer
    fullyConnectedLayer(criticLayerSizes(2), ...
        Weights=sqrt(2/criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(2),1))
    reluLayer
    fullyConnectedLayer(criticLayerSizes(3), ...
        Weights=sqrt(2/criticLayerSizes(2))*(rand(criticLayerSizes(3),criticLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(3),1))
    reluLayer
    fullyConnectedLayer(1, ...
        Weights=sqrt(2/criticLayerSizes(3))*(rand(1,criticLayerSizes(3))-0.5), ...
        Bias=1e-3)
];
criticNetwork = dlnetwork(criticNetwork);
criticOpts    = rlOptimizerOptions(LearnRate=learnRate, GradientThreshold=1);
critic        = rlValueFunction(criticNetwork, obsInfo);

% Actor (trunk + mean + stddev)
inPath = [
    featureInputLayer(numObs, Name="netOin")
    fullyConnectedLayer(actorLayerSizes(1), Name="fc1", ...
        Weights=sqrt(2/numObs)*(rand(actorLayerSizes(1),numObs)-0.5), ...
        Bias=1e-3*ones(actorLayerSizes(1),1))
    reluLayer
    fullyConnectedLayer(actorLayerSizes(2), Name="fc2", ...
        Weights=sqrt(2/actorLayerSizes(1))*(rand(actorLayerSizes(2),actorLayerSizes(1))-0.5), ...
        Bias=1e-3*ones(actorLayerSizes(2),1))
    reluLayer(Name="relulast")
];
meanPath = [
    fullyConnectedLayer(actorLayerSizes(3), Name="MeanLyr", ...
        Weights=sqrt(2/actorLayerSizes(2))*(rand(actorLayerSizes(3),actorLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(actorLayerSizes(3),1))
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension), Name="meanOutLyr", ...
        Weights=sqrt(2/actorLayerSizes(3))*(rand(prod(actInfo.Dimension),actorLayerSizes(3))-0.5), ...
        Bias=1e-3*ones(prod(actInfo.Dimension),1))
    tanhLayer(Name="thmeanOutLyr")
];
sdevPath = [
    fullyConnectedLayer(actorLayerSizes(3), Name="StdLyr", ...
        Weights=sqrt(2/actorLayerSizes(2))*(rand(actorLayerSizes(3),actorLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(actorLayerSizes(3),1))
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension), Name="stdFCLyr", ...
        Weights=sqrt(2/actorLayerSizes(3))*(rand(prod(actInfo.Dimension),actorLayerSizes(3))-0.5), ...
        Bias=1e-3*ones(prod(actInfo.Dimension),1))
    softplusLayer(Name="stdOutLyr")
];
actorGraph = layerGraph(inPath);
actorGraph = addLayers(actorGraph, meanPath);
actorGraph = addLayers(actorGraph, sdevPath);
actorGraph = connectLayers(actorGraph, "relulast", "MeanLyr/in");
actorGraph = connectLayers(actorGraph, "relulast", "StdLyr/in");
actorNet   = dlnetwork(actorGraph);

actorOpts = rlOptimizerOptions(LearnRate=learnRate, GradientThreshold=1);
actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    ActionMeanOutputNames           = "thmeanOutLyr", ...
    ActionStandardDeviationOutputNames = "stdOutLyr", ...
    ObservationInputNames           = "netOin");

% --- Configurazione PPO ---
agentOpts = rlPPOAgentOptions( ...
    SampleTime              = env.dt, ...
    ActorOptimizerOptions   = actorOpts, ...
    CriticOptimizerOptions  = criticOpts, ...
    ExperienceHorizon       = 1024, ...
    ClipFactor              = 0.2, ... 
    EntropyLossWeight       = entropy, ...
    MiniBatchSize           = 256, ...
    NumEpoch                = 3, ...
    AdvantageEstimateMethod = "gae", ...
    GAEFactor               = 0.95, ...
    DiscountFactor          = 0.99);

agent = rlPPOAgent(actor, critic, agentOpts);

%% =========================================================
%  ADDESTRAMENTO CONTINUO
% ==========================================================
trainOpts = rlTrainingOptions( ...
    MaxEpisodes              = maxEpisodes, ...
    MaxStepsPerEpisode       = floor(15/env.dt), ...
    ScoreAveragingWindowLength = 100, ...
    Verbose                  = false, ...
    Plots                    = "training-progress", ...
    StopTrainingCriteria     = "AverageReward", ...
    StopTrainingValue        = stopValue, ...
    SaveAgentCriteria        = "EpisodeReward", ...
    SaveAgentValue           = stopValue * 1.1);

tPhaseStart = tic;
fprintf('Avvio Addestramento Adattivo (L''ambiente cambiera'' difficoltà da solo)...\n\n');
trainingStats = train(agent, env, trainOpts);
tPhaseElapsed = toc(tPhaseStart);

%% =========================================================
%  SALVATAGGIO E VALUTAZIONE
% ==========================================================
runFolder = fullfile('Phases', RUN_NAME);
if ~exist(runFolder, 'dir'), mkdir(runFolder); end

save(fullfile(runFolder, 'agent_trained.mat'), 'agent');
save(fullfile(runFolder, 'trainingStats.mat'), 'trainingStats');

% --- Valutazione Finale ---
% Forza il curriculum_level al massimo per la valutazione
env.curriculum_level = 10;
env.curriculum_limit = 10;
fprintf('\nAvvio valutazione Finale (30 episodi) a 10 pacchi...\n');
nEvalEpisodes = 30;
[meanAcc, stdAcc, allAcc] = evaluate_curriculum(env, agent, nEvalEpisodes);

save(fullfile(runFolder, 'evaluation_stats.mat'), 'meanAcc', 'stdAcc', 'allAcc', 'nEvalEpisodes', 'tPhaseElapsed');

% --- Grafici ---
% 1. Curva di Addestramento Singola (mostrerà scalini di difficoltà)
hFig = figure('Visible','off','Position',[100 100 1000 500]);
episodes = 1:numel(trainingStats.EpisodeReward);
movAvg  = movmean(trainingStats.EpisodeReward, 100);
hold on;
plot(episodes, trainingStats.EpisodeReward, 'Color',[0.75 0.85 0.95], 'LineWidth',0.5, 'DisplayName','Episode Reward');
plot(episodes, movAvg, 'Color',[0.0 0.45 0.74], 'LineWidth',2.5, 'DisplayName','Moving Avg (100 ep)');
yline(stopValue, 'r--', 'Soglia Finale (10 pacchi)', 'LineWidth',1.5, 'DisplayName','Target Threshold');
hold off;
xlabel('Episodio', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('Reward', 'FontSize', 12, 'FontWeight', 'bold');
title('Curriculum Adattivo Continuo — Training Curve', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter','none');
legend('Location','northwest', 'FontSize', 10);
grid on;
set(gca, 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.1);
exportgraphics(hFig, fullfile(runFolder, 'training_curve.png'), 'Resolution', 300);
close(hFig);

% 2. Istogramma Accuratezza
hEvalFig = figure('Visible','off','Position',[100 100 800 450]);
bar(allAcc, 'FaceColor', [0.12 0.56 1.0], 'EdgeColor', 'none');
yline(meanAcc, 'Color', [0.9 0.2 0.2], 'LineStyle', '--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Media: %.1f%%', meanAcc));
xlabel('Episodio di Test', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sorting Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Valutazione Finale (%d pacchi) — accuratezza per Episodio', env.curriculum_limit), ...
    'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;
ylim([0 110]);
set(gca, 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.1);
exportgraphics(hEvalFig, fullfile(runFolder, 'evaluation_accuracy_hist.png'), 'Resolution', 300);
close(hEvalFig);

% 3. Report Globale Markdown
reportFile = fullfile(runFolder, 'adaptive_curriculum_report.md');
fid = fopen(reportFile, 'w');
if fid ~= -1
    fprintf(fid, '# Report Globale: Curriculum Adattivo Continuo\n\n');
    fprintf(fid, '*   **Data di Completamento**: %s\n', datestr(now));
    fprintf(fid, '*   **Livello Finale Raggiunto**: %d Pacchi\n', env.curriculum_limit);
    fprintf(fid, '*   **Tempo di Calcolo Totale**: %.1f secondi (%.2f minuti)\n\n', tPhaseElapsed, tPhaseElapsed/60);
    fprintf(fid, '## Performance Finali (su 10 pacchi)\n\n');
    fprintf(fid, '| Metrica | Valore |\n');
    fprintf(fid, '| :--- | :--- |\n');
    fprintf(fid, '| **Sorting Accuracy Media** | **%.1f%%** |\n', meanAcc);
    fprintf(fid, '| Deviazione Standard | %.1f%% |\n', stdAcc);
    fprintf(fid, '| Episodi Totali Eseguiti | %d |\n', numel(trainingStats.EpisodeReward));
    fprintf(fid, '\n## Visualizzazioni\n\n');
    fprintf(fid, '### Curva di Addestramento Adattiva\n');
    fprintf(fid, '*(Nota: i gradini nel grafico corrispondono ai Level Up automatici della simulazione)*\n\n');
    fprintf(fid, '![Training Curve](training_curve.png)\n\n');
    fprintf(fid, '### Istogramma Valutazione Finale\n\n');
    fprintf(fid, '![Evaluation Accuracy](evaluation_accuracy_hist.png)\n');
    fclose(fid);
end

fprintf('\n*** Addestramento Adattivo Completato! ***\n');
fprintf('Dati e report salvati in: %s\n', runFolder);

%% =========================================================
%  VISUALIZZATORE FISICO
% ==========================================================
fprintf('\nPremi un tasto sulla Command Window di MATLAB per avviare il visualizzatore fisico...\n');
pause;

env.reset();
plot(env);
rng(10);
simOpts = rlSimulationOptions('MaxSteps', floor(15/env.dt));
experience = sim(env, agent, simOpts);
