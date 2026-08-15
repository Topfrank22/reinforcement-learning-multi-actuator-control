clear all
close all
clc
addpath('..'); % Aggiunge la cartella genitore al path per poter usare AMSVisualizer

%% =========================================================
%  RL_FINETUNE.M
%  Script per il Fine-Tuning di un agente pre-addestrato.
%  Carica la Run 07 e la addestra SOLO sui 10 pacchi con
%  un learning rate molto basso per cercare la perfezione.
% ==========================================================

fprintf('=============================================\n');
fprintf('   FINE-TUNING AGENTE (Transfer Learning)\n');
fprintf('=============================================\n\n');

% --- Configurazione Fine-Tuning ---
LOAD_RUN_NAME = 'Finetuned_Run_14_ProfParams_SafeGap';
SAVE_RUN_NAME = 'Finetuned_Run_16_RiskAverse';

finetuneLearnRate = 1e-5;  % 10x più basso del prof per micro-tuning
entropy           = 0.001; % Entropia quasi nulla per annullare errori casuali
maxEpisodes       = 15000;
stopValue         = 12500; % Threshold volutamente alto per fermare manualmente

%% =========================================================
%  CREAZIONE AMBIENTE (Forzato a 10 pacchi)
% ==========================================================
clear RL_curriculum_env;
env = RL_curriculum_env();

% FORZIAMO L'AMBIENTE AL LIVELLO MASSIMO (10 PACCHI)
env.curriculum_level = 10;
env.curriculum_limit = 10;

actInfo = getActionInfo(env);
obsInfo = getObservationInfo(env);

%% =========================================================
%  CARICAMENTO AGENTE PRE-ADDESTRATO
% ==========================================================
agentFile = fullfile('Phases', LOAD_RUN_NAME, 'agent_trained.mat');
if ~exist(agentFile, 'file')
    error('File dell''agente %s non trovato! Assicurati di aver eseguito la Run 07.', agentFile);
end

fprintf('Caricamento dell''agente pre-addestrato da: %s...\n', LOAD_RUN_NAME);
load(agentFile, 'agent');

% Estraiamo le reti neurali già addestrate (i "cervelli")
actor = getActor(agent);
critic = getCritic(agent);

% Aggiorniamo le opzioni dell'ottimizzatore con il nuovo Learning Rate basso
actorOpts  = rlOptimizerOptions(LearnRate=finetuneLearnRate, GradientThreshold=1);
criticOpts = rlOptimizerOptions(LearnRate=finetuneLearnRate, GradientThreshold=1);

% Ricreiamo l'agente mantenendo i pesi pre-addestrati ma con le nuove opzioni
agentOpts = rlPPOAgentOptions( ...
    SampleTime              = env.dt, ...
    ActorOptimizerOptions   = actorOpts, ...
    CriticOptimizerOptions  = criticOpts, ...
    ExperienceHorizon       = 1024, ...
    ClipFactor              = 0.02, ... % Allineato al professore (era 0.2)
    EntropyLossWeight       = entropy, ...
    MiniBatchSize           = 256, ...
    NumEpoch                = 3, ...
    AdvantageEstimateMethod = "gae", ...
    GAEFactor               = 0.95, ...
    DiscountFactor          = 0.99);

agent = rlPPOAgent(actor, critic, agentOpts);

%% =========================================================
%  ADDESTRAMENTO FINE-TUNING
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
fprintf('Avvio Addestramento di Fine-Tuning a 10 pacchi fissi...\n\n');
trainingStats = train(agent, env, trainOpts);
tPhaseElapsed = toc(tPhaseStart);

%% =========================================================
%  SALVATAGGIO E VALUTAZIONE
% ==========================================================
runFolder = fullfile('Phases', SAVE_RUN_NAME);
if ~exist(runFolder, 'dir'), mkdir(runFolder); end

save(fullfile(runFolder, 'agent_trained.mat'), 'agent');
save(fullfile(runFolder, 'trainingStats.mat'), 'trainingStats');

% --- Valutazione Finale ---
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
yline(stopValue, 'r--', 'Soglia Fine-Tuning', 'LineWidth',1.5, 'DisplayName','Target Threshold');
hold off;
xlabel('Episodio', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('Reward', 'FontSize', 12, 'FontWeight', 'bold');
title('Fine-Tuning — Training Curve', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter','none');
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
reportFile = fullfile(runFolder, 'finetune_report.md');
fid = fopen(reportFile, 'w');
if fid ~= -1
    fprintf(fid, '# Report Globale: Fine-Tuning\n\n');
    fprintf(fid, '*   **Data di Completamento**: %s\n', datestr(now));
    fprintf(fid, '*   **Modello Base**: %s\n', LOAD_RUN_NAME);
    fprintf(fid, '*   **Tempo di Calcolo Totale**: %.1f secondi (%.2f minuti)\n\n', tPhaseElapsed, tPhaseElapsed/60);
    fprintf(fid, '## Performance Finali (su 10 pacchi)\n\n');
    fprintf(fid, '| Metrica | Valore |\n');
    fprintf(fid, '| :--- | :--- |\n');
    fprintf(fid, '| **Sorting Accuracy Media** | **%.1f%%** |\n', meanAcc);
    fprintf(fid, '| Deviazione Standard | %.1f%% |\n', stdAcc);
    fprintf(fid, '| Episodi Totali Eseguiti | %d |\n', numel(trainingStats.EpisodeReward));
    fprintf(fid, '\n## Visualizzazioni\n\n');
    fprintf(fid, '### Curva di Addestramento (Fine-Tuning)\n');
    fprintf(fid, '![Training Curve](training_curve.png)\n\n');
    fprintf(fid, '### Istogramma Valutazione Finale\n\n');
    fprintf(fid, '![Evaluation Accuracy](evaluation_accuracy_hist.png)\n');
    fclose(fid);
end

fprintf('\n*** Fine-Tuning Completato! ***\n');
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
