clear all
close all
clc
addpath('..'); % Per AMSVisualizer

%% =========================================================
%  RL_INERTIA_FINETUNE.M
%  Fine-Tuning dell'agente perfetto (Run 16 Curriculum) sulla
%  variante con Inerzia Dinamica dei Motori.
%
%  L'agente deve imparare che i pacchi pesanti rallentano i
%  motori delle celle AMS (costante di tempo tau ~ massa).
% ==========================================================

fprintf('=============================================\n');
fprintf('   FINE-TUNING AGENTE — MOTOR INERTIA\n');
fprintf('=============================================\n\n');

% --- Configurazione Fine-Tuning ---
% Carichiamo dalla cartella Runs/Inertia_Run_02_RiskAverse
LOAD_PATH     = fullfile('Runs', 'Inertia_Run_02_RiskAverse');
SAVE_RUN_NAME = 'Inertia_Run_03_DynamicGap';

% Parametri PPO per il Fine-Tuning adattivo alle nuove reward:
finetuneLearnRate = 5e-5;   % Adattamento rapido ma stabile alle nuove soglie dinamiche
entropy           = 0.002;  % Minima esplorazione per assestare la policy
clipFactor        = 0.05;   % Margine moderato per facilitare il cambiamento
maxEpisodes       = 15000;
stopValue         = 11500;

%% =========================================================
%  CREAZIONE AMBIENTE CON INERZIA (Forzato a 10 pacchi)
% ==========================================================
clear RL_inertia_env;
env = RL_inertia_env();

% FORZIAMO L'AMBIENTE AL LIVELLO MASSIMO (10 PACCHI)
env.curriculum_level = 10;
env.curriculum_limit = 10;

actInfo = getActionInfo(env);
obsInfo = getObservationInfo(env);

%% =========================================================
%  CARICAMENTO AGENTE PRE-ADDESTRATO (dalla Run 16 Curriculum)
% ==========================================================
agentFile = fullfile(LOAD_PATH, 'agent_trained.mat');
if ~exist(agentFile, 'file')
    error('File agente non trovato: %s\nAssicurati che la Run 16 Curriculum esista.', agentFile);
end

fprintf('Caricamento agente pre-addestrato da: %s...\n', LOAD_PATH);
load(agentFile, 'agent');

% Estraiamo le reti neurali già addestrate
actor = getActor(agent);
critic = getCritic(agent);

% Aggiorniamo le opzioni dell'ottimizzatore
actorOpts  = rlOptimizerOptions(LearnRate=finetuneLearnRate, GradientThreshold=1);
criticOpts = rlOptimizerOptions(LearnRate=finetuneLearnRate, GradientThreshold=1);

% Ricreiamo l'agente mantenendo i pesi pre-addestrati
agentOpts = rlPPOAgentOptions( ...
    SampleTime              = env.dt, ...
    ActorOptimizerOptions   = actorOpts, ...
    CriticOptimizerOptions  = criticOpts, ...
    ExperienceHorizon       = 1024, ...
    ClipFactor              = clipFactor, ...
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

tStart = tic;
fprintf('Avvio Fine-Tuning con Motor Inertia a 10 pacchi...\n\n');
fprintf('Parametri Motor Inertia:\n');
fprintf('  tau_0 = %.3f s (costante di tempo a vuoto)\n', env.tau_0);
fprintf('  tau_k = %.3f s/kg (fattore inerziale)\n', env.tau_k);
fprintf('  Massa pacchi: [%.1f, %.1f] kg\n', env.min_mass, env.max_mass);
fprintf('  tau range: [%.3f, %.3f] s\n\n', ...
    env.tau_0 + env.tau_k * env.min_mass, ...
    env.tau_0 + env.tau_k * env.max_mass);
trainingStats = train(agent, env, trainOpts);
tElapsed = toc(tStart);

%% =========================================================
%  SALVATAGGIO E VALUTAZIONE
% ==========================================================
scriptDir = fileparts(mfilename('fullpath'));
runFolder = fullfile(scriptDir, 'Runs', SAVE_RUN_NAME);
if ~exist(runFolder, 'dir'), mkdir(runFolder); end

save(fullfile(runFolder, 'agent_trained.mat'), 'agent');
save(fullfile(runFolder, 'trainingStats.mat'), 'trainingStats');

% Snapshot dell'ambiente
copyfile(fullfile(scriptDir, 'RL_inertia_env.m'), fullfile(runFolder, 'RL_inertia_env.m'));

% --- Valutazione Finale ---
fprintf('\nAvvio valutazione Finale (30 episodi) a 10 pacchi con Inerzia...\n');
nEvalEpisodes = 30;
[meanAcc, stdAcc, allAcc] = evaluate_inertia(env, agent, nEvalEpisodes);

save(fullfile(runFolder, 'evaluation_stats.mat'), ...
    'meanAcc', 'stdAcc', 'allAcc', 'nEvalEpisodes', 'tElapsed');

% --- Grafici ---
hFig = figure('Visible','off','Position',[100 100 1000 500]);
episodes = 1:numel(trainingStats.EpisodeReward);
movAvg  = movmean(trainingStats.EpisodeReward, 100);
hold on;
plot(episodes, trainingStats.EpisodeReward, 'Color',[0.75 0.85 0.95], 'LineWidth',0.5, 'DisplayName','Episode Reward');
plot(episodes, movAvg, 'Color',[0.0 0.45 0.74], 'LineWidth',2.5, 'DisplayName','Moving Avg (100 ep)');
yline(stopValue, 'r--', 'Target', 'LineWidth',1.5, 'DisplayName','Target Threshold');
hold off;
xlabel('Episodio', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('Reward', 'FontSize', 12, 'FontWeight', 'bold');
title('Motor Inertia Fine-Tuning — Training Curve', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location','northwest', 'FontSize', 10);
grid on;
set(gca, 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.1);
exportgraphics(hFig, fullfile(runFolder, 'training_curve.png'), 'Resolution', 300);
close(hFig);

% Istogramma Accuratezza
hEvalFig = figure('Visible','off','Position',[100 100 800 450]);
bar(allAcc, 'FaceColor', [0.12 0.56 1.0], 'EdgeColor', 'none');
yline(meanAcc, 'Color', [0.9 0.2 0.2], 'LineStyle', '--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Media: %.1f%%', meanAcc));
xlabel('Episodio di Test', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sorting Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Motor Inertia — Valutazione Finale (%d pacchi)', env.curriculum_limit), ...
    'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;
ylim([0 110]);
set(gca, 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.1);
exportgraphics(hEvalFig, fullfile(runFolder, 'evaluation_accuracy_hist.png'), 'Resolution', 300);
close(hEvalFig);

% Report Markdown
reportFile = fullfile(runFolder, 'inertia_report.md');
fid = fopen(reportFile, 'w');
if fid ~= -1
    fprintf(fid, '# Report: Motor Inertia Fine-Tuning\n\n');
    fprintf(fid, '## Modello Fisico\n');
    fprintf(fid, 'Dinamica del motore sotto carico: `V(t+dt) = V(t) + (dt/tau) * (V_target - V(t))`\n\n');
    fprintf(fid, '| Parametro | Valore |\n');
    fprintf(fid, '| :--- | :--- |\n');
    fprintf(fid, '| tau_0 (a vuoto) | %.3f s |\n', env.tau_0);
    fprintf(fid, '| tau_k (fattore inerziale) | %.3f s/kg |\n', env.tau_k);
    fprintf(fid, '| Massa pacchi | [%.1f, %.1f] kg |\n', env.min_mass, env.max_mass);
    fprintf(fid, '| tau (pacco leggero, 1 kg) | %.3f s |\n', env.tau_0 + env.tau_k * env.min_mass);
    fprintf(fid, '| tau (pacco pesante, 10 kg) | %.3f s |\n\n', env.tau_0 + env.tau_k * env.max_mass);
    fprintf(fid, '## Performance Finali\n\n');
    fprintf(fid, '| Metrica | Valore |\n');
    fprintf(fid, '| :--- | :--- |\n');
    fprintf(fid, '| **Data** | %s |\n', datestr(now));
    fprintf(fid, '| **Modello Base** | Finetuned_Run_16_RiskAverse (Curriculum) |\n');
    fprintf(fid, '| **Tempo di Calcolo** | %.1f s (%.2f min) |\n', tElapsed, tElapsed/60);
    fprintf(fid, '| **Sorting Accuracy Media** | **%.1f%%** |\n', meanAcc);
    fprintf(fid, '| Deviazione Standard | %.1f%% |\n', stdAcc);
    fprintf(fid, '| Episodi Totali | %d |\n', numel(trainingStats.EpisodeReward));
    fprintf(fid, '\n## Visualizzazioni\n\n');
    fprintf(fid, '![Training Curve](training_curve.png)\n\n');
    fprintf(fid, '![Evaluation Accuracy](evaluation_accuracy_hist.png)\n');
    fclose(fid);
end

fprintf('\n*** Fine-Tuning Motor Inertia Completato! ***\n');
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
