clear all
close all
clc

%% =========================================================
%  RUN CONFIGURATION
%  Cambia solo questa sezione per ogni nuova run.
% ==========================================================
RUN_NAME     = 'Run_17_Standard_ProfParams';
RUN_NOTES    = 'Test baseline (no curriculum) con 10 pacchi e parametri originali del prof (ClipFactor 0.02, LR 1e-4, Entropy 0.01) per confronto con il Curriculum.';
REWARD_NOTES = 'Baseline Dense Rewards (uguale alla Run 07 ma senza curriculum)';

% false = addestra nuovo agente da zero
% true  = carica agente gia' salvato per visualizzarlo
loadSavedAgent = false;
LOAD_FROM_RUN  = 'Run_07_PerfectBalance';

%% =========================================================
%  TRAINING  (eseguito solo se loadSavedAgent == false)
% ==========================================================
if ~loadSavedAgent

    clear RL_environment;
    env = RL_environment();

    % -- Info spazio osservazioni e azioni --
    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);
    numObs  = prod(obsInfo.Dimension);   % 100

    criticLayerSizes = [512 256 128];
    actorLayerSizes  = [512 256 128];

    % -- CRITIC (rete sequenziale → 1 uscita scalare) --
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
    criticOpts    = rlOptimizerOptions(LearnRate=3e-4, GradientThreshold=1);
    critic        = rlValueFunction(criticNetwork, obsInfo);

    % -- ACTOR (rete a 2 rami: mean + stddev, richiesto da rlContinuousGaussianActor) --
    % Trunk condiviso
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

    % Ramo media (tanh → output normalizzato in [-1,1])
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

    % Ramo deviazione standard (softplus → sempre positiva)
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

    % Assemblaggio del grafo
    actorGraph = layerGraph(inPath);
    actorGraph = addLayers(actorGraph, meanPath);
    actorGraph = addLayers(actorGraph, sdevPath);
    actorGraph = connectLayers(actorGraph, "relulast", "MeanLyr/in");
    actorGraph = connectLayers(actorGraph, "relulast", "StdLyr/in");
    actorNet   = dlnetwork(actorGraph);

    actorOpts = rlOptimizerOptions(LearnRate=1e-4, GradientThreshold=1);
    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
        ActionMeanOutputNames           = "thmeanOutLyr", ...
        ActionStandardDeviationOutputNames = "stdOutLyr", ...
        ObservationInputNames           = "netOin");

    % -- OPZIONI AGENTE PPO --
    agentOpts = rlPPOAgentOptions( ...
        SampleTime              = env.dt, ...
        ActorOptimizerOptions   = actorOpts, ...
        CriticOptimizerOptions  = criticOpts, ...
        ExperienceHorizon       = 1024, ...
        ClipFactor              = 0.02, ...
        EntropyLossWeight       = 0.01, ...   % Entropia moderata per la baseline
        MiniBatchSize           = 256, ...
        NumEpoch                = 3, ...
        AdvantageEstimateMethod = "gae", ...
        GAEFactor               = 0.95, ...
        DiscountFactor          = 0.99);

    agent = rlPPOAgent(actor, critic, agentOpts);

    % -- OPZIONI DI ADDESTRAMENTO --
    trainOpts = rlTrainingOptions( ...
        MaxEpisodes              = 25000, ...
        MaxStepsPerEpisode       = floor(15/env.dt), ...
        ScoreAveragingWindowLength = 100, ...
        Verbose                  = false, ...
        Plots                    = "training-progress", ...
        StopTrainingCriteria     = "AverageReward", ...
        StopTrainingValue        = 30000, ...
        SaveAgentCriteria        = "EpisodeReward", ...
        SaveAgentValue           = 20000);

    % -- TRAINING --
    trainingStats = train(agent, env, trainOpts);

    %% -------------------------------------------------------
    %  SALVATAGGIO AUTOMATICO
    % -------------------------------------------------------
    runFolder = fullfile('Runs', RUN_NAME);
    if ~exist(runFolder, 'dir'), mkdir(runFolder); end

    save(fullfile(runFolder, 'agent_trained.mat'), 'agent');
    save(fullfile(runFolder, 'trainingStats.mat'), 'trainingStats');

    % Snapshot dell'ambiente usato per questo training
    scriptDir = fileparts(mfilename('fullpath'));
    copyfile(fullfile(scriptDir, 'RL_environment.m'), fullfile(runFolder, 'RL_environment.m'));

    % Calcolo Sorting Accuracy
    fprintf('\nCalcolo Sorting Accuracy su 10 episodi...\n');
    [meanAcc, stdAcc, allAcc] = evaluate_agent(env, agent, 10);

    % Salvataggio configurazione completa
    runConfig.name                  = RUN_NAME;
    runConfig.notes                 = RUN_NOTES;
    runConfig.rewardNotes           = REWARD_NOTES;
    runConfig.date                  = datestr(now);
    runConfig.ClipFactor            = agentOpts.ClipFactor;
    runConfig.ExperienceHorizon     = agentOpts.ExperienceHorizon;
    runConfig.EntropyLossWeight     = agentOpts.EntropyLossWeight;
    runConfig.LearnRate             = 3e-4;
    runConfig.NumEpoch              = agentOpts.NumEpoch;
    runConfig.GAEFactor             = agentOpts.GAEFactor;
    runConfig.DiscountFactor        = agentOpts.DiscountFactor;
    runConfig.SampleTime            = agentOpts.SampleTime;
    runConfig.criticLayerSizes      = criticLayerSizes;
    runConfig.actorLayerSizes       = actorLayerSizes;
    runConfig.numObservations       = numObs;
    runConfig.numActions            = prod(actInfo.Dimension);
    runConfig.maxEpisodes           = trainOpts.MaxEpisodes;
    runConfig.finalAvgReward        = mean(trainingStats.EpisodeReward(max(1,end-99):end));
    runConfig.peakAvgReward         = max(movmean(trainingStats.EpisodeReward, 100));
    runConfig.sortingAccuracy_mean  = meanAcc;
    runConfig.sortingAccuracy_std   = stdAcc;
    runConfig.sortingAccuracy_all   = allAcc;

    save(fullfile(runFolder, 'run_config.mat'), 'runConfig');

    % Export grafico training
    % Salvataggio grafico training (generato dai dati, non dalla finestra MATLAB)
    hFig = figure('Visible','off','Position',[100 100 1000 500]);
    episodes = 1:numel(trainingStats.EpisodeReward);
    movAvg  = movmean(trainingStats.EpisodeReward, 100);
    hold on;
    plot(episodes, trainingStats.EpisodeReward, 'Color',[0.6 0.8 1.0], 'LineWidth',0.5, 'DisplayName','Episode Reward');
    plot(episodes, movAvg, 'b-', 'LineWidth',2, 'DisplayName','Moving Avg (100 ep)');
    hold off;
    xlabel('Episodio'); ylabel('Reward');
    title(sprintf('%s — Training Curve', RUN_NAME), 'Interpreter','none');
    legend('Location','northwest');
    grid on;
    exportgraphics(hFig, fullfile(runFolder, 'training_curve.png'), 'Resolution', 300);
    close(hFig);
    fprintf('Grafico training salvato in %s/training_curve.png\n', runFolder);

    fprintf('\n========================================\n');
    fprintf(' Run:              %s\n',    RUN_NAME);
    fprintf(' Data:             %s\n',    runConfig.date);
    fprintf(' Episodi completati: %d/%d\n', numel(trainingStats.EpisodeReward), trainOpts.MaxEpisodes);
    fprintf(' Avg Reward:       %.1f\n',  runConfig.finalAvgReward);
    fprintf(' Peak Avg Reward:  %.1f\n',  runConfig.peakAvgReward);
    fprintf(' Sorting Accuracy: %.1f%% +/- %.1f%%\n', meanAcc, stdAcc);
    fprintf(' Salvato in:       %s/\n',   runFolder);
    fprintf('========================================\n');

%% =========================================================
%  MODALITA' VISUALIZZAZIONE  (loadSavedAgent == true)
% ==========================================================
else
    runFolder = fullfile('Runs', LOAD_FROM_RUN);

    % Carica snapshot ambiente del run specifico (se disponibile)
    if exist(fullfile(runFolder, 'RL_environment.m'), 'file')
        addpath(fullfile(pwd, runFolder));
        clear RL_environment;
        env = RL_environment();
        rmpath(fullfile(pwd, runFolder));
        fprintf('Ambiente caricato dalla snapshot del run.\n');
    else
        clear RL_environment;
        env = RL_environment();
        fprintf('Snapshot ambiente non trovata, uso l''ambiente corrente.\n');
    end

    load(fullfile(runFolder, 'agent_trained.mat'), 'agent');

    if exist(fullfile(runFolder, 'run_config.mat'), 'file')
        load(fullfile(runFolder, 'run_config.mat'), 'runConfig');
        fprintf('\nCaricato: %s\n', LOAD_FROM_RUN);
        fprintf('  Note:     %s\n', runConfig.notes);
        fprintf('  Reward:   %s\n', runConfig.rewardNotes);
        if isfield(runConfig, 'sortingAccuracy_mean')
            fprintf('  Accuracy: %.1f%%\n', runConfig.sortingAccuracy_mean);
        end
    end
end

%% =========================================================
%  VISUALIZZATORE FISICO
% ==========================================================
fprintf('\nPremi un tasto per avviare il visualizzatore fisico...\n');
pause;

env.reset();
plot(env);
rng(10);
simOpts = rlSimulationOptions('MaxSteps', floor(15/env.dt));
experience = sim(env, agent, simOpts);
