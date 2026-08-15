function [meanAccuracy, stdAccuracy, allAccuracies] = evaluate_agent(env, agent, nEpisodes)
%EVALUATE_AGENT Calcola la Sorting Accuracy su N episodi.
%
%  Replica la stessa logica di AMSVisualizer (gap a yEval = 2.0m)
%  in modo programmatico, senza aprire la finestra grafica.
%
%  Output:
%    meanAccuracy  - Media % di singolarizzazione su nEpisodes episodi
%    stdAccuracy   - Deviazione standard
%    allAccuracies - Vettore [nEpisodes x 1] con accuracy per episodio

yEval         = env.d_AMS * env.n_i_AMS + 0.05;  % 1.05m — stesso valore usato in getReward!
allAccuracies = zeros(nEpisodes, 1);

fprintf('\n--- Valutazione Sorting Accuracy (%d episodi) ---\n', nEpisodes);

for ep = 1:nEpisodes

    obs    = reset(env);
    isDone = false;

    % Stato di tracking — resettato ad ogni episodio
    eval_top_recorded = false(env.max_gen_boxes, 1);
    eval_order        = zeros(0, 1);
    eval_success      = 0;
    eval_pairs        = 0;

    while ~isDone
        action = getAction(agent, {obs});
        [obs, ~, isDone, ~] = step(env, action{1});

        % Controlla quali pacchi hanno attraversato yEval in questo step
        nActive = env.n_boxes_tot;
        for ii = 1:nActive
            if eval_top_recorded(ii),                             continue; end
            if env.x_boxes(ii) < 0 || env.y_boxes(ii) < 0,      continue; end
            if env.d_boxes(ii) <= 0,                              continue; end

            yTop = env.y_boxes(ii) + env.d_boxes(ii) / 2;

            if yTop >= yEval
                eval_top_recorded(ii) = true;

                % Calcola gap con il pacco precedente nella sequenza di uscita
                if ~isempty(eval_order)
                    prev       = eval_order(end);
                    bottomPrev = env.y_boxes(prev) - env.d_boxes(prev) / 2;
                    gap        = bottomPrev - yTop;   % >0 = singularizzato
                    eval_pairs = eval_pairs + 1;
                    if gap > 0
                        eval_success = eval_success + 1;
                    end
                end

                eval_order(end+1, 1) = ii; %#ok<AGROW>
            end
        end
    end  % while

    % Accuracy episodio (0% se nessuna coppia valutata)
    if eval_pairs > 0
        allAccuracies(ep) = 100 * eval_success / eval_pairs;
    else
        allAccuracies(ep) = 0;
    end

    fprintf('  Ep %2d/%d | Accuracy: %5.1f%% | Coppie: %d/%d\n', ...
        ep, nEpisodes, allAccuracies(ep), eval_success, eval_pairs);

end  % for ep

meanAccuracy = mean(allAccuracies);
stdAccuracy  = std(allAccuracies);

fprintf('----------------------------------------------\n');
fprintf('  MEDIA: %.1f%%  |  STD: %.1f%%  |  MIN: %.0f%%  |  MAX: %.0f%%\n', ...
    meanAccuracy, stdAccuracy, min(allAccuracies), max(allAccuracies));
fprintf('----------------------------------------------\n\n');

end
