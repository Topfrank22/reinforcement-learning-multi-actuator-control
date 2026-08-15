classdef RL_curriculum_env < rl.env.MATLABEnvironment
    %AMS_RL: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        v_treadmill = 1.9; % m/s

        d_AMS = 0.2; % m - size of the AMS
        n_i_AMS = 5; % numeber of AMS along y
        n_j_AMS = 5; % number of AMS along x
        n_actions_art = 2; % number of actions of each AMS

        upperlim_rot = pi*45/180; % upper limit rotation for AMS action
        upperlim_v = 2.2; % max velocity for AMS
        lowerlim_rot = - pi*45/180; % lower limit rotation for AMS action
        lowerlim_v = 0.5; % min velocity for AMS

        vy_boxes = [];
        x_boxes_prec = [];
        y_boxes_prec = [];

        index_AMS = [];

        LastAction = zeros(50,1);

        i_box = [];
        j_box = [];

        n_boxes_tot_vect = [];
        n_boxes_tot = 0;
        max_gen_boxes = 10;
        curriculum_limit = 10;  % Quanti pacchi far spawnare al massimo (10)
        
        % Adaptive Curriculum (Self-Paced Learning) Properties
        curriculum_level = 2;   % Parte da 2 pacchi
        reward_history = zeros(1, 50);
        history_idx = 1;
        current_ep_reward = 0;
        
        cont_new_box = 0;

        d_boxes = zeros(10,1);
        x_boxes = zeros(10,1);
        y_boxes = zeros(10,1);
        x_boxes_vect = zeros(10000,10);
        y_boxes_vect = zeros(10000,10);
        cont = 0;
        pack_exited = zeros(10,1);
        exit_order = zeros(10,1);
        index_exit = 0;

        toll_contatto = 0.005;

        dt = 0.01; % s - timestep for simulation

        time = 0; % s - simulation time
    end

    properties (Hidden)
        % Flags for visualization
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false        
    end
    
    properties
        % Initialize system state [on1,x1,y1,on2,x2,y2,...,on25,x25,y25]'
        State = zeros(100,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = RL_curriculum_env()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([100 1]); % 10 pacchi x 10 feature ciascuno
            ObservationInfo.Name = 'System States';
            ObservationInfo.Description = 'Per ogni pacco: attivo, x, y, dim, vy, riga, col, uscito, ultimo_uscito, tempo';
            
            % Initialize Action settings
            n_actions = 5*5*2;
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Action';
            ActionInfo.Description = 'r1, v1, r2, v2, ...';
            ActionInfo.LowerLimit = zeros(n_actions,1);
            ActionInfo.UpperLimit = zeros(n_actions,1);

            for ii=1:2:n_actions

                ActionInfo.UpperLimit(ii) = pi*45/180;
                ActionInfo.UpperLimit(ii+1) = 2.2;
                ActionInfo.LowerLimit(ii) = - pi*45/180;
                ActionInfo.LowerLimit(ii+1) = 0.5;
                
            end
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];

            this.time = this.time + this.dt;

            this.cont = this.cont + 1;

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);

            for ii=1:2:50
                ActLimUp(ii) = this.upperlim_rot;
                ActLimUp(ii+1) = this.upperlim_v;
                ActLimLow(ii) = this.lowerlim_rot;
                ActLimLow(ii+1) = this.lowerlim_v;
            end

            % Actions are normalized [0-1]
            % De-normalizing the actions
            AMS_actions = ActLimLow + (1 + Action) .* (ActLimUp - ActLimLow)./2;
            for ii=1:50
                AMS_actions(ii) = max(ActLimLow(ii),min(ActLimUp(ii),AMS_actions(ii)));
            end

            this.LastAction = AMS_actions;

            v_AMS = zeros(5,5);
            rotation_AMS = zeros(5,5);

            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    % rotational actions
                    rotation_AMS(ii,jj) = AMS_actions(jj*2-1+(ii-1)*10);
                    % velocity actions
                    v_AMS(ii,jj) = AMS_actions(jj*2+(ii-1)*10);
                end
            end

            % new boxes generation each 0.75 s. 2 boxes are generated.

            if mod(round(this.time,2),0.75) == 0 && this.n_boxes_tot<this.curriculum_limit && this.time>0.25
                
                this.cont_new_box = this.cont_new_box + 1;
                
                gen_n_boxes = 2;
                if gen_n_boxes>2
                    gen_n_boxes=2;
                end

                while this.n_boxes_tot+gen_n_boxes>this.max_gen_boxes
                    gen_n_boxes = gen_n_boxes-1;
                end

                this.n_boxes_tot = this.n_boxes_tot+gen_n_boxes;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n_boxes;

                for ii=1:gen_n_boxes

                    if gen_n_boxes>1
                        if ii == 1
                            this.x_boxes(this.n_boxes_tot-1) = this.d_boxes(this.n_boxes_tot-1)*0.5 + (this.d_AMS*2.5-this.d_boxes(this.n_boxes_tot-1)*0.5)*rand(1);
                            this.y_boxes(this.n_boxes_tot-1) = 0.001 + rand(1)*0.01;
                            this.x_boxes_prec(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot-1);
                            this.y_boxes_prec(this.n_boxes_tot-1) = this.y_boxes(this.n_boxes_tot-1);
                        else
                            this.x_boxes(this.n_boxes_tot) = this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5))*rand(1);
                            this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.05;
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot-1) + this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1;
                            end
                            if this.x_boxes(this.n_boxes_tot)-this.d_boxes(this.n_boxes_tot)/2<0
                                this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)/2;
                            elseif this.x_boxes(this.n_boxes_tot)+this.d_boxes(this.n_boxes_tot)/2>l_AMS_matrix
                                this.x_boxes(this.n_boxes_tot) = l_AMS_matrix-this.d_boxes(this.n_boxes_tot)/2;
                            end
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot) - (this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1);
                            end
                            this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                            this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        end
                    else
                        this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)*0.55 + (l_AMS_matrix-this.d_boxes(this.n_boxes_tot)*0.55)*rand(1);
                        this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.01;
                        this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        this.x_boxes_vect(this.n_boxes_tot,this.cont-1) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_vect(this.n_boxes_tot,this.cont-1) = this.y_boxes(this.n_boxes_tot);
                    end

                end

            end

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = [];

            % checking collisions

            for ii=1:this.n_boxes_tot

                if this.x_boxes(ii)-this.d_boxes(ii)/2<0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;
                elseif this.x_boxes(ii)+this.d_boxes(ii)/2>l_AMS_matrix
                    this.x_boxes(ii) = l_AMS_matrix-this.d_boxes(ii)/2;
                end

                % identifying on which AMS the boxes are (geometrical baricenter)

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                % packages kinematics

                if this.i_box(ii) < 6
                    this.x_boxes(ii) = this.x_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.y_boxes(ii) = this.y_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.vy_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                else
                    this.x_boxes(ii) = this.x_boxes(ii);
                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                end

                % collisions

                if ii>1 && this.cont>1
                    for jj=ii:-1:2
                        if abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.x_boxes_vect(ii,this.cont-1)-this.x_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2
                            if (this.x_boxes(ii)-this.d_boxes(ii)/2 <= this.x_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.x_boxes(ii)-this.d_boxes(ii)/2>this.x_boxes(jj-1)-this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(jj-1)+this.d_boxes(jj-1)/2) - (this.x_boxes(ii)-this.d_boxes(ii)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) + penetrazione_x/2 + this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) - penetrazione_x/2 - this.toll_contatto;
                            elseif (this.x_boxes(ii)+this.d_boxes(ii)/2 >= this.x_boxes(jj-1)-this.d_boxes(jj-1)/2 && this.x_boxes(ii)+this.d_boxes(ii)/2<this.x_boxes(jj-1)+this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(ii)+this.d_boxes(ii)/2) - (this.x_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) - penetrazione_x/2 - this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) + penetrazione_x/2 + this.toll_contatto;
                            end
                        elseif abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                            if (this.y_boxes(ii)-this.d_boxes(ii)/2 <= this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)-this.d_boxes(ii)/2>this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2 <= this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)-this.d_boxes(jj-1)/2>this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2)-(this.y_boxes(ii)-this.d_boxes(ii)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) + penetrazione_y/2 + this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) - penetrazione_y/2 - this.toll_contatto;
                            elseif (this.y_boxes(ii)+this.d_boxes(ii)/2 < this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)+this.d_boxes(ii)/2 >= this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 < this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 >= this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(ii)+this.d_boxes(ii)/2) - (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) - penetrazione_y/2  - this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) + penetrazione_y/2 + this.toll_contatto;
                            end
                        end
                    end
                end
                
                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            % Observation: vettore 100 elementi (10 pacchi x 10 feature), valori normalizzati [0,1]
            Observation = zeros(100,1);
            for ii = 1:this.max_gen_boxes
                base = (ii-1)*10;
                if ii <= this.n_boxes_tot
                    Observation(base+1)  = 1;                                                  % Pacco attivo
                    Observation(base+2)  = this.x_boxes(ii) / (this.d_AMS * this.n_j_AMS);   % Pos X norm [0,1]
                    Observation(base+3)  = this.y_boxes(ii) / 2.0;                            % Pos Y norm [0,1]
                    Observation(base+4)  = this.d_boxes(ii) / (2 * this.d_AMS);               % Dim norm [0,1]
                    Observation(base+5)  = this.vy_boxes(ii) / 2.2;                           % Vel Y norm [0,1]
                    Observation(base+6)  = this.i_box(ii) / 6;                                % Riga norm [0,1]
                    Observation(base+7)  = this.j_box(ii) / 6;                                % Col norm [0,1]
                    Observation(base+8)  = this.pack_exited(ii);                               % Uscito? (0/1)
                    Observation(base+9)  = double(this.index_exit > 0 && ...
                                          this.exit_order(max(1,this.index_exit)) == ii);      % Ultimo uscito?
                    Observation(base+10) = this.time / 10;                                     % Tempo norm [0,1]
                end  % i pacchi non ancora spawnati restano a zero
            end


            % Update system states
            this.State = Observation;

            cont_exit = 0;

            % Check terminal condition
            for ii=1:this.n_boxes_tot
                if (this.y_boxes(ii) + this.d_boxes(ii)/2) >= (this.d_AMS*5 + 0.05)
                    cont_exit = cont_exit + 1;
                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                    end
                end
            end

            if cont_exit==this.curriculum_limit
                IsDone = true;
            else
                IsDone = false;
            end

            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this,AMS_actions);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
            
            % AGGIORNAMENTO HISTORY ALLA FINE DELLO STEP
            % Deve avvenire DOPO getReward, altrimenti y_boxes e y_boxes_prec sono uguali!
            for ii=1:this.max_gen_boxes
                if ii<=this.n_boxes_tot
                    this.x_boxes_vect(ii,this.cont) = this.x_boxes(ii);
                    this.y_boxes_vect(ii,this.cont) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(ii,this.cont) = -1;
                    this.y_boxes_vect(ii,this.cont) = -1;
                end
            end
            
            % Adaptive Curriculum Tracking
            this.current_ep_reward = this.current_ep_reward + Reward;
            
            if IsDone
                this.reward_history(this.history_idx) = this.current_ep_reward;
                this.history_idx = this.history_idx + 1;
                if this.history_idx > 50
                    this.history_idx = 1;
                end
                
                avg_rew = mean(this.reward_history);
                if this.curriculum_level == 2 && avg_rew > 1100
                    this.curriculum_level = 4;
                    this.reward_history = zeros(1, 50);
                    fprintf('\n>>> LEVEL UP! Passaggio a 4 pacchi! (Avg Reward: %.1f) <<<\n', avg_rew);
                elseif this.curriculum_level == 4 && avg_rew > 3300
                    this.curriculum_level = 6;
                    this.reward_history = zeros(1, 50);
                    fprintf('\n>>> LEVEL UP! Passaggio a 6 pacchi! (Avg Reward: %.1f) <<<\n', avg_rew);
                elseif this.curriculum_level == 6 && avg_rew > 5450
                    this.curriculum_level = 8;
                    this.reward_history = zeros(1, 50);
                    fprintf('\n>>> LEVEL UP! Passaggio a 8 pacchi! (Avg Reward: %.1f) <<<\n', avg_rew);
                elseif this.curriculum_level == 8 && avg_rew > 7700
                    this.curriculum_level = 10;
                    this.reward_history = zeros(1, 50);
                    fprintf('\n>>> LEVEL UP! Passaggio a 10 pacchi! (Avg Reward: %.1f) <<<\n', avg_rew);
                end
            end
        end
        
        % Reset environment to initial state and output initial observation
        % for each episod
        function InitialObservation = reset(this)
            
            this.curriculum_limit = this.curriculum_level;
            this.current_ep_reward = 0;

            this.index_exit = 0;
            this.pack_exited = zeros(10,1); % is package exited the AMS or not?
            this.exit_order = zeros(10,1); % packages ordered by exit

            this.cont_new_box = 1;

            this.n_boxes_tot_vect = 2;
            this.n_boxes_tot = this.n_boxes_tot_vect;

            this.time = 0;

            this.vy_boxes = zeros(10,1);

            this.cont = 0;

            max_d_box = 2.*this.d_AMS; % m
            min_d_box = 0.25*this.d_AMS; % m

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            % generating initial boxes (2)
            for ii=1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box-min_d_box)*rand(1);
            end

            for ii=1:2
                if ii == 1
                    x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5-this.d_boxes(1)*0.5)*rand(1);
                else
                    x2 = this.d_AMS*2.5+this.d_boxes(2)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(2)*0.5))*rand(1);
                    if abs(x2-x1) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                        x2 = x1 + this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1;
                    end
                    if x2-this.d_boxes(2)/2<0
                        x2 = this.d_boxes(2)/2;
                    elseif x2+this.d_boxes(2)/2>l_AMS_matrix
                        x2 = l_AMS_matrix-this.d_boxes(2)/2;
                    end
                    if abs(x2-x1) <= this.d_boxes(2)/2+this.d_boxes(1)/2
                        x1 = x2 - (this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1);
                    end
                end

                y1 = 0.001 + rand(1)*0.01;
                y2 = 0.001 + rand(1)*0.05;
            end

            this.x_boxes(1) = x1;
            this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;
            this.y_boxes(2) = y2;

            this.x_boxes_prec(1) = x1;
            this.x_boxes_prec(2) = x2;
            this.y_boxes_prec(1) = y1;
            this.y_boxes_prec(2) = y2;

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = 0;
                this.y_boxes(ii) = 0;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = zeros(2,1);

            % identifying on which AMS the boxes are (geometrical
            % baricenter)

            for ii=1:this.n_boxes_tot

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end

            end

            % InitialObservation: vettore 100 elementi (10 pacchi x 10 feature), valori normalizzati [0,1]
            InitialObservation = zeros(100,1);
            for ii = 1:this.max_gen_boxes
                base = (ii-1)*10;
                if ii <= this.n_boxes_tot
                    InitialObservation(base+1)  = 1;
                    InitialObservation(base+2)  = this.x_boxes(ii) / (this.d_AMS * this.n_j_AMS);
                    InitialObservation(base+3)  = this.y_boxes(ii) / 2.0;
                    InitialObservation(base+4)  = this.d_boxes(ii) / (2 * this.d_AMS);
                    InitialObservation(base+5)  = this.vy_boxes(ii) / 2.2;
                    InitialObservation(base+6)  = this.i_box(ii) / 6;
                    InitialObservation(base+7)  = this.j_box(ii) / 6;
                    InitialObservation(base+8)  = this.pack_exited(ii);
                    InitialObservation(base+9)  = double(this.index_exit > 0 && ...
                                                  this.exit_order(max(1,this.index_exit)) == ii);
                    InitialObservation(base+10) = this.time / 10;
                end
            end
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);

        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods  

        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            % Reset Visualizations
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end

        % Reward function
        function Reward = getReward(this, AMS_actions)
            Reward = 0;
            yExit = this.d_AMS * this.n_i_AMS;  % 1.0m — fine griglia rulli
            yEval = yExit + 0.05;                % 1.05m — Reward immediato appena esce dalla griglia

            % 1. FORWARD PROGRESS — piccolo premio per avanzare verso l'uscita
            for ii = 1:this.n_boxes_tot
                if this.pack_exited(ii) == 0 && this.i_box(ii) < 6
                    dy = this.y_boxes(ii) - this.y_boxes_prec(ii);
                    Reward = Reward + dy * 5;
                end
            end

            % 2. EXIT BONUS — grande premio quando un pacco esce dalla griglia
            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) > yExit && this.y_boxes_prec(ii) <= yExit
                    Reward = Reward + 100;
                end
            end

            % 3. SINGULATION REWARD — specchio della logica di AMSVisualizer
            %    Quando un pacco supera la linea yEval (1.05m), misuriamo
            %    il gap con il pacco IMMEDIATAMENTE DAVANTI. Gap>0: singolarizzato! Gap<=0: sovrapposto.
            for ii = 1:this.n_boxes_tot
                yTop     = this.y_boxes(ii)      + this.d_boxes(ii)/2;
                yTopPrec = this.y_boxes_prec(ii) + this.d_boxes(ii)/2;
                
                if yTop >= yEval && yTopPrec < yEval  % Ha appena attraversato yEval
                    % Trova il pacco *immediatamente* davanti a ii
                    target_jj = 0;
                    min_dist = inf;
                    
                    for jj = 1:this.n_boxes_tot
                        if jj ~= ii
                            jTop = this.y_boxes(jj) + this.d_boxes(jj)/2;
                            % Consideriamo jj solo se è FISICAMENTE davanti a ii
                            if jTop >= yTop
                                % Se sono appaiati esatti, usiamo l'indice per evitare doppi controlli
                                if jTop == yTop && jj < ii
                                    continue;
                                end
                                dist = jTop - yTop;
                                if dist < min_dist
                                    min_dist = dist;
                                    target_jj = jj;
                                end
                            end
                        end
                    end
                    
                    % Se target_jj > 0, significa che non è il primissimo pacco, quindi valutiamo il gap
                    if target_jj > 0
                        gap = (this.y_boxes(target_jj) - this.d_boxes(target_jj)/2) - yTop;
                        if gap > 0
                            safe_bonus = min(gap, 0.3) * 500; % Bonus fino a +150 punti se distanziati bene
                            Reward = Reward + 1000 + safe_bonus;  % Singolarizzato con successo!
                        else
                            Reward = Reward - 2000;  % Sovrapposto! Penalità catastrofica
                        end
                    end
                end
            end

            % 4. LATERAL SPREAD — premia il gap FISICO tra i bordi laterali (non la distanza tra centri)
            %    gap_x > 0: c'e' spazio fisico. gap_x = 0: si toccano (no premio). Cappato a 0.2m.
            d_gap_cap = 0.20;  % cap sul gap fisico: 20cm di aria e' sufficiente
            for ii = 1:this.n_boxes_tot
                for jj = ii+1:this.n_boxes_tot
                    if this.pack_exited(ii)==0 && this.pack_exited(jj)==0
                        % Gap fisico tra i bordi: distanza centri - somma dei raggi
                        gap_x = abs(this.x_boxes(ii) - this.x_boxes(jj)) ...
                                - (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                        gap_x = max(gap_x, 0);  % se sovrapposti = 0, non premiare
                        Reward = Reward + min(gap_x, d_gap_cap) * 1.5;
                    end
                end
            end

            % 5. LONGITUDINAL SPREAD — premia il gap FISICO tra i bordi in Y (fila indiana)
            %    Questo incentiva a separare i pacchi fin dal primo rullo, non all'ultimo momento.
            %    Due casse che si toccano hanno gap_y=0 e non guadagnano nulla.
            for ii = 1:this.n_boxes_tot
                for jj = ii+1:this.n_boxes_tot
                    if this.pack_exited(ii)==0 && this.pack_exited(jj)==0
                        % Gap fisico tra i bordi: distanza centri - somma dei raggi
                        gap_y = abs(this.y_boxes(ii) - this.y_boxes(jj)) ...
                                - (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                        gap_y = max(gap_y, 0);  % se sovrapposti = 0, non premiare
                        Reward = Reward + min(gap_y, d_gap_cap) * 2.0;
                    end
                end
            end

            % 6. VELOCITY GRADIENT REWARD (Early Separation Incentive)
            %    Se due pacchi sono troppo vicini in Y (< 0.2m), premia la differenza di velocità
            %    che tende a separarli (pacco davanti va più veloce del pacco dietro).
            %    Compensa la penalità di tempo e incoraggia a creare subito il gap.
            for ii = 1:this.n_boxes_tot
                for jj = ii+1:this.n_boxes_tot
                    if this.pack_exited(ii)==0 && this.pack_exited(jj)==0
                        % Gap fisico in Y tra i bordi
                        gap_y = abs(this.y_boxes(ii) - this.y_boxes(jj)) ...
                                - (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                        
                        % Solo se sono a rischio collisione/non singolarizzati
                        if gap_y < 0.2
                            if this.y_boxes(ii) > this.y_boxes(jj)
                                v_front = this.vy_boxes(ii);
                                v_back  = this.vy_boxes(jj);
                            else
                                v_front = this.vy_boxes(jj);
                                v_back  = this.vy_boxes(ii);
                            end
                            
                            delta_v = v_front - v_back;
                            if delta_v > 0
                                % "Sussidio di frenata": 0.3 annulla perfettamente la Time Penalty
                                % permettendo all'agente di separare i pacchi gratis!
                                Reward = Reward + delta_v * 0.3;
                            end
                        end
                    end
                end
            end

            % 7. TIME PENALTY — ripristinata a -0.3 per mantenere il bilancio in positivo
            Reward = Reward - 0.3;

            % 8. WALL PENALTY — penalità se i pacchi toccano o si avvicinano ai bordi laterali
            l_AMS_matrix = this.d_AMS * this.n_j_AMS; % 1.0m
            d_prox = 0.05;  % zona di pericolo: 5cm dal muro
            for ii = 1:this.n_boxes_tot
                if this.pack_exited(ii) == 0
                    dist_left  = this.x_boxes(ii) - this.d_boxes(ii)/2;          % distanza dal muro sinistro
                    dist_right = l_AMS_matrix - (this.x_boxes(ii) + this.d_boxes(ii)/2); % distanza dal muro destro
                    dist_wall  = min(dist_left, dist_right);                      % distanza dal muro piu' vicino

                    if dist_wall <= 0
                        % Contatto fisico con il muro
                        Reward = Reward - 2.0;
                    elseif dist_wall < d_prox
                        % Zona di pericolo: penalita' graduata proporzionale alla vicinanza
                        Reward = Reward - 1.0 * (1 - dist_wall / d_prox);
                    end
                end
            end
        end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',100},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
