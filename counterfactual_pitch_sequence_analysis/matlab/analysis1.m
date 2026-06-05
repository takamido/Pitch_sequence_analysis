% Program code for running Analysis 1
% before running these programs, please save the outputs from the 
% corresponding notebooks above in the ./results folder under the names analysis1

clear
beep off

% set the path for the data
data_path = "./results/analysis1";
pitcher_folders = dir(fullfile(data_path,"202*"));

p_info=readtable("pitcher_info.xlsx");

data_path_adjustment = "./results/adjustment";
prob_high_st=readmatrix(fullfile(data_path_adjustment,"high_strike.csv"));
prob_mid_st=readmatrix(fullfile(data_path_adjustment,"middle_strike.csv"));
prob_low_st=readmatrix(fullfile(data_path_adjustment,"low_strike.csv"));

prob_st=cat(3, prob_high_st, prob_mid_st, prob_low_st);
prob_mean=mean(mean(mean(prob_st)));
prob_st=prob_st./prob_mean;

prob_high_ba=readmatrix(fullfile(data_path_adjustment,"high_ball.csv"));
prob_mid_ba=readmatrix(fullfile(data_path_adjustment,"middle_ball.csv"));
prob_low_ba=readmatrix(fullfile(data_path_adjustment,"low_ball.csv"));

prob_ba=cat(3,prob_high_ba,prob_mid_ba,prob_low_ba);
prob_ba=prob_ba./prob_mean;

% set sindow size (3 or 4, 5)
window_size=3;

speed_thre_h=92.5;
speed_thre_m=86.7;

pitcher_stats_all=zeros(size(pitcher_folders,1),3);
model_outputs_all=zeros(size(pitcher_folders,1),1);
loc_all=[];type_all=[];
speed_all=[];combination_all=[];

original_x=[];
original_z=[];
original_out=[];
original_out_adj=[];

model_outputs_replace_all=zeros(size(pitcher_folders,1),1);

% set adjustment weight (w_slg)
w_=1.25;
for p_id=1:1:size(pitcher_folders,1)    
    disp(p_id)
    temp_name=pitcher_folders(p_id).name;
    p_name_temp=temp_name(6:end);

    idx=strcmp(p_info.Name, p_name_temp);
    rows=p_info(idx, :);
    
    k_rate=str2double(p_info(idx, :).strikeoutsPer9Inn{:});
    era=str2double(p_info(idx, :).era{:});
    slg=str2double(p_info(idx, :).slg{:});
    
    pitch_files=dir(fullfile(data_path,temp_name,"*.csv"));

    ml_output_pitcher=zeros(size(pitch_files,1),1); 
    
    ball_loc_pitcher=""; 
    ball_type_pitcher=""; 
    ball_speed_pitcher=""; 
    combinations_pitcher="";  
    strike_count=1;

    ml_output_replace=zeros(size(pitch_files,1),1); 
    for pitch_id=1:1:size(pitch_files,1)
        
        % load target pitch information
        temp_pitch = readtable(fullfile(data_path,temp_name,pitch_files(pitch_id).name));
        
        % set original pitch information
        original_data = horzcat(table2array(temp_pitch(1,10:11)), ...
                                table2array(temp_pitch(1,15)));
        original_x=vertcat(original_x,original_data(1));
        original_z=vertcat(original_z,original_data(2));       
        original_out=vertcat(original_out,original_data(3));

        ball_flag=0; 
        if abs(original_data(1))>0.83 || original_data(2) > 1.0 || original_data(2) < 0.0
           ball_flag=1; 
           if original_data(2) > 1.0; loc_z=1; 
           else; loc_z=2;
           end
           if original_data(1) > 0.83; loc_x=1; 
           else; loc_x=2;
           end           
        else
           if original_data(1) >= 0.553; loc_x=1; 
           elseif original_data(1) >= 0.276; loc_x=2;  
           elseif original_data(1) >= 0.0; loc_x=3; 
           elseif original_data(1) >= -0.276; loc_x=4; 
           elseif original_data(1) >= -0.553; loc_x=5;     
           else; loc_x=6; 
           end

           if original_data(2)>=0.83; loc_z=1;
           elseif original_data(2)>=0.66; loc_z=2; 
           elseif original_data(2)>=0.50; loc_z=3;
           elseif original_data(2)>=0.33; loc_z=4; 
           elseif original_data(2)>=0.16; loc_z=5;
           else; loc_z=6;
           end
        end

        loc_temp=[];
        if original_data(1)<0 && original_data(2) >=0.50
           loc_temp="HI";
        elseif original_data(1)>=0 && original_data(2) >=0.50
           loc_temp="HO";  
        elseif original_data(1)>=0 && original_data(2) < 0.50
           loc_temp="LO";
        else
           loc_temp="LI";
        end

        original_ball_name=temp_pitch.pitch_type{1};

        original_speed=table2array(temp_pitch(1,5));
        if original_speed>speed_thre_h
           sp_index=1;speed_name_ori="H";
        elseif original_speed>speed_thre_m
           sp_index=2;speed_name_ori="M";
        else 
           sp_index=3;speed_name_ori="L";
        end

        if ball_flag==0
           exp_slg=prob_st(loc_x,loc_z,sp_index);
        else
           exp_slg=prob_ba(loc_x,loc_z,sp_index); 
        end

        out_adj=original_data(3)+w_*exp_slg;        

        ml_output_pitcher(pitch_id,1)=out_adj;

        pitch_types_temp=unique(temp_pitch.pitch_type);

        % Remove "PO (Pitch Out)" from the analysis
        po_idx=0;
        if sum(pitch_types_temp=="PO")==1
           po_idx = find(contains(pitch_types_temp, "PO"));
        end

        pitch_type_num = length(pitch_types_temp); 
        
        best_value_temp=inf;
        best_ball_name_temp=[];
        best_loc_name_temp=[];
        best_ball_speed_temp=[];

        loc_names=["HI" "LI" "LO" "HO"];
        for pitch_type_id = 1:1:pitch_type_num
            if pitch_type_id==po_idx
               continue
            end

            ball_name_temp = temp_pitch.pitch_type{36*(pitch_type_id-1)+3};
            
            %calculate 6*6 probability map
            model_outputs = table2array( ...
                temp_pitch(36*(pitch_type_id-1)+3 : 36*pitch_type_id+2, 15));           
            model_outputs=reshape(model_outputs, 6, 6);
            model_outputs=flip(model_outputs);

            speed_temp=mean(temp_pitch.effective_speed(36*(pitch_type_id-1)+3 : 36*pitch_type_id+2));

            % w_slg asjustment
            if speed_temp>speed_thre_h
               model_outputs=model_outputs+w_.*fliplr(prob_high_st./prob_mean);
               speed_name_temp="H";
            elseif speed_temp>speed_thre_m
               model_outputs=model_outputs+w_.*fliplr(prob_mid_st./prob_mean);
               speed_name_temp="M"; 
            else
               model_outputs=model_outputs+w_.*fliplr(prob_low_st./prob_mean);
               speed_name_temp="L"; 
            end            

            % calculate mean output in each area
            model_outputs_regions=[mean(mean(model_outputs(1:window_size,1:window_size))); ...
                mean(mean(model_outputs(1:window_size,6-window_size+1:6))); ...
                mean(mean(model_outputs(6-window_size+1:6,6-window_size+1:6))); ...
                mean(mean(model_outputs(6-window_size+1:6,1:window_size)))];             

           % find the minimum area
           [min_value_temp min_loc]=min(model_outputs_regions);

           % update the result
           if min_value_temp<best_value_temp
              best_value_temp=min_value_temp; 
              best_ball_speed_temp=speed_name_temp;
              best_loc_name_temp=loc_names(min_loc);
              best_ball_name_temp=string(ball_name_temp);
           end
        end

        % target pitches in the strike zone; leave balls unchanged
        if ~(abs(original_data(1))>0.83 || original_data(2)<0.0 || original_data(2)>1.0)
           ml_output_replace(pitch_id,1)=best_value_temp;

           combinations_pitcher(strike_count,1)=append(speed_name_ori,"_",loc_temp);
           ball_speed_pitcher(strike_count,1)=speed_name_ori;        
           ball_loc_pitcher(strike_count,1)=loc_temp; 
           ball_type_pitcher(strike_count,1)=string(original_ball_name); 

           ball_loc_pitcher(strike_count,2)=best_loc_name_temp; 
           ball_type_pitcher(strike_count,2)=best_ball_name_temp; 
           ball_speed_pitcher(strike_count,2)=best_ball_speed_temp; 
           combinations_pitcher(strike_count,2)=append(best_ball_speed_temp,"-",best_loc_name_temp);

           strike_count=strike_count+1;
        else
           ml_output_replace(pitch_id,1)=out_adj; 
        end
    end

    pitcher_stats(p_id,1)=k_rate;
    pitcher_stats(p_id,2)=era;
    pitcher_stats(p_id,3)=slg;

    model_outputs_all(p_id,1)=mean(ml_output_pitcher);

    loc_all=vertcat(loc_all,ball_loc_pitcher);
    type_all=vertcat(type_all,ball_type_pitcher);
    speed_all=vertcat(speed_all,ball_speed_pitcher);
    combination_all=vertcat(combination_all,combinations_pitcher);

    model_outputs_replace_all(p_id,1)=mean(ml_output_replace);
end

summary(categorical(type_all(:,1)))
summary(categorical(type_all(:,2)))

summary(categorical(loc_all(:,1)))
summary(categorical(loc_all(:,2)))

summary(categorical(speed_all(:,1)))
summary(categorical(speed_all(:,2)))
 
summary(categorical(combination_all(:,1)))
summary(categorical(combination_all(:,2)))

%% Analyze the impact on season-level statistics

stats_name=["K/9", "ERA", "oSLG"];
stats_increase_exp=zeros(1,3);
for i=1:1:3
    X_data=model_outputs_all;
    Y_data=pitcher_stats(:,i);
    
    figure
    hold on
    scatter(X_data, Y_data,200,"k","filled")
    
    xlabel("Mean model output of each pitcher")
    ylabel(append(stats_name(i)))
    
    p = polyfit(X_data, Y_data, 1);
    a = p(1);
    b = p(2);
    
    disp(corrcoef(X_data, Y_data))
    
    X_fit = linspace(min(X_data), max(X_data), 100);
    Y_fit = polyval(p, X_fit);
    
    plot(X_fit, Y_fit, "r", 'LineWidth', 2.0);
    
    set(gca,"Fontsize", 15)
    
    % calculate expected improvements
    stats_increase_exp(i)=a*mean(model_outputs_replace_all-model_outputs_all);
end

disp(stats_increase_exp)

