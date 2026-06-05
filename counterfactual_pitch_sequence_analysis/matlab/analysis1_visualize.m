% Program code for visualizing the 6*6 probability map for each sample
clear
beep off

data_path = "./results/analysis1";
pitcher_folders = dir(fullfile(data_path,"202*"));

p_info=readtable("pitcher_info.xlsx");

%set the centers of the six-part grid
plate_x = [0.5833 0.35 0.1167 -0.1167 -0.35 -0.5833];
plate_z = [1.0 0.80 0.60 0.40 0.20 0.00];

%pitcher ID: 1~52
target_pitcher_id=5;

speed_thre_h=92.5;
speed_thre_m=86.7;
w_=1.25;

temp_name=pitcher_folders(target_pitcher_id).name;
p_name_temp=temp_name(6:end);

disp(p_name_temp)

idx=strcmp(p_info.Name, p_name_temp);

out_path_fig_temp=fullfile("./figures/last_pitch",p_name_temp);
if ~exist(out_path_fig_temp,"dir")
   mkdir(out_path_fig_temp)
end

rows=p_info(idx, :);

era=str2double(p_info(idx, :).era{:});
k_rate=str2double(p_info(idx, :).strikeoutsPer9Inn{:});
slg=str2double(p_info(idx, :).slg{:});

p_stats_temp=[era k_rate slg];

pitch_files=dir(fullfile(data_path,temp_name,"*.csv"));

data_path_adjustment = "./results/adjustment";
prob_high_st=readmatrix(fullfile(data_path_adjustment,"high_strike.csv"));
prob_mid_st=readmatrix(fullfile(data_path_adjustment,"middle_strike.csv"));
prob_low_st=readmatrix(fullfile(data_path_adjustment,"low_strike.csv"));

prob_st=cat(3, prob_high_st, prob_mid_st, prob_low_st);
prob_mean=mean(mean(mean(prob_st)));
prob_st=prob_st./prob_mean;

for pitch_id=1:1:5 %size(pitch_files,1)

    temp_pitch = readtable(fullfile(data_path,temp_name,pitch_files(pitch_id).name));
    pitch_type_num = length(unique(temp_pitch.pitch_type));

    original_data = horzcat(table2array(temp_pitch(1,10:11)), ...
                            table2array(temp_pitch(1,15)));
    original_ball_name=temp_pitch.pitch_type{1};

    previous_data = horzcat(table2array(temp_pitch(2,10:11)), ...
                            table2array(temp_pitch(2,15)));

    previous_ball_name=temp_pitch.pitch_type{2};    

    figure      
    for pitch_type_id = 1:1:pitch_type_num
        subplot(pitch_type_num - ceil(pitch_type_num/2), ...
                ceil(pitch_type_num/2), ...
                pitch_type_id)

        ball_name = temp_pitch.pitch_type{36*(pitch_type_id-1)+3};

        model_outputs = table2array( ...
            temp_pitch(36*(pitch_type_id-1)+3 : 36*pitch_type_id+2, 15));           
        model_outputs=reshape(model_outputs, 6, 6);
        model_outputs=flip(model_outputs);

        speed_temp=mean(temp_pitch.effective_speed(36*(pitch_type_id-1)+3 : 36*pitch_type_id+2));

        if speed_temp>speed_thre_h
           model_outputs=model_outputs+w_.*fliplr(prob_high_st./prob_mean);
        elseif speed_temp>speed_thre_m
           model_outputs=model_outputs+w_.*fliplr(prob_mid_st./prob_mean);
        else
           model_outputs=model_outputs+w_.*fliplr(prob_low_st./prob_mean);
        end   
        
        model_outputs = fliplr(model_outputs);

        imagesc(model_outputs);
        colormap(flipud(gray));
        clim([1.0 2.5]);
        colorbar;
        axis equal tight;
        xticks(1:6);
        yticks(1:6);            
        xticklabels(compose('%.2f', plate_x));
        yticklabels(string(plate_z));
        xlabel('x');
        ylabel('z');
        title(ball_name, 'Interpreter', 'none');
        set(gca, 'FontSize', 8);            
        for row = 1:6
            for col = 1:6
                text(col, row, sprintf('%.2f', model_outputs(row,col)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 8, ...
                    'Color', 'w');
            end
        end           

        % [~,rect_col]=min(abs(plate_x-(-1*original_data(1))));
        % [~,rect_row]=min(abs(plate_z-(original_data(2))));
        % 
        % if ball_name==original_ball_name
        %    rectangle('Position', ...
        %         [rect_col - 0.5, ...
        %          rect_row + 0.5, ...
        %          1, ...
        %          1], ...
        %         'EdgeColor', 'r', ...
        %         'LineWidth', 2);
        % end
      
        % [~,rect_col]=min(abs(plate_x-(-1*previous_data(1))));
        % [~,rect_row]=min(abs(plate_z-(previous_data(2))));
        % 
        % if ball_name==previous_ball_name
        %    rectangle('Position', ...
        %         [rect_col - 0.5, ...
        %          rect_row + 0.5, ...
        %          1, ...
        %          1], ...
        %         'EdgeColor', 'b', ...
        %         'LineWidth', 2);
        % end           

    end
end
