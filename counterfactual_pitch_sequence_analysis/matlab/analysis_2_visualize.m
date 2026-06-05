% Program code for visualizing the probability map for each sample
clear
beep off

data_path = "./results/analysis2";
pitcher_folders = dir(fullfile(data_path,"202*"));

p_info=readtable("pitcher_info.xlsx");

plate_width_half=8.5/12.0;
plate_x = (-1.4:0.2:1.4)*plate_width_half;
plate_z = -0.2:0.1:1.2;

%pitcher ID: 1~52
target_pitcher_id=5;

disp(target_pitcher_id)
temp_name=pitcher_folders(target_pitcher_id).name;
p_name_temp=temp_name(6:end);

idx=strcmp(p_info.Name, p_name_temp); 

out_path_fig_temp=fullfile("./figures/set_up",p_name_temp);
if ~exist(out_path_fig_temp,"dir")
   mkdir(out_path_fig_temp)
end

rows=p_info(idx, :);

pitch_files=dir(fullfile(data_path,temp_name,"*.csv"));
for pitch_id=1:1:5 %size(pitch_files,1)
    temp_pitch = readtable(fullfile(data_path,temp_name,pitch_files(pitch_id).name));
    pitch_type_num = length(unique(temp_pitch.pitch_type));

    last_data = horzcat(table2array(temp_pitch(2,11)), ...
                            table2array(temp_pitch(2,13)), ...
                            table2array(temp_pitch(2,17)));
    last_ball_name=temp_pitch.pitch_type{2};

    previous_data = horzcat(table2array(temp_pitch(1,11)), ...
                            table2array(temp_pitch(1,13)), ...
                            table2array(temp_pitch(1,17)));

    previous_ball_name=temp_pitch.pitch_type{1};    

    figure
    z_for_hist = zeros(15,15);
    for pitch_type_id = 1:1:pitch_type_num
        subplot(pitch_type_num - ceil(pitch_type_num/2), ...
                ceil(pitch_type_num/2), ...
                pitch_type_id)

        ball_name = temp_pitch.pitch_type{104*(pitch_type_id-1)+3};

        cf_data = table2array(temp_pitch(104*(pitch_type_id-1)+3 : 104*pitch_type_id+2, 17));

        x_ref=table2array(temp_pitch(104*(pitch_type_id-1)+3 : 104*pitch_type_id+2, 12)).*10+3;
        y_ref=table2array(temp_pitch(104*(pitch_type_id-1)+3 : 104*pitch_type_id+2, 13)).*10+3;    
      
        for loc_id=1:1:length(cf_data)
            z_for_hist(x_ref(loc_id),y_ref(loc_id))=cf_data(loc_id)-last_data(3);
        end

        hole_mask = false(15,15);        
        hole_mask(3:13, 3:13) = true;
        z_plot = z_for_hist;
        z_plot(hole_mask) = NaN;   
        z_plot = fliplr(z_plot);
            
        imagesc(z_plot);
        colormap(flipud(gray));
        clim([min(temp_pitch.model_output-last_data(3)) max(temp_pitch.model_output-last_data(3))]);
        colorbar;
        axis equal tight;
        xticks(1:15);
        yticks(1:15);            
        xticklabels(compose('%.1f', -0.2:0.1:1.2));
        yticklabels(compose('%.1f', flip(-0.2:0.1:1.2)));
        xlabel('x');
        ylabel('z');
        title(ball_name, 'Interpreter', 'none');    
        set(gca, "FontSize", 8)

        % mark the final pitch with a red square
        [~,rect_col]=min(abs(plate_x-(-1*last_data(1))));
        [~,rect_row]=min(abs(plate_z-(last_data(2))));
        rect_row=length(plate_z)-rect_row;            
        if ball_name==last_ball_name
           rectangle('Position', ...
                [rect_col - 0.5, ...
                 rect_row + 0.5, ...
                 1, ...
                 1], ...
                'FaceColor','red', ...
                'EdgeColor', 'r', ...
                'LineWidth', 2);
        end

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
