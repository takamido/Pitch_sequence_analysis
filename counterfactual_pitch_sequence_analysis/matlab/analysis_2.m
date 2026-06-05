% Program code for running Analysis 2
% before running these programs, please save the outputs from the 
% corresponding notebooks above in the ./results folder under the names analysis2

clear
beep off

data_path = "./results/analysis2";
pitcher_folders = dir(fullfile(data_path,"202*"));

p_info=readtable("pitcher_info.xlsx");

pitcher_stats_all=[];
model_outputs_all=[];
model_outputs_replace_all=[];

loc_all="";
speed_all="";
combination_all="";

speed_thre_h=92.5;
speed_thre_m=86.7;

% set window size (4: wide, 8: narrow)
zone_number=4;
count_all=0;
for p_id=1:1:size(pitcher_folders,1)
    disp(p_id)
    temp_name=pitcher_folders(p_id).name;
    p_name_temp=temp_name(6:end);

    idx=strcmp(p_info.Name, p_name_temp);
    if sum(idx)==0
       continue
    end

    rows=p_info(idx, :);
    
    era=str2double(p_info(idx, :).era{:});
    k_rate=str2double(p_info(idx, :).strikeoutsPer9Inn{:});
    slg=str2double(p_info(idx, :).slg{:});

    p_stats_temp=[k_rate era slg];
    
    pitch_files=dir(fullfile(data_path,temp_name,"*.csv"));

    mode_output_replace_temp=zeros(size(pitch_files,1),1);   
    mode_output_temp=zeros(1,size(pitch_files,1));
    for pitch_id=1:1:size(pitch_files,1)        
        temp_pitch = readtable(fullfile(data_path,temp_name,pitch_files(pitch_id).name));
        pitch_type_num = length(unique(temp_pitch.pitch_type));

        original_data = horzcat(table2array(temp_pitch(1,10:11)), ...
                                table2array(temp_pitch(1,17)));

        original_last_data = horzcat(table2array(temp_pitch(2,10:11)), ...
                                table2array(temp_pitch(2,17)));        

       count_all=count_all+1;
       if original_data(1)<0 && original_data(2) >=0.50
           loc_all(count_all,1)="HI";
       elseif original_data(1)>=0 && original_data(2) >=0.50
           loc_all(count_all,1)="HO";  
       elseif original_data(1)>=0 && original_data(2) < 0.50
           loc_all(count_all,1)="LO";
       else
           loc_all(count_all,1)="LI";
       end

       if original_last_data(1)<0 && original_last_data(2) >=0.50
           loc_all(count_all,3)="HI";
       elseif original_last_data(1)>=0 && original_last_data(2) >=0.50
           loc_all(count_all,3)="HO";  
       elseif original_last_data(1)>=0 && original_last_data(2) < 0.50
           loc_all(count_all,3)="LO";
       else
           loc_all(count_all,3)="LI";
       end       

        original_speed=table2array(temp_pitch(1,6));
        if original_speed>speed_thre_h
           speed_all(count_all,1)="H";
        elseif original_speed>speed_thre_m
           speed_all(count_all,1)="M";
        else 
           speed_all(count_all,1)="L";
        end      

        original_last_speed=table2array(temp_pitch(2,6));
        if original_last_speed>speed_thre_h
           speed_all(count_all,3)="H";
        elseif original_last_speed>speed_thre_m
           speed_all(count_all,3)="M";
        else 
           speed_all(count_all,3)="L";
        end 

        combination_all(count_all,1)=append(speed_all(count_all,1),"-",loc_all(count_all,1));
        combination_all(count_all,3)=append(speed_all(count_all,3),"-",loc_all(count_all,3));

        mode_output_temp(pitch_id)=original_data(3);   
        min_z_all=zeros(1,pitch_type_num);
        ball_loc_temp="";
        ball_speed_temp="";

        for pitch_type_id = 1:1:pitch_type_num            
            if zone_number==8
               zone1=[];zone2=[];zone3=[];zone4=[];
               zone5=[];zone6=[];zone7=[];zone8=[];
               zone_name=["HI" "HM" "HO" "MO" "LO" "LM" "LI" "MI"];
               for i=2+104*(pitch_type_id-1)+1:1:2+104*pitch_type_id
                   if temp_pitch.plate_x_norm(i)<=0.2 && temp_pitch.plate_z_norm(i)>=0.8
                      zone1=horzcat(zone1,temp_pitch.model_output(i)); 
                   elseif temp_pitch.plate_x_norm(i)>0.2 && temp_pitch.plate_x_norm(i)<0.8 && temp_pitch.plate_z_norm(i)>1.0
                      zone2=horzcat(zone2,temp_pitch.model_output(i));
                   elseif temp_pitch.plate_x_norm(i)>=0.8 && temp_pitch.plate_z_norm(i)>=0.8
                      zone3=horzcat(zone3,temp_pitch.model_output(i)); 
                   elseif temp_pitch.plate_x_norm(i)>1.0 && temp_pitch.plate_z_norm(i)<0.8 && temp_pitch.plate_z_norm(i)>0.2
                      zone4=horzcat(zone4,temp_pitch.model_output(i)); 
                   elseif temp_pitch.plate_x_norm(i)>=0.8 && temp_pitch.plate_z_norm(i)<=0.2
                      zone5=horzcat(zone5,temp_pitch.model_output(i));
                   elseif temp_pitch.plate_x_norm(i)>0.2 && temp_pitch.plate_x_norm(i)<0.8 && temp_pitch.plate_z_norm(i)<0.0
                      zone6=horzcat(zone6,temp_pitch.model_output(i));
                   elseif temp_pitch.plate_x_norm(i)<=0.2 && temp_pitch.plate_z_norm(i)<=0.2
                      zone7=horzcat(zone7,temp_pitch.model_output(i));
                   elseif temp_pitch.plate_x_norm(i)<0.0 && temp_pitch.plate_z_norm(i)<0.8 && temp_pitch.plate_z_norm(i)>0.2
                      zone8=horzcat(zone8,temp_pitch.model_output(i));
                   end                   
               end
               [min_z_all(pitch_type_id) min_index]=min([mean(zone1) mean(zone2) mean(zone3) mean(zone4) ...
                    mean(zone5) mean(zone6) mean(zone7) mean(zone8)]);
               ball_loc_temp(pitch_type_id)=zone_name(min_index);
            end

            if zone_number==4
               zone1=[];zone2=[];zone3=[];zone4=[];
               zone_name=["HI" "HO" "LO" "LI"];       
               for i=2+104*(pitch_type_id-1)+1:1:2+104*pitch_type_id
                   if temp_pitch.plate_x_norm(i)<0.5 && temp_pitch.plate_z_norm(i)>=0.5
                      zone1=horzcat(zone1,temp_pitch.model_output(i)); 
                   elseif temp_pitch.plate_x_norm(i)>=0.5 && temp_pitch.plate_z_norm(i)>=0.5
                      zone2=horzcat(zone2,temp_pitch.model_output(i));  
                   elseif temp_pitch.plate_x_norm(i)>=0.5 && temp_pitch.plate_z_norm(i)<0.5
                      zone3=horzcat(zone3,temp_pitch.model_output(i));
                   elseif temp_pitch.plate_x_norm(i)<0.5 && temp_pitch.plate_z_norm(i)<0.5
                      zone4=horzcat(zone4,temp_pitch.model_output(i));
                   end
               end
               [min_z_all(pitch_type_id) min_index]=min([mean(zone1) mean(zone2) mean(zone3) mean(zone4)]);    
               ball_loc_temp(pitch_type_id)=zone_name(min_index);
            end  

            speed_temp=mean(temp_pitch.effective_speed(36*(pitch_type_id-1)+3 : 36*pitch_type_id+2));
            if speed_temp>speed_thre_h
               ball_speed_temp(pitch_type_id,1)="H";
            elseif speed_temp>speed_thre_m
               ball_speed_temp(pitch_type_id,1)="M"; 
            else
               ball_speed_temp(pitch_type_id,1)="L"; 
            end    
        end

       [~,min_index]=min(min_z_all);
       loc_all(count_all,2)=ball_loc_temp(min_index);  
       speed_all(count_all,2)=ball_speed_temp(min_index);
       combination_all(count_all,2)=append(speed_all(count_all,2),"-",loc_all(count_all,2));       

        mode_output_replace_temp(pitch_id)=min(min_z_all);  
    end

   pitcher_stats_all=vertcat(pitcher_stats_all,p_stats_temp);

   model_outputs_replace_all=vertcat(model_outputs_replace_all,mean(mode_output_replace_temp));
   model_outputs_all=vertcat(model_outputs_all,mean(mode_output_temp)); 
end

%% Summarize the results (for wide version: zone_number=4)

all_change="";
for i=1:1:length(combination_all)
    if loc_all(i,1)==loc_all(i,3) && speed_all(i,1)==speed_all(i,3)
       all_change(i,1)="same";
    elseif loc_all(i,1)==loc_all(i,3)
       all_change(i,1)="speed"; 
    elseif speed_all(i,1)==speed_all(i,3)
       all_change(i,1)="loc"; 
    else
       all_change(i,1)="both";  
    end   

    if loc_all(i,2)==loc_all(i,3) && speed_all(i,2)==speed_all(i,3)
       all_change(i,2)="same";
    elseif loc_all(i,2)==loc_all(i,3)
       all_change(i,2)="speed"; 
    elseif speed_all(i,2)==speed_all(i,3)
       all_change(i,2)="loc"; 
    else
       all_change(i,2)="both";  
    end     
end

summary(categorical(all_change(:,1)))
summary(categorical(all_change(:,2)))

speed_comb="";
for i=1:1:length(combination_all)
    speed_comb(i,1)=append(speed_all(i,1),"-",speed_all(i,3));
    speed_comb(i,2)=append(speed_all(i,2),"-",speed_all(i,3));
end
summary(categorical(speed_comb(:,1)))
summary(categorical(speed_comb(:,2)))

loc_comb="";
for i=1:1:length(combination_all)
    loc_comb(i,1)=append(loc_all(i,1),"-",loc_all(i,3));
    loc_comb(i,2)=append(loc_all(i,2),"-",loc_all(i,3));
end
summary(categorical(loc_comb(:,1)))
summary(categorical(loc_comb(:,2)))

%% Analyze the impact on season-level statistics

% Use the same coefficients as in Analysis 1
a_analysis1=[-21.5111 7.7877 0.4184];
stats_increase_exp=zeros(1,3);
for i=1:1:3
    X_data=model_outputs_all;
    Y_data=pitcher_stats_all(:,i);

    stats_increase_exp(i)=a_analysis1(i)*mean(model_outputs_replace_all-model_outputs_all);
end

disp(stats_increase_exp)