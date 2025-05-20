function OpenSimIK()
    tic
    baseDir = 'F:\AlterG\Control\Data';
    contents = dir(baseDir);
    dirFlags = [contents.isdir];
    subfolders = contents(dirFlags);
    % Generate Scaled things 
    for k = 1 : length(subfolders)
        subfolderName = subfolders(k).name;

        if strcmp(subfolderName, '.') || strcmp(subfolderName, '..')
            continue;
        end
        
        SetupPath = fullfile(baseDir, subfolderName, 'Setup');
        IKResultsPath = fullfile(baseDir, subfolderName, 'IKResults');
        GaitPath = fullfile(baseDir, subfolderName, 'Gait');
        ScaledModelPath = fullfile(baseDir, subfolderName, 'ScaledModel');


        %Check for the existance of the scaled model if not scale it
        %if exist 

        scaledModelFiles = dir(fullfile(ScaledModelPath, '*.osim'));
        if isempty(scaledModelFiles) 
            OpenScale(baseDir, subfolderName)
        else
            disp(['Participant ',subfolderName,' Scaled'])
        end

    end


    for k = 1 : length(subfolders)
        subfolderName = subfolders(k).name;

        if strcmp(subfolderName, '.') || strcmp(subfolderName, '..')
            continue;
        end
        
        SetupPath = fullfile(baseDir, subfolderName, 'Setup');
        IKResultsPath = fullfile(baseDir, subfolderName, 'IKResults');
        GaitPath = fullfile(baseDir, subfolderName, 'Gait');
        ScaledModelPath = fullfile(baseDir, subfolderName, 'ScaledModel');

        if exist(GaitPath, 'dir') == 7
            
            participantID = subfolderName
            
            % Gait
            saveDir = 'F:\AlterG\Control\InverseKinematics\Gait';
            saveFileName = fullfile(saveDir, ['InverseKinematicsGait_' participantID '.mat']);
            if ~exist(saveFileName, 'file')
                cd(GaitPath);
                OpenSimConfig(GaitPath, SetupPath, IKResultsPath, ScaledModelPath);
                %save(saveFileName, 'InverseKinematics');
                cd(baseDir);
            end
            
        end
    end
    toc
    disp('Time for the batch processing')
end






function OpenScale(baseDir, subfolderName)
    % Inititiate libraries 
    disp('Nexus - Opensim Scale')
    vicon = ViconNexus();
    import org.opensim.modeling.*
    org.opensim.modeling.opensimCommon.GetVersion()

    %CD the Static/calibration folder
    participantDir = fullfile(baseDir, subfolderName);
    calibrationFilePath = '';
    calibrationFolder = '';
    folderNames = {'Static', 'Cal', 'static', 'Calibration'};
    for i = 1:length(folderNames)
        folderPath = fullfile(participantDir, folderNames{i});
        if exist(folderPath, 'dir')
            calibrationFolder = folderPath;
            cd(calibrationFolder);
            break;
        end
    end

    % Find the Static file. 
    [status, list] = system( 'dir /B /S *.x2d' );
    result = textscan( list, '%s', 'delimiter', '\n' );
    fileList = result{1};
    fileList = natsort(fileList)
    vicon.OpenTrial(fileList{1}(1:end-4), 30);

    [path, name] = vicon.GetTrialName();
    disp(['Scaling Trial == ' name]);
    %vicon.RunPipeline('Auto Label Static Trial', 'Private', 45)
    vicon.OpenTrial(fileList{1}(1:end-4), 30);

    % Pause and display instructions
    disp('Pause execution.');
    disp('Please follow these instructions with Nexus:');
    disp('- Go to the static folder, delete: .vsk, .mp, .trc, .c3d files');
    disp('- Reload the static trial');
    disp('- Add subject (ABIAlterg)');
    disp('- Run Pipeline with the following steps:');
    disp('   1. Reconstruct');
    disp('   2. Autolabel Static Frame');
    disp('   3. Scale subject (VSK)');
    disp('   4. Marker-only subject calibration (Skeleton)');
    disp('   5. Export TRC');
    disp('   6. Export C3D');
    disp('   7. Visually Check Labelling');
    disp('   8. Save the subject');
    pause;
    vicon.SaveTrial(20);

    %Configure Opensim for the scaling 
    disp('Configuring OpenSim for scaling...');
    
    % Define paths
    setupFile = 'F:\AlterG\Control\Data\SetupScale.xml';
    initialModel = 'F:\AlterG\Control\Gait2392_Simbody\gait2392_simbody.osim';
    resultsDir = fullfile(participantDir, 'ScaledModel');
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end

    % Read Participants data 
    demographicDataFile = 'F:\AlterG\Control\Data\Demographic Data.xlsx';
    opts = detectImportOptions(demographicDataFile);
    demographicData = readtable(demographicDataFile, opts);

    % Extract the subject ID from the subfolder name
    subjectID = str2double(subfolderName);

    % Find the row corresponding to the subject ID
    subjectRow = demographicData(demographicData.ID == subjectID, :);
    subjectMass = subjectRow.Mass;
    subjectHeight = subjectRow.Height / 100; % Convert height to meters
    subjectAge = subjectRow.Age;

    % Initialize the Scale Tool
    scaleTool = ScaleTool(setupFile);

    % **Set the subject parameters**
    scaleTool.setSubjectMass(subjectMass);
    scaleTool.setSubjectHeight(subjectHeight);
    scaleTool.setSubjectAge(subjectAge);

    % Set the model to be scaled
    modelMaker = scaleTool.getGenericModelMaker();
    modelMaker.setModelFileName(initialModel);

    % Set the marker set file
    markerSetFilePath = "F:\AlterG\Scripts\markersPO.xml";
    modelMaker.setMarkerSetFileName(markerSetFilePath);

    % Specify the output model file name
    outputModelFileName = fullfile(resultsDir, ['ScaledModel_' subfolderName '.osim']);
    modelScaler = scaleTool.getModelScaler();
    modelScaler.setOutputModelFileName(outputModelFileName);

    % Get the TRC file for the scaling
    trialForScale = dir(fullfile(calibrationFolder, '*.trc'));
    if isempty(trialForScale)
        error('No TRC files found in the calibration folder.');
    end
    trcFilePath = fullfile(calibrationFolder, trialForScale(1).name);
    modelScaler.setMarkerFileName(trcFilePath);


    xmlFileName = fullfile(resultsDir, ['ScaleSetup_' subfolderName '.xml']);
    scaleTool.print(xmlFileName);

    % Run the scaling tool
    %scaleTool.run();
    
end
    






function [] = OpenSimConfig(GaitPath, SetupPath, IKResultsPath, ScaledModelPath)
    
    import org.opensim.modeling.*
    % Validate OpenSim Configuration by disp version
    org.opensim.modeling.opensimCommon.GetVersion();
    
    % Load the model w dynamic naming 
    folderPath = GaitPath;
    setupFilePath = SetupPath;
    folder_parts = strsplit(folderPath, filesep);
    participant = folder_parts{end-2};
    time_instance = folder_parts{end-1};
    %dynamic_model_path = 'S:\Data\02\ScaledModel\ScaledModel_S2.osim';

    % Construct dynamic_model_path using the first .osim file found --
    files = dir(fullfile(ScaledModelPath, 'ScaledModel_*.osim'));
    if isempty(files)
        error('No .osim files found in the ScaledModelPath directory.');
    end
    dynamic_model_path = fullfile(ScaledModelPath, files(1).name);



    %if exist(dynamic_model_path, 'file') == 2
        %disp(['The file ' dynamic_model_path ' exists.']);
    %else
        %disp(['The file ' dynamic_model_path ' does not exist.']);
    %end


    %% UPDATE WHEN MODELS ARE HERE
    model = Model(dynamic_model_path)
    %model = Model('S:\Data\01\ScaledModel\ScaledModel_S1.osim');
    model.initSystem();
    
    % Define IK tool
%     if strcmp(participant, '11') && strcmp(time_instance, 'eight')
%         SetupFile = 'F:\Stryker\IkFiles\Setup_IK_Alternative.xml';
%     else
        SetupFile = 'F:\AlterG\Control\Data\Setup_IK.xml';   
%     end
    ikTool = InverseKinematicsTool(SetupFile);
    ikTool.setModel(model);
    % Get .trc files from folder
    % Get .trc files from the folder
    trc_data_folder = folderPath;
    trialsForIK = dir(fullfile(trc_data_folder, '*.trc'));
    nTrials = length(trialsForIK);

    % Specify where results will be printed.
    %results_folder = fullfile(folderPath, 'IKResults');
    results_folder = IKResultsPath
    nResults_folder = length(dir(fullfile(results_folder, '*.mot')))
    
    % Create the folder if it doesn't exist
    %if ~exist(results_folder, 'dir')
        %mkdir(results_folder);
    %end
    
    if nTrials == nResults_folder
        return
    end

    %Process the trials with IK
    for trial = 1:nTrials
        
        % Get the name of the file for this trial
        markerFile = trialsForIK(trial).name;
        
        % Create the name of the trial from the .trc file name
        name = regexprep(markerFile, '.trc', '');
        fullpath = fullfile(trc_data_folder, markerFile);
        
        % Get file parts
        [filepath, filename, ext] = fileparts(fullpath);
           
           % Get trc data to determine time range
           markerData = MarkerData(fullpath);
           initial_time = markerData.getStartFrameTime();
           final_time = markerData.getLastFrameTime();
        
        % Setup the ikTool for this trial
        ikTool.setName(name);
        ikTool.setMarkerDataFileName(fullpath);
        ikTool.setStartTime(initial_time);
        ikTool.setEndTime(final_time);
        
        % Output file
        outputFileName = fullfile(results_folder, [name '_ik.mot']);
        ikTool.setOutputMotionFileName(outputFileName);
        
        % Save the settings in a setup file
        % Extract directory path from SetupFile
        %[setupFilePath, ~, ~] = fileparts(SetupFile);
        disp(setupFilePath)

        % Save the settings in a setup file
        outfile = ['Setup_IK_' name '.xml'];
        ikTool.print(fullfile(setupFilePath, outfile));
 
        if exist(outputFileName, 'file')
            fprintf(['Result file for cycle # ' num2str(trial) ' already exists. Skipping.\n']);
            continue; % Skip the current iteration of the loop
        end

        fprintf(['Performing IK on cycle # ' num2str(trial) '\n']);
        
        % Run IK
        ikTool.run();
    end    
    
    %subsampled [Data, Descriptive, Headers, trialDurations]  = loadAndAnalyseMOT(results_folder);
    % Store the data after processing in F:\Stryker\Squat\Kinematics
    %saveResults(subsampledData, Descriptive, Headers, folderPath, participant, time_instance, trialDurations);

end

function [subsampledData, Descriptive, Headers, trialDurations] = loadAndAnalyseMOT(folderPath)
    % Load all MOT files in the folder
    motFiles = dir(fullfile(folderPath, '*.mot'));
    nTrials = length(motFiles);
    nVariables = 0; % Will be updated later
    
    % Initialize cell arrays to store all data and time
    allData = cell(1, nTrials);
    allTimeData = cell(1, nTrials);
    trialDurations = zeros(1, nTrials);;

    for trial = 1:nTrials
        % Load each MOT file
        motFilePath = fullfile(folderPath, motFiles(trial).name);
        dataStruct = importdata(motFilePath);
        
        % Extract all data and time
        allData{trial} = dataStruct.data(:, 2:end);
        allTimeData{trial} = dataStruct.data(:, 1);
        trialDurations(trial) = allTimeData{trial}(end) - allTimeData{trial}(1);
        % Update the number of variables
        if trial == 1
            nVariables = size(allData{trial}, 2);
        end
    end
    
    % Perform DTW on hip_flexion and align all trials to the first trial
    templateHipFlexion = allData{1}(:, find(strcmp(dataStruct.colheaders(2:end), 'hip_flexion_r')));
    
    % Initialize cell array to store aligned data
    alignedData = cell(1, nTrials);
    
    % Self-align the first trial to make it comparable to other trials
    [~, ix, iy] = dtw(templateHipFlexion, templateHipFlexion);
    alignedData{1} = allData{1}(iy, :);
    
    for trial = 2:nTrials
        [~, ~, iy] = dtw(templateHipFlexion, allData{trial}(:, find(strcmp(dataStruct.colheaders(2:end), 'hip_flexion_r'))));
        alignedData{trial} = allData{trial}(iy, :);
    end
    
    % Additional step: Align the first trial to another (e.g., the second)
    [~, ~, iy] = dtw(allData{2}(:, find(strcmp(dataStruct.colheaders(2:end), 'hip_flexion_r'))), alignedData{1}(:, find(strcmp(dataStruct.colheaders(2:end), 'hip_flexion_r'))));
    alignedData{1} = alignedData{1}(iy, :);
    
    % Find the maximum length across all aligned trials
    maxLength = max(cellfun(@(x) size(x, 1), alignedData));
    
    % Initialize a matrix to store padded data
    paddedData = NaN(maxLength, nVariables, nTrials);  % Using NaN initially to identify any non-updated entries
    
    for trial = 1:nTrials
        trialLength = size(alignedData{trial}, 1);
        
        % Copy aligned data into padded data
        paddedData(1:trialLength, :, trial) = alignedData{trial};
        
        % Pad the remaining entries with the average of the last three values
        if trialLength < maxLength
            avgTail = mean(alignedData{trial}(end-2:end, :), 1);  % Average of the last three values
            paddedData(trialLength+1:maxLength, :, trial) = repmat(avgTail, [maxLength - trialLength, 1]);
        end
    end

    % Initialize a matrix to store subsampled data
    subsampledData = NaN(100, nVariables, nTrials);
    
    % Define the new time vector for the subsampled data
    newTime = linspace(1, maxLength, 100);
    
    for trial = 1:nTrials
        % Current time vector for this trial
        currentLength = size(paddedData(:,:,trial), 1);
        oldTime = 1:currentLength;
        
        % Perform the interpolation
        for var = 1:nVariables
            subsampledData(:, var, trial) = interp1(oldTime, paddedData(:, var, trial), newTime, 'linear');
        end
    end

    % Define the variables of interest
    PlotVariables(subsampledData, dataStruct, nTrials, newTime)

    calcDescriptiveStats(subsampledData, dataStruct)


    % Initialize Descriptive structure to hold basic statistics for each trial
    Descriptive.meanData = zeros(size(subsampledData, 2), size(subsampledData, 3));
    Descriptive.stdData = zeros(size(subsampledData, 2), size(subsampledData, 3));
    Descriptive.rangeData = zeros(size(subsampledData, 2), size(subsampledData, 3));
    Descriptive.maxData = zeros(size(subsampledData, 2), size(subsampledData, 3));
    
    % Compute descriptive statistics per trial
    for trial = 1:size(subsampledData, 3)
        Descriptive.meanData(:, trial) = mean(subsampledData(:, :, trial), 1);
        Descriptive.stdData(:, trial) = std(subsampledData(:, :, trial), 0, 1);
        Descriptive.rangeData(:, trial) = range(subsampledData(:, :, trial), 1);
        Descriptive.maxData(:, trial) = max(subsampledData(:, :, trial), [], 1);
    end
    Headers = dataStruct.colheaders;
end

function PlotVariables(subsampledData, dataStruct, nTrials, newTime)

% Define the variables of interest
variables_single = {'pelvis_tilt', 'pelvis_list', 'pelvis_rotation'};
variables_right = {'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r'};
variables_left = {'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'};

% Initialize a single figure for subplots
figure;

% Subplot for pelvis variables
subplot(2, 3, 1);
hold on;
colors = {'g', 'm', 'y'}; % Different colors for pelvis variables
for i = 1:length(variables_single)
    var_name = variables_single{i};
    var_idx = find(strcmp(dataStruct.colheaders(2:end), var_name));
    meanData = mean(subsampledData(:, var_idx, :), 3);
    stdData = std(subsampledData(:, var_idx, :), 0, 3);
    confInterval = 1.96 * (stdData / sqrt(nTrials));
    plot(newTime, meanData, 'Color', colors{i}, 'LineWidth', 2);
    fill([newTime, fliplr(newTime)], ...
         [meanData' + confInterval', fliplr(meanData' - confInterval')], ...
         colors{i}, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
title('Pelvis Variables');
legend(variables_single);
hold off;

% Subplots for paired right-left variables
for i = 1:length(variables_right)
    subplot(2, 3, i + 1);  % Offset by 1 due to the pelvis subplot
    hold on;
    
    % Plot right-side variable
    var_name_r = variables_right{i};
    var_idx_r = find(strcmp(dataStruct.colheaders(2:end), var_name_r));
    meanData_r = mean(subsampledData(:, var_idx_r, :), 3);
    stdData_r = std(subsampledData(:, var_idx_r, :), 0, 3);
    confInterval_r = 1.96 * (stdData_r / sqrt(nTrials));
    plot(newTime, meanData_r, 'r', 'LineWidth', 2);
    fill([newTime, fliplr(newTime)], ...
         [meanData_r' + confInterval_r', fliplr(meanData_r' - confInterval_r')], ...
         'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Plot left-side variable
    var_name_l = variables_left{i};
    var_idx_l = find(strcmp(dataStruct.colheaders(2:end), var_name_l));
    meanData_l = mean(subsampledData(:, var_idx_l, :), 3);
    stdData_l = std(subsampledData(:, var_idx_l, :), 0, 3);
    confInterval_l = 1.96 * (stdData_l / sqrt(nTrials));
    plot(newTime, meanData_l, 'b', 'LineWidth', 2);
    fill([newTime, fliplr(newTime)], ...
         [meanData_l' + confInterval_l', fliplr(meanData_l' - confInterval_l')], ...
         'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    title([var_name_r(1:end-2), ' (R/L)']);  % Remove the '_r' and add '(R/L)'
    hold off;
end

end

function calcDescriptiveStats(subsampledData, dataStruct)
    % Define the hip and knee variables of interest
    hip_knee_vars_r = {'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r'};
    hip_knee_vars_l = {'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l'};
    
    % Initialize figure for range
    figure_range = figure;
    sgtitle('Box Plot of Range for Hip and Knee Variables');
    
    % Initialize figure for maximum values
    figure_max = figure;
    sgtitle('Box Plot of Max Values for Hip and Knee Variables');
    
    % Loop through each variable pair to plot
    for i = 1:length(hip_knee_vars_r)
        var_name_r = hip_knee_vars_r{i};
        var_name_l = hip_knee_vars_l{i};
        
        var_idx_r = find(strcmp(dataStruct.colheaders(2:end), var_name_r));
        var_idx_l = find(strcmp(dataStruct.colheaders(2:end), var_name_l));
        
        % Extract the data for these variables
        var_data_r = squeeze(subsampledData(:, var_idx_r, :));
        var_data_l = squeeze(subsampledData(:, var_idx_l, :));
        
        % Calculate range and maximum values for both left and right
        range_r = range(var_data_r, 1);
        range_l = range(var_data_l, 1);
        max_r = max(var_data_r, [], 1);
        max_l = max(var_data_l, [], 1);
        
          % Plot the range
        figure(figure_range);
        subplot(2, 2, i);
        boxplot([range_r', range_l'], 'Labels', {var_name_r, var_name_l});
        title(['Range of ', var_name_r(1:end-2)]);
        h = findobj(gca, 'Tag', 'Box');
        for j = 1:length(h)
            boxPos = get(h(j), 'YData');
            xData = get(h(j), 'XData');
            if j == 1 % Left side variable
                patch(xData, boxPos, 'blue', 'FaceAlpha', 0.3);
            else % Right side variable
                patch(xData, boxPos, 'red', 'FaceAlpha', 0.3);
            end
        end

        % Plot the max values
        figure(figure_max);
        subplot(2, 2, i);
        boxplot([max_r', max_l'], 'Labels', {var_name_r, var_name_l});
        title(['Max of ', var_name_r(1:end-2)]);
        h = findobj(gca, 'Tag', 'Box');
        for j = 1:length(h)
            boxPos = get(h(j), 'YData');
            xData = get(h(j), 'XData');
            if j == 1 % Left side variable
                patch(xData, boxPos, 'blue', 'FaceAlpha', 0.3);
            else % Right side variable
                patch(xData, boxPos, 'red', 'FaceAlpha', 0.3);
            end
        end
    end
end

function saveResults(subsampledData, Descriptive, Headers, folderPath, participant, time_instance, trialDurations)
    % Create a structure to hold the data
    Results.subsampledData = subsampledData;
    Results.Descriptive = Descriptive;
    Results.trialDurations = trialDurations;
    Results.Headers = Headers;
    
    % Parse folderPath to extract the trial nature
    folderComponents = strsplit(folderPath, filesep);  % Split the folderPath by the file separator
    trialNature = folderComponents{end};  % Get the last component, which indicates the trial nature (STS, Squat, Gait, etc.)
    
    % Dynamically define the save directory based on the trial nature
    baseSaveDir = 'F:\Stryker';
    saveDir = fullfile(baseSaveDir, trialNature, 'Kinematics');
    
    % Check if directory exists; if not, create it
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    
    % Define the filename
    saveFile = fullfile(saveDir, sprintf('%sKin_%s_%s.mat', trialNature, participant, time_instance));
    
    % Save the structure to a .mat file
    save(saveFile, 'Results');
end



