function write_rec_dataset(theta, eI, dirList)

% Test dirs are Mfc08TS_seta,Mfc08TS_setc,Mfc08TS_setb

%% setup
if ~isfield(eI, 'subdirs'),
    eI.subdirs = {};
end

htkCode = 8262;
%% for each input directory
for d = dirList
    d = d{1};
    %% load data and forward prop
    [data_cell, targets_cell, in_fnames]=load_aurora(eI.featInBase, d, eI.subdirs, -1, eI, 1);
    [ cost, grad, numTotal, output ] = drdae_obj( theta, eI, data_cell, targets_cell, 1, 1);

    % create output subdir
    mkdir([eI.featOutBase d]);
    %% write out each file. loop over unique lens
    for l = 1:length(output)
        %% loop over all files for this len
        for i=1:size(output{l},2),        
            %% pull filename from cell array. special case when only 1 file
            if size(output{l},2) == 1
                fnameIn = in_fnames{l};
            else
                fnameIn = in_fnames{l}{i};
            end;
            result = reshape(output{l}(:, i), eI.featDim, []);
            fnameOut = strrep(fnameIn, eI.featInBase, eI.featOutBase);
            [pathOut, ~, ~] = fileparts(fnameOut);
            if ~exist(pathOut,'dir')
                mkdir(pathOut);
            end;
            htkwrite(result',fnameOut, htkCode);
        end;
    end;
end;

return
%% Unit test 1: Normal usage

load('model/model_0.mat');
eI.featInBase = '/path/to/aurora2/features/';
eI.featOutBase = '/path/to/aurora2/features_rec_new/';
eI.subdirs = {};
dirList = {'Mfc08_multiTR', 'Mfc08TS_seta', 'Mfc08TS_setb','Mfc08TS_setc'};

write_rec_dataset(theta, eI, dirList);

end
