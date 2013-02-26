function  varargout = load_aurora(featBase, trainingSet, subdir, M, eI, mode)
% featBase: Location of the features (ex: aurora2/features/)
% trainingSet: 'Mfc08_multiTR', 'Mfc08_setaTR', etc
% subdir: cell array ofsubdirectory of the features (ex: {N1_SNR5/})
% M: Number of sample files to use. value < 0 loads all
% eI.winSize: Size of window
% eI.seqLen: unique lengths (in ascending order) 
%             files are chopped by these lengths. (ex: [1, 10, 100])
% eI.useCache: If true, read + save to aurora_data.mat
%              !!!WARNING!!!: If you change this function, delete
%              aurora_data.mat.
% eI.targetWhiten: Specify the path to the whitening data.
% data_ag: noisy data. cell array of different training lengths. 
% target_ag: clean data. cell array of different training lengths.
% AMAAS edit. Applying CMN to input and output so whitening file not used

% mode:
%     0: Training (noisy data and clean data)
%     1: Testing (just noisy data)
%     2: Error testing (noisy data and clean data, both loaded without chunking)

if nargin <= 5,
	mode = 0;
end

%% Handle caching. Dont cache for testing. 
% AMAAS caching makes me nervous
% if ~mode && isfield(eI, 'useCache') && eI.useCache
%     try
%         load aurora_data.mat;
%         if M == cache_M && ...
%               isequal(eI, cache_eI) && ...
%               isequal(featBase, cache_featBase) && ...
%               isequal(subdir, cache_subdir),
%             data_ag = cache_data_ag;
%             target_ag = cache_target_ag;
%         end
%     catch
%     end
%     if exist('data_ag', 'var') && exist('target_ag', 'var'),
%         return;
%     end
% end;

%% Testing code
input_fnames = {};
unique_lengths = [];

%% Get all file names
multi = {};
if length(subdir) > 0
  for i=1:length(subdir),
    multi = [multi; getAllFiles([featBase trainingSet '/' subdir{i}])];
  end
else
    multi = getAllFiles([featBase trainingSet]);
end;

%% Get whitening data.
% whitening not being used at all
% try
%     load(eI.targetWhiten);
% catch
%     % AMAAS this is dangerous as it could result in people using different
%     % whitening matrices. Better to compute them once and all share
%     % make_aurora_whitening;
%     %load(eI.targetWhiten);
%     disp('whitening matrix not found');
% end

%% Loop through every file.
if M < 0
    M = length(multi);
end

%% Set up. During testing, dont know the lengths so cant pre-allocate
if mode,
    data_ag = {};
    target_ag = {};  % returns empty targets
else,
    seqLenSizes = zeros(1,length(eI.seqLen));
  for i=1:M
    [pathstr, name, ext] = fileparts(multi{i});
    [multi_data, htkCode] = htkread(multi{i});
    [T, nfeat] = size(multi_data);

    remainder = T;  
    for i=length(eI.seqLen):-1:1
      num = floor(remainder/eI.seqLen(i));
      remainder = mod(remainder,eI.seqLen(i));
      seqLenSizes(i) = seqLenSizes(i)+num;
    end
  end
  data_ag = cell(1,length(eI.seqLen));
  target_ag = cell(1,length(eI.seqLen));
  for i=length(eI.seqLen):-1:1
    data_ag{i} = zeros(eI.inputDim*eI.seqLen(i),seqLenSizes(i));
    target_ag{i} = zeros(nfeat*eI.seqLen(i),seqLenSizes(i));
  end
end


seqLenPositions = ones(1,length(eI.seqLen));



for i=1:M,
    %% Load multi data.
    [pathstr, name, ext] = fileparts(multi{i});
    [multi_data, htkCode] = htkread(multi{i});
    multi_data = multi_data'; 
    [nFeat,T] = size(multi_data);
    assert(nFeat == 14);
    
    %% apply CMVN to input
    cur_mean = mean(multi_data, 2);
    cur_std = std(multi_data, 0, 2);
    multi_data = bsxfun(@minus, multi_data, cur_mean);
    multi_data = bsxfun(@rdivide, multi_data, cur_std);
    %% estimate noise of the input signal
    noise_ind = 1:min(10,size(multi_data,2));
    noise_est = mean(multi_data(:,noise_ind),2);
    %% GLobal X/Y normalization
    % multi_data = bsxfun(@minus, multi_data, w_mean);
    % multi_data = bsxfun(@rdivide, multi_data, w_std);
    
    
    %% Load clean data if not testing
    if mode ~= 1
      [clean_data dummy] = htkread([featBase 'Mfc08_cleanTR/' name ext]);
      clean_data = clean_data';
      assert(length(multi_data) == length(clean_data));
      %% apply CMVN to targets
      cur_mean = mean(clean_data, 2);
      cur_std = std(clean_data, 0, 2);
      clean_data = bsxfun(@minus, clean_data, cur_mean);
      clean_data = bsxfun(@rdivide, clean_data, cur_std);
      %% GLobal X/Y normalization
      %       clean_data = bsxfun(@minus, clean_data, w_mean);
      %       clean_data = bsxfun(@rdivide, clean_data, w_std);
    end
    
    %% zero pad
    if eI.winSize > 1
        % winSize must be odd for padding to work
        if mod(eI.winSize,2) ~= 1
            fprintf(1,'error! winSize must be odd!');
            return
        end;
        % pad with repeated frames on both sides so im2col data
        % aligns with output data
        nP = (eI.winSize-1)/2;
        multi_data = [repmat(multi_data(:,1),1,nP), multi_data, ...
            repmat(multi_data(:,end),1,nP)];
    end;
    %% im2col puts winSize frames in each column
    multi_data_slid = im2col(multi_data,[nFeat, eI.winSize],'sliding');
    % concatenate noise estimate to each input
    multi_data_slid = [multi_data_slid; repmat(noise_est,1,T)];
     if mode == 1, % Testing
        c = find(unique_lengths == T);
        if isempty(c)
            % add new unique length if necessary
            data_ag = [data_ag, multi_data_slid(:)];
            unique_lengths = [unique_lengths, T];
            input_fnames = [input_fnames; {multi{i}}];
        else
            data_ag{c} = [data_ag{c}, multi_data_slid(:)];
            input_fnames{c} = [input_fnames{c}; {multi{i}}];
        end;
    elseif mode == 2, % Error analysis. 
		c = find(unique_lengths == T);
        if isempty(c)
            % add new unique length if necessary
            data_ag = [data_ag, multi_data_slid(:)];
            target_ag = [target_ag, clean_data(:)];
            unique_lengths = [unique_lengths, T];
        else
            data_ag{c} = [data_ag{c}, multi_data_slid(:)];
            target_ag{c} = [target_ag{c}, clean_data(:) ];
        end;
	else,
		%% put it in the correct cell area.
		while T > 0
			% assumes length in ascending order.
			% Finds longest length shorter than utterance
			c = find(eI.seqLen <= T, 1,'last');
			
			binLen = eI.seqLen(c);
			assert(~isempty(c),'could not find length bin for %d',T);
			% copy data for this chunk
			data_ag{c}(:,seqLenPositions(c))=reshape(multi_data_slid(:,1:binLen),[],1);
                        target_ag{c}(:,seqLenPositions(c))=reshape(clean_data(:,1:binLen),[],1);
                        seqLenPositions(c) = seqLenPositions(c)+1;
			% trim for next iteration
			T = T-binLen;
			if T > 0
				multi_data_slid = multi_data_slid(:,(binLen+1):end);
				clean_data = clean_data(:,(binLen+1):end);
			end;
		end;
	end;
end;

% if isfield(eI, 'useCache') && eI.useCache
%     cache_data_ag = data_ag;
%     cache_target_ag = target_ag;
%     cache_featBase = featBase;
%     cache_subdir = subdir;
%     cache_M = M;
%     cache_eI = eI;
%     save aurora_data.mat cache_data_ag cache_target_ag cache_featBase cache_subdir cache_M cache_eI;
% end

theoutputs = {data_ag, target_ag, input_fnames};
varargout = theoutputs(1:nargout);

return;
%% Unit test 1
eI.seqLen = [1 50 100];
eI.winSize = 3;
baseDir = 'C:\Users\Tyler-Sager\Documents\MATLAB\DrDraeLoader\features_backup\';
[data_ag, target_ag] = load_aurora(baseDir, 'Mfc08_multiTR', {'N1_SNR5'}, -1, eI); % Notice ascending order

% see if the next frame's window has the current frame in it.
data_sliding = reshape(data_ag{3}, eI.winSize*14, 100*size(data_ag{3},2));
imagesc(data_sliding(15:28,1:end-1) - data_sliding(1:14,2:end));

%% Unit test 2: Requires manually looking at output.
eI.seqLen = [1 50 100];
eI.winSize = 3;
eI.useCache = 1;
baseDir = '/afs/cs.stanford.edu/u/amaas/scratch/aurora2/features/';
[data_ag, target_ag] = load_aurora(baseDir, 'Mfc08_multiTR', {'N1_SNR10'}, 10, eI);
fprintf('Bust the cache -- different subdir. Expecting output.');
[data_ag, target_ag] = load_aurora(baseDir, 'Mfc08_multiTR', {'N1_SNR5'}, 10, eI);
fprintf('Bust the cache -- different M. Expecting output.');
[data_ag, target_ag] = load_aurora(baseDir, 'Mfc08_multiTR', {'N1_SNR5'}, 11, eI);
fprintf('Cache hit. NO OUTPUT BELOW THIS LINE!!!!\n');
[data_ag, target_ag] = load_aurora(baseDir, {'N1_SNR5'}, 11, eI);


%% Unit test 3: Testing data
eI.seqLen = [1 50 100];
eI.winSize = 3;
eI.useCache = 1;
baseDir = '/afs/cs.stanford.edu/u/amaas/scratch/aurora2/features/';
[data_ag, target_ag, fnames] = load_aurora(baseDir, 'Mfc08_multiTR', {'N1_SNR10'}, 10, eI, 1);

end

