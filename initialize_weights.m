function [ stack, W_t ] = initialize_weights( eI )
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes an RNN via the fields layerSizes, inputDim and
%   temporalLayer
%   
%   This uses Xavier's weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training 
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers
stack = cell(1, numel(eI.layerSizes));
for l = 1 : numel(eI.layerSizes)
    if l > 1
        prevSize = eI.layerSizes(l-1);
    else
        prevSize = eI.inputDim;
    end;
    curSize = eI.layerSizes(l);
    % Xaxier's scaling factor
    s = sqrt(6) / sqrt(prevSize + curSize);
    % Ilya suggests smaller scaling for recurrent layer
    if l == eI.temporalLayer
        s = sqrt(6) / sqrt(prevSize + 2*curSize);
    end;
    stack{l}.W = rand(curSize, prevSize)*2*s - s;
    stack{l}.b = zeros(curSize, 1);
end
%% weight tying
% default weight tying to false
if ~isfield(eI, 'tieWeights')
    eI.tieWeights = 0;
end;
% overwrite decoder layers for tied weights
if eI.tieWeights
    decList = [(numel(eI.layerSizes)/2)+1 : numel(eI.layerSizes)-1];
    for l = 1:numel(decList)
        lDec = decList(l);        
        lEnc = decList(1) - l;
        assert( norm(size(stack{lEnc}.W') - size(stack{lDec}.W)) == 0, ...
            'Layersizes dont match for tied weights');
        stack{lDec}.W = stack{lEnc}.W';
    end;
end;
%% initialize temporal weights if they should exist
W_t = [];
if eI.temporalLayer 
    % assuems temporal init type set
    if strcmpi(eI.temporalInit, 'zero')
        W_t = zeros(eI.layerSizes(eI.temporalLayer));
    elseif strcmpi(eI.temporalInit, 'rand')
        % Ilya's modification to Xavier's update rule
        s = sqrt(6) / sqrt(3*eI.layerSizes(eI.temporalLayer));
        W_t = rand(eI.layerSizes(eI.temporalLayer))*2*s - s;
    elseif strcmpi(eI.temporalInit, 'eye')
        W_t = eye(eI.layerSizes(eI.temporalLayer));
    else
        error('unrecognized temporal initialization: %s', eI.temporalInit);
    end;    
end;
%% init short circuit connections
% default short circuits to false
if ~isfield(eI, 'shortCircuit')
    eI.shortCircuit = 0;
end;
if eI.shortCircuit
    %padSize = (eI.winSize-1) / 2;
    %stack{end}.W_ss = [zeros(eI.featDim, padSize*eI.featDim), eye(eI.featDim),...
    %    zeros(eI.featDim, padSize*eI.featDim)];
    % use random init since input might contain noise estimate
    s = sqrt(6) / sqrt(eI.inputDim + eI.layerSizes(end));
    stack{end}.W_ss = rand(eI.inputDim, eI.layerSizes(end))*2*s - s;
end;
     


