classdef imageset < handle
    
    properties(GetAccess='public', SetAccess='private')
        images;
        labels;
        inshape;
        outshape;
        num_examples;
        epochs_completed;
        index_in_epoch;
    end
    
    methods
        
        % constructor
        % obj = imageset(images, labels, dtype, need_reshape)
        function obj = imageset(varargin)
            
            dtype = 'float';
            need_reshape = true;
            
            if (nargin < 2) 
                error('At least 2 input arguments are required'); 
            else
                images = varargin{1};
                labels = varargin{2};
            end
            if (nargin >= 3) need_reshape = varargin{3}; end
            if (nargin >= 4) dtype = varargin{4}; end
            
            
            if ~strcmpi(dtype, 'int') && ~strcmpi(dtype, 'float')
                error(['Invalid image dtype ' dtype ', expected int or float']);
            end
            
            assert(size(images, 1) == size(labels, 1), ...
                ['images.shape: ' num2str(size(images)) ', labels.shape: ' num2str(size(labels))]);
        	
        	obj.inshape = size(images);
            obj.outshape = obj.inshape;
            obj.num_examples = obj.inshape(1);
    
            % Convert shape from [num examples, rows, columns, depth]
            % to [num examples, rows*columns] (assuming depth == 1)
            if (need_reshape)
                obj.outshape = [obj.inshape(1), prod(obj.inshape(2:end))];
                images = reshape(images, obj.outshape);
            end
            if (strcmpi(dtype, 'float') && ~isfloat(images))
                % Convert from [0, 255] -> [0.0, 1.0].
                images = double(images) .* (1.0 / 255.0);
                labels = double(labels);
            end
            
            obj.images = images;
            obj.labels = labels;
            obj.epochs_completed = 0;
            obj.index_in_epoch = 0;
        end
        
        % Return the next `batchsize` examples from this data set
        function [batch_x, batch_y] = nextbatch(obj, batchsize, shuffle)
        
            start = obj.index_in_epoch;
            % Shuffle for the first epoch
            if (obj.epochs_completed == 0 && start == 0 && shuffle)
                perm0 = randperm(obj.num_examples);
                obj.images = reshape(obj.images(perm0,:), obj.outshape);
                obj.labels = obj.labels(perm0,:);
            end
            % Go to the next epoch
            if (start + batchsize > obj.num_examples)
                % Finished epoch
                obj.epochs_completed = obj.epochs_completed + 1;
                % Get the rest examples in this epoch
                rest_num_examples = obj.num_examples - start;
                images_rest_part = obj.images(start+1:obj.num_examples,:);
                labels_rest_part = obj.labels(start+1:obj.num_examples,:);
                % Shuffle the data
                if (shuffle)
                    perm = randperm(obj.num_examples);
                    obj.images = reshape(obj.images(perm,:), obj.outshape);
                    obj.labels = obj.labels(perm,:);
                end
                % Start next epoch
                start = 0;
                obj.index_in_epoch = batchsize - rest_num_examples;
                stop = obj.index_in_epoch;
                images_new_part = obj.images(start+1:stop,:);
                labels_new_part = obj.labels(start+1:stop,:);
                batch_x = cat(1, images_rest_part, images_new_part);
                batch_x = reshape(batch_x, [batchsize, obj.outshape(2:end)]);
                batch_y = cat(1, labels_rest_part, labels_new_part);
            else
                obj.index_in_epoch = obj.index_in_epoch + batchsize;
                stop = obj.index_in_epoch;
                batch_x = obj.images(start+1:stop,:);
                batch_x = reshape(batch_x, [batchsize, obj.outshape(2:end)]);
                batch_y = obj.labels(start+1:stop,:);
            end
            
        end
        
    end
    
end