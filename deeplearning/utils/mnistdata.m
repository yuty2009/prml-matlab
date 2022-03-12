classdef mnistdata < handle
    
    properties(Constant, GetAccess='private')
        SOURCE_URL   = 'http://yann.lecun.com/exdb/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte'
        TRAIN_LABELS = 'train-labels-idx1-ubyte'
        TEST_IMAGES  = 't10k-images-idx3-ubyte'
        TEST_LABELS  = 't10k-labels-idx1-ubyte'
    end
    
    methods
        
        % load(path = ".", onehot=false, dtype='float', 
        % need_reshape=true, validation_size=5000)
        function [train, validation, test] = load(obj, varargin)
            
            datapath = '.';
            onehot = false;
            need_reshape= true;
            dtype = 'float';
            validation_size=0;
            
            if (nargin >= 2) datapath = varargin{1}; end
            if (nargin >= 3) onehot = varargin{2}; end
            if (nargin >= 4) need_reshape = varargin{3}; end
            if (nargin >= 5) dtype = varargin{4}; end
            if (nargin >= 6) validation_size = varargin{5}; end
            
            [train_images, train_labels] = obj.read_dataset('train', datapath);
            [test_images, test_labels]  = obj.read_dataset('test', datapath);
            
            if (onehot)
                train_labels = obj.onehot_labels(train_labels, 10);
                test_labels = obj.onehot_labels(test_labels, 10);
            end
            
            if ~(validation_size >= 0  && validation_size <= size(train_images,1))
                error(['Validation size should be between 0 and ' num2str(size(train_images,1)) ...
                    'Received: ' num2str(validation_size)]);
            end
            
            imageshape = size(train_images);
            validation_images = reshape(train_images(1:validation_size,:), [validation_size, imageshape(2:end)]);
            validation_labels = train_labels(1:validation_size,:);
            train_images = reshape(train_images(validation_size+1:end,:), [imageshape(1)-validation_size, imageshape(2:end)]);
            train_labels = train_labels(validation_size+1:end,:);
            
            train = imageset(train_images, train_labels, need_reshape, dtype);
            validation = imageset(validation_images, validation_labels, need_reshape, dtype);
            test = imageset(test_images, test_labels, need_reshape, dtype);
            
        end
        
        function [images, labels] = read_dataset(obj, dataname, datapath)
            if strcmpi(dataname, 'train')
                fname_img = [datapath '/' obj.TRAIN_IMAGES];
                images = obj.read_images(fname_img);
                fname_lbl = [datapath '/' obj.TRAIN_LABELS];
                labels = obj.read_labels(fname_lbl);
            elseif strcmpi(dataname, 'test')
                fname_img = [datapath '/' obj.TEST_IMAGES];
                images = obj.read_images(fname_img);
                fname_lbl = [datapath '/' obj.TEST_LABELS];
                labels = obj.read_labels(fname_lbl);
            else
                error('dataset must be `test` or `train`');
            end
            fprintf('Read %d %s images from [%s]\n', length(labels), dataname, datapath);
        end
        
        % returns a 28x28x[number of MNIST images] matrix containing
        % the raw MNIST images
        function images = read_images(~, filename)
            fp = fopen(filename, 'rb');
            assert(fp ~= -1, ['Could not open ', filename, '']);
            
            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2051, ['Bad magic number in ', filename, '']);
            
            numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
            numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
            numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
            
            images = fread(fp, inf, 'unsigned char');
            images = reshape(images, numCols, numRows, numImages);
            images = permute(images,[3 2 1]);
            
            fclose(fp);
            
            % Convert to double and rescale to [0,1]
            images = double(images) / 255;   
        end
        
        % returns a [number of MNIST images]x1 matrix containing
        % the labels for the MNIST images
        function labels = read_labels(~, filename)
            fp = fopen(filename, 'rb');
            assert(fp ~= -1, ['Could not open ', filename, '']);
            
            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2049, ['Bad magic number in ', filename, '']);
            
            numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
            
            labels = fread(fp, inf, 'unsigned char');
            
            assert(size(labels,1) == numLabels, 'Mismatch in label count');
            
            fclose(fp);
            
            labels(labels == 0) = 10; % Remap 0 to 10
        end
        
        % Convert class labels from scalars to one-hot vectors
        function labels_onehot = onehot_labels(~, labels, num_classes)
            labels_onehot = full(sparse(1:length(labels),labels,1));
            % num_labels = size(labels,1);
            % labels_onehot = zeros(num_labels, num_classes);
            % for i = 1:num_labels
            %     labels_onehot(i, labels(i)) = 1;
            % end
        end
        
    end
    
end