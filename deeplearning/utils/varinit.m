function var = varinit(shape, type, opt)

global useGPU;

switch(type)
    case 'constant'
        if ~isfield(opt, 'value')
            opt.value = 0;
        end
        if useGPU
            var = opt.value*gpuArray.ones(shape);
        else
            var = opt.value*ones(shape);
        end
    case 'normal'
        if ~isfield(opt, 'mean')
            opt.mean = 0;
        end
        if ~isfield(opt, 'std')
            opt.std = 1;
        end
        if useGPU
            var = opt.mean + opt.std*gpuArray.randn(shape);
        else
            var = opt.mean + opt.std*randn(shape);
        end
    case 'normal_truncated'
        if ~isfield(opt, 'mean')
            opt.mean = 0;
        end
        if ~isfield(opt, 'std')
            opt.std = 1;
        end
        if ~isfield(opt, 'low')
            opt.low = opt.mean - opt.std;
        end
        if ~isfield(opt, 'high')
            opt.high = opt.mean + opt.std;
        end
        var = randn_truncated(shape, opt.mean, opt.std, opt.low, opt.high);
    case 'uniform'
        if ~isfield(opt, 'a')
            opt.a = 0;
        end
        if ~isfield(opt, 'b')
            opt.b = 1;
        end
        if useGPU
            var = opt.a + (opt.b-opt.a)*gpuArray.rand(shape);
        else
            var = opt.a + (opt.b-opt.a)*rand(shape);
        end
    case 'inout'
        if useGPU
            var = (gpuArray.rand(shape) - 0.5) * 2 * sqrt(6 / (opt.in + opt.out));
        else
            var = (rand(shape) - 0.5) * 2 * sqrt(6 / (opt.in + opt.out));
        end
end