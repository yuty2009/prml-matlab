function varargout = mlpparam(varargin)

mlp = varargin{1};
type = varargin{2};

switch(type)
    case 1 % vectorize
        W = varargin{3};
        b = varargin{4};
        theta = [];
        for L = 1:mlp.NL-1
            theta = [theta; W{L}(:); b{L}(:)];
        end
        varargout{1} = theta;
    case 2 % stack
        theta = varargin{3};
        W = cell(mlp.NL-1,1); % weight matrices (Sj by Si)
        b = cell(mlp.NL-1,1); % biases (1 by Si)
        start = 0;
        for L = 1:mlp.NL-1
            Sj = mlp.SN(L);
            Si = mlp.SN(L+1);
            W{L} = reshape(theta(start+(1:Sj*Si)),Sj,Si);
            start = start + Sj*Si;
            b{L} = reshape(theta(start+(1:Si)),1,Si);
            start = start + Si;
        end
        varargout{1} = W;
        varargout{2} = b;
end