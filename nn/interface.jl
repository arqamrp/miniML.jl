

mutable struct tensor_function # for general scalar function definitions
    _func::Function # should be a function of params and input
    params::Vector{Tensor} # should be a array of tensors

    function tensor_function(func::Function, input_params::Vector)
        init_params = [Tensor(float.(param), true) for param in input_params]
        new(func, init_params)
    end
end


function forward(model::tensor_func, input::Vector)
    if !isTensor(input[1])
        input = [Tensor(float.(inp), false) for inp in input]
    end            
    return model._func(model.params, input)
end


function Backward!(loss::Tensor, graph_arr::Array{Tensor})
    loss.grad = 1.
    
    function back(node::Tensor)
        if node._flag & !isempty(node._prev) # if not leaf node or scalar
            backward!(node)
            back(node._prev[1])
            back(node._prev[2])
        end
    end
    
    back(loss)
        
end


function zerograd!(model::tensor_function)
    for param in model.params
        param.grad = zeros(param.dtype ,param.shape)
    end
end

function train!(model::tensor_function)
    for param in model.params
        param._train = true 
    end
end