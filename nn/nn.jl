function linear(params::Vector{Tensor}, input::Vector{Tensor})

    W = params[1]
    b = params[2]
    
    @assert W.shape[2] == input[1].shape[1]
    @assert W.shape[1] == b.shape[1]
    
    ans = W * input[1] + b

    return ans

end



function Linear(n_in::Int64, n_out::Int64)
    W = randn(n_out, n_in)
    b = randn(n_out,1)
    params = (W,b)
    return diff_func(linear, params)
end


mutable struct nn
    structure::Array{Float64}
    layers::Array{Array{Float64}}
    activations::Array{Function}
    
    function nn(structure::Array{64})
        layers = Array()
        new(structure, )
    end
    
end