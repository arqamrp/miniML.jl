mutable struct Tensor #scalar
    shape::Tuple #stores the shape of the Tensor
    dtype::Type
    
    data::Union{Number, Array{<:Number}} # stores the tensor
    grad::Union{Number, Array{<:Number}} # stores gradients
    
    _flag::Bool # denotes whether it should store a gradient
    _train::Bool # whether the tensor can be modified
    
    _prev::Tuple # stores connected nodes info
    _op::String # stores edge info (operation)

    
    function Tensor(shape::Tuple, dtype::Type, data::Union{Number, Array{<:Number}}, grad::Union{Number, Array{<:Number}}, _flag::Bool, _train::Bool, _prev::Tuple, _op::String)
        new(shape, dtype, data, grad, _flag, _train, _prev, _op)
    end
    
end


# Leaf constructor for arrays
function Tensor(data::Array{<:Number}, _hasGrad::Bool= true) # custom constructor for leaf node
    Tensor(size(data), eltype(data), data, zeros(eltype(data), size(data)) , _hasGrad, false, (), "")
end

# Opres constructor for arrays
function Tensor(data::Array{<:Number}, _children::Tuple, _op::String) # custom constructor for non leaf node
    Tensor(size(data), eltype(data), data, zeros(eltype(data), size(data)), true, false, _children, _op)
end 



# Leaf constructor for numbers
function Tensor(data::Number, _hasGrad::Bool = true) # custom constructor for leaf node
    Tensor( () , typeof(data), data, zero(typeof(data)), _hasGrad, false, (), "")
end

# Opres constructor for numbers
function Tensor(data::Number, _children::Tuple, _op::String) # custom constructor for non leaf node
    Tensor( () ,  typeof(data), data, zero(typeof(data)) , true, false, _children, _op)
end
