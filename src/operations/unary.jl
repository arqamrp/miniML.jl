# UTILITY FUNCTIONS

# Check if the given object is a valid tensor
function isTensor(arg)
    return isa(arg , Tensor)
end

# Returns shape of tensor
function Base.size(arg::Tensor)
    return arg.shape
end

# Check equality of two tensors
function Base.isequal(arg1::Tensor, arg2::Tensor)
    return (arg1.data == arg2.data)
end


# UNARY OPERATIONS

# e^tensor
function Base.exp(arg::Tensor)
    val = exp.(arg.data)
    _op = "exp"
    return Tensor(val, (arg, ), _op)
end

# log(tensor)
function Base.log(arg::Tensor)
    val = log.(arg.data) 
    _op = "log"
    return Tensor(val, (arg, ), _op)
end

# transpose(matrix tensor)
function Base.adjoint(arg::Tensor)
    @assert length(arg.shape)==2
    val = Array(arg.data')
    _op = "T"
    return Tensor(val, (arg, ), _op)
end


# SCALAR TENSOR OPERATIONS

# Exponentiation (tensor^k)
import Base:^
function ^(arg1::Tensor, arg2::Number)
    ans = arg1.data.^arg2
    return Tensor( ans, (arg1, arg2), "^" )
end

# Scalar mult (k * tensor)
import Base:*
function *(arg1::Tensor, arg2::Number)
    val = arg1.data * arg2
    _op = "scalar*"
    return Tensor(val, (arg1, arg2), _op)
end

function *(arg1::Number, arg2::Tensor)
    return arg2*arg1
end

# Scalar div (tensor/k)
import Base:/
function /(arg1::Tensor, arg2::Number)
    arg2 = 1/arg2
    return arg1 * arg2
end

# scalar division (k/tensor)
function /(arg1::Number, arg2::Tensor)
    val = arg1 ./ arg2.data
    _op = "/"
    return Tensor( val, ( arg2, arg1), _op)
end

# Scalar addition (tensor + k)
import Base:+
function +(arg1::Tensor, arg2::Number)
    val = arg1.data .+ arg2
    _op = "scalar+"
    return Tensor(val, (arg1, arg2), _op)
end
function +(arg1::Number, arg2::Tensor)
    return +(arg2, arg1)
end

# scalar subtraction (tensor - k)
import Base:-
function -(arg1::Tensor, arg2::Number)
    arg2 = -arg2
    return arg1 + arg2
end

function -(arg1::Number, arg2::Tensor)
    return arg1 + (-arg2)
end

# Negation
import Base:-
function -(arg::Tensor)
    return -1 * arg
end

# Comparison
import Base.isless
function isless(arg1::Tensor, arg2::Number)
    val = (arg1.data .< arg2) * 1
    _op = "<"
    return Tensor(val, (arg1, arg2), _op)
end
function isless(arg1::Number, arg2::Tensor)
    val = (arg2.data .> arg1) * 1
    _op = ">"
    return Tensor(val, (arg2, arg1), _op)
end

# ACCESS OPERATIONS

# Indexing (tensor[idx])
function Base.getindex(arg::Tensor, idx...)
    val = arg.data[idx...]
    _op = "getindex"
    return Tensor(val, (arg, (idx...)), _op)
end

# RESHAPING OPERATIONS

# Naive broadcasting along 1 dimension
function broadcast( arg::Tensor, newshape::Tuple) # only from n-1 to n dimensions
    arr = arg.data
    shp = arg.shape
    
    @assert length(shp) == length(newshape)
    idx = nothing
    for i in 1:length(shp)
        if shp[i] != newshape[i] && shp[i] == 1
            idx = i
        end
    end
    
    rep = ones(Int, length(newshape))
    rep[idx] = newshape[idx]
    
    arr_new = repeat(arr, outer= rep)
    _op = "broadcast"
        
    return Tensor(arr_new, (arg, shp, idx), _op )
end


# BINARY OPERATIONS

import Base:+, *, ^, exp, -, /

# Tensor addition
function +(arg1::Tensor, arg2::Tensor)
    Tensor(arg1.data .+ arg2.data,  (arg1, arg2), "+")
end

# Tensor subtraction
function -(arg1::Tensor, arg2::Tensor)
    return arg1 + (-arg2)
end

# Matrix multiplication
function *(arg1::Tensor, arg2::Tensor)
    if(ndims(arg1.data) == ndims(arg2.data) ==2  ) # vanilla matmult
        Tensor(arg1.data * arg2.data,  (arg1, arg2), "*")
    else
        A = arg1.data; B = arg2.data
        C = Array{T}(undef, size(A, 1), size(B)[2:end]...)
        Threads.@threads for I in CartesianIndices(axes(A)[3:end])
            @views C[:, :, Tuple(I)...] = A[:, :, Tuple(I)...] * B[:, :, Tuple(I)...]
        end

        Tensor(C, (arg1, arg2), '*')
    end
end

# Element wise multiplication
function elemwise(arg1::Tensor, arg2::Tensor)
    @assert arg1.shape == arg2.shape
    val = arg1.data .* arg2.data
    Tensor(val,(arg1, arg2) ,"elemwise")
end

# Element wise division
function /(arg1::Tensor, arg2::Tensor)
    return elemwise(arg1, 1/arg2)
end

# AGGREGATION

# Sum along some dimension
import Base:sum
function sum(arg::Tensor, dims::Integer)
    val = sum(arg.data, dims = dims)
    _op = "sum"
    return Tensor(val, (arg, dims), _op)
end



