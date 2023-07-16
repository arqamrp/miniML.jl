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
