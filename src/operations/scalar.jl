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
