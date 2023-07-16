
# Sigmoid function
function sigmoid(arg::Tensor)
    exparr = exp(arg)
    ans = exparr/ (1+exparr)
    return ans
end


# ReLu function
function relu(arg::Tensor)
    elemwise((arg > 0), (arg))
end

# Softmax function

function softmax(arg::Tensor, dim::Integer)
    e_arg = exp(arg)
    sums = sum(e_arg, dim)
    bsums = broadcast(sums, size(e_arg))
    return e_arg/bsums
end