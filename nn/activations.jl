
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