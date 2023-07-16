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

Base.:(==)(arg1::Tensor, arg2::Tensor) = isequal(arg1, arg2)