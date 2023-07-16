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


# AGGREGATION

# Sum along some dimension
import Base:sum
function sum(arg::Tensor, dims::Integer)
    val = sum(arg.data, dims = dims)
    _op = "sum"
    return Tensor(val, (arg, dims), _op)
end

# Mean along some dimension

function mean(arg::Tensor,dim::Integer)
    sum = sum(arg, dim)
    mean = sum/size(arg)[dim]
    return mean
end

