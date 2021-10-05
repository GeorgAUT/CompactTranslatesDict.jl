
using BasisFunctions: VerticalBandedMatrix, default_mixedgram_discretemeasure,
    LebesgueDomain, DiscreteWeight
using DomainSets: width
using GridArrays: similargrid

import BasisFunctions: isperiodic, period, support, name, hasgrid_transform, transform_from_grid,
    transform_to_grid, evaluation, gram, unsafe_eval_element, unsafe_eval_element_derivative,
    similar, rescale, dict_in_support
import LinearAlgebra: norm

export PeriodicEquispacedTranslates
"""
    abstract type PeriodicEquispacedTranslates{T,S,PERIODIZATION} <: EquispacedTranslates{T,S}

A dictionary consisting of the equispaced translates of a periodic kernel.
The kernel is periodized during evaluation.
"""
abstract type PeriodicEquispacedTranslates{T,S,PERIODIZATION} <: EquispacedTranslates{T,S}
end

compatible_translationgrid(::Type{<:PeriodicEquispacedTranslates}, grid::AbstractGrid) =
    compatible_translationgrid(EquispacedTranslates, grid) && isperiodic(grid)
isperiodic(::PeriodicEquispacedTranslates) = true
function dict_in_support(dict::PeriodicEquispacedTranslates, idxn, x)
    c = x-translationgrid(dict)[idxn]
    per = period(dict)
    A = infimum(kernel_support(dict))
    steps =floor((c-A)/per)
    c -= steps*per
    c ∈ kernel_support(dict)
end

"""
    period(dict::PeriodicEquispacedTranslates)

The period of the periodic dictionary
"""
period(dict::PeriodicEquispacedTranslates) = width(support(dict))
support(dict::PeriodicEquispacedTranslates, idx) =
	PeriodicInterval(translationgrid(dict)[idx] .+ kernel_support(dict), support(dict))

name(::PeriodicEquispacedTranslates) = "Periodic equispaced translates of a periodic kernel function"


export CompactTranslationDict
const CompactTranslationDict = PeriodicEquispacedTranslates

hasgrid_transform(dict::PeriodicEquispacedTranslates, _, grid::AbstractEquispacedGrid) =
    size(dict)==size(grid) && compatible_translationgrid(typeof(dict), grid)

transform_from_grid(src, dest::PeriodicEquispacedTranslates, grid; options...) =
    inv(transform_to_grid(dest, src, grid; options...))

function transform_to_grid(src::PeriodicEquispacedTranslates, dest, grid::AbstractEquispacedGrid; options...)
    @assert hasgrid_transform(src, dest, grid)
    CirculantOperator(sample(grid, x->eval_kernel(src, x)), src, dest; options...)
end

function vertical_banded_matrix(dict::Dictionary, kernel, kernel_support::AbstractInterval, support::AbstractInterval, m::Int, y, T)
    l, r = extrema(support)
    width = r-l
    step = width/m
    a, b = extrema(kernel_support)
    c, d = ceil(Int,(a+y-l)/step), floor(Int,(b+y-l)/step)
    offset = c
    x = l .+ LinRange(c, d, (d-c+1)).*step
    a = convert.(T, kernel.(x.-y))
    f = 1
    l = length(a)
    for i in 1:length(a)
        if !(a[i] + 1 ≈ 1)
            break
        end
        f += 1
    end
    for j in length(a):-1:1
        if !(a[j] + 1 ≈ 1)
            break
        end
        l -= 1
    end
    VerticalBandedMatrix(m, length(dict), a[f:l], div(m,length(dict)), offset+f-1)
end

function evaluation(::Type{T}, dict::PeriodicEquispacedTranslates, dgs::GridBasis, grid::AbstractEquispacedGrid;
            options...) where {T}
    lg = length(grid)
    ls = length(dict)
    m, rem = divrem(lg, ls)
    if rem == 0
        M = vertical_banded_matrix(dict, x->eval_kernel(dict, x), kernel_support(dict), support(dict), m*length(dict), translationgrid(dict)[1]-grid[1], T)
        # firstcolumn = sample(grid, x->(unsafe_eval_element(s, 1, x)))
        # firstcolumn = convert.(T, firstcolumn)
        # a, offset = _get_array_offset(firstcolumn)
        # M = VerticalBandedMatrix{T}(length(dgs), length(s), a, m, offset-1)
        ArrayOperator{T}(M, dict, dgs)
        # # VerticalBandedOperator(s, dgs, a, m, offset-1; T=T)
    else
        @debug "slow evaluation operator"
        # Not type stable if we allow this branch
        BasisFunctions.dense_evaluation(T, dict, dgs; options...)
        # error("Not type stable if we allow this branch")
    end
end

if isdefined(BasisFunctions, :isgramcompatible)
    import BasisFunctions: isgramcompatible
else
    isgramcompatible(dict::Dictionary, grid::AbstractGrid) = false
end

function isgramcompatible(b::PeriodicEquispacedTranslates, grid::AbstractEquispacedGrid)
    l1 = length(b)
    l2 = length(grid)
    l1 > l2 && ((l2,l1) = (l1, l2))
    n = l2/l1
    nInt = round(Int, n)
    support(b)≈coverdomain(grid) && (n≈nInt)
end

function gram(::Type{T}, dict::PeriodicEquispacedTranslates, measure::DiscreteWeight, grid::AbstractEquispacedGrid, weights::FillArrays.AbstractFill;
        options...) where {T}
    if isgramcompatible(dict, grid)
        CirculantOperator(default_mixedgram_discretemeasure(T, dict, dict, measure, grid, weights; options...))
    else
        default_mixedgram_discretemeasure(T, dict, dict, measure, grid, weights; options...)
    end
end


function gram(::Type{T}, dict::PeriodicEquispacedTranslates, measure::Union{LebesgueDomain,LegendreWeight,FourierWeight};
        options...) where {T}
    @assert support(dict) ≈ support(measure)
    CirculantOperator{T}(firstgramcolumn(T, dict, measure; options...), dict, dict)
end

function firstgramcolumn(T, dict::Dictionary, measure::Weight; options...)
    firstcolumn = zeros(T, length(dict))
    for (index,i) in enumerate(ordering(dict))
        firstcolumn[index] = innerproduct(dict, i, dict, ordering(dict)[1], measure; options...)
    end
    firstcolumn
end

for (f,g) in ((:unsafe_eval_element,:unsafe_eval_kernel),
                (:unsafe_eval_element_derivative,:unsafe_eval_kernel_derivative))
    # The evaluation of the basis functions explicitly periodizes the kernel function
    # by summing over its translates.
    @eval function $f(dict::PeriodicEquispacedTranslates{T,S,:sum}, idx, x, args...) where {T,S}
        c = translationgrid(dict)[idx]
        per = period(dict)
		A,B = extrema(kernel_support(dict))

        z = $g(dict, x - c, args...)
    	# Now evaluate the periodic extension. We add and subtract the period
    	# repeatedly until we are beyond the support of the kernel.
        x1 = x + per
        while (x1 <= c + B)
            z += $g(dict, x1-c, args...)
            x1 += per
        end
        x2 = x - per
        while (x2 >= c + A)
            z += $g(dict, x2-c, args...)
    		x2 -= per
        end
        z
    end

    @eval $f(dict::PeriodicEquispacedTranslates{T,S,PERIODIZATION}, args...) where {T,S,PERIODIZATION} =
        error("Periodization $PERIODIZATION not known.")

    function norm(dict::PeriodicEquispacedTranslates{T,S,:norm}, x, y) where {T,S}
        per = period(dict)
        ELT = typeof(per)
        # r = max(abs.(extrema(kernel_support(dict)))...)
        # t = r^2/(1-cos(2ELT(pi)/per*r))
        sqrt((1-cos(2ELT(pi)/per*(x - y))))
    end
    @eval function $f(dict::PeriodicEquispacedTranslates{T,S,:norm}, idx, x, args...) where {T,S}
        c = translationgrid(dict)[idx]
        z = $g(dict, norm(dict, x, c), args...)
    end
end


export GenericPeriodicEquispacedTranslates
"""
struct GenericPeriodicEquispacedTranslates{T,S,PERIODIZATION} <: PeriodicEquispacedTranslates{T,S,PERIODIZATION}

A `Translates` with a general periodic equispaced grid and periodic kernel

# Example
```jldocs
julia> GenericPeriodicEquispacedTranslates(PeriodicEquispacedGrid(10,0,2π), cos)
GenericPeriodicEquispacedTranslates
    ↳ length = 10
    ↳ Float64 -> Float64
    ↳ support = 0.0..6.283185307179586
```
"""
struct GenericPeriodicEquispacedTranslates{T,S,PERIODIZATION,GRID<:AbstractGrid,DOMAIN<:Domain,KERN} <: PeriodicEquispacedTranslates{T,S,PERIODIZATION}
    grid::GRID
    kernel::KERN
    kernel_support::DOMAIN
    function GenericPeriodicEquispacedTranslates(grid, kernel, kernel_support=DomainSets.FullSpace{typeof(first(grid))}(),periodization=:sum)
        T = eltype(grid)
        S = typeof(kernel(first(grid)))
        @assert compatible_translationgrid(GenericPeriodicEquispacedTranslates, grid)
        new{T,S,periodization,typeof(grid),typeof(kernel_support),typeof(kernel)}(grid, kernel, kernel_support)
    end
end

similar(dict::GenericPeriodicEquispacedTranslates{K,S,PERIODIZATION}, ::Type{T}, n::Int...) where {K,S,T,PERIODIZATION} =
    GenericPeriodicEquispacedTranslates(similargrid(translationgrid(dict),real(T), n...), dict.kernel, dict.kernel_support,PERIODIZATION)

function rescale(dict::PeriodicEquispacedTranslates, a::T, b::T) where {T<:Number}
    map = DomainSets.interval_map(extrema(support(dict))..., a, b)
    GenericPeriodicEquispacedTranslates(map_grid(map,translationgrid(dict)),
        x->eval_kernel(dict, inverse(map, x)),
        map.(kernel_support(dict)))
end
