module PeriodicIntervals
using DomainSets, BasisFunctions, IntervalSets

using DomainSets: width

import DomainSets: indomain, approx_indomain, infimum, supremum, elements, numelements,
    element
import BasisFunctions: iscomposite, period

export PeriodicInterval
"""
For two given intervals `[a,b]` and `[A,B]`, a periodic subinterval represents
the intersection of the closed interval `[A,B]` with the periodic repetition of
the closed interval `[a,b]` with period `B-A`:

`[a+k*(B-A),b+k*(B-A))] ∩ [A,B]`.

Depending on the relative location of the two intervals, the periodic subinterval
may be a single interval or a union of two intervals.
"""
struct PeriodicInterval{T} <: Domain{T}
    subdomain       ::  ClosedInterval{T}
    periodicdomain  ::  ClosedInterval{T}
    numelements     ::  Int
    interval1       ::  ClosedInterval{T}
    interval2       ::  ClosedInterval{T}
end

# Invoke PeriodicInterval{T}
PeriodicInterval(subdomain::AbstractInterval{T}, periodicdomain::AbstractInterval{T}) where {T} =
    PeriodicInterval{T}(subdomain, periodicdomain)

# Convert the domains to closed intervals
PeriodicInterval{T}(subdomain::AbstractInterval{T}, periodicdomain::AbstractInterval{T}) where {T} =
    PeriodicInterval{T}(Interval(extrema(subdomain)...), Interval(extrema(periodicdomain)...))

period(d::PeriodicInterval) = width(d.periodicdomain)

# Figure out the subintervals upon construction. Allow for some small tolerance.
function PeriodicInterval{T}(subdomain::ClosedInterval{T}, periodicdomain::ClosedInterval{T}) where {T}
    tol = 100eps(T)
    a, b = extrema(subdomain)
    A, B = extrema(periodicdomain)
    if b-a >= B-A-tol
        numelements = 1
        interval1 = ClosedInterval(A, B)
        interval2 = ClosedInterval(A, A)
    else
        a = A + mod(a-A, B-A)
        b = A + mod(b-A, B-A)
        if (b < a)
            # We make sure that each interval is at least tol wide
            if (a < B-tol)
                if (b > A+tol)
                    numelements = 2
                    interval1 = ClosedInterval(A, b)
                    interval2 = ClosedInterval(a, B)
                else
                    numelements = 1
                    interval1 = ClosedInterval(a, B)
                    interval2 = ClosedInterval(A, A)
                end
            else
                if (b > A+tol)
                    numelements = 1
                    interval1 = ClosedInterval(A, b)
                    interval2 = ClosedInterval(A, A)
                else
                    # This should never happen, they are both close to the endpoints
                    numelements = 1
                    interval1 = ClosedInterval(A, B)
                    interval2 = ClosedInterval(A, A)
                end
            end
        else
            numelements = 1
            interval1 = ClosedInterval(a, b)
            interval2 = ClosedInterval(A, A)
        end
    end
    PeriodicInterval{T}(subdomain, periodicdomain, numelements, interval1, interval2)
end

iscomposite(domain::PeriodicInterval) = true

numelements(domain::PeriodicInterval) = domain.numelements

function element(domain::PeriodicInterval, idx)
    if idx == 1
        domain.interval1
    elseif idx == 2
        domain.interval2
    else
        error("Index too large in element function of PeriodicInterval: ", idx)
    end
end

elements(domain::PeriodicInterval) =
    domain.numelements == 1 ? [domain.interval1] : [domain.interval1, domain.interval2]

indomain(x, d::PeriodicInterval) =
    _indomain(x, d, d.numelements, d.interval1, d.interval2)

_indomain(x, d::PeriodicInterval, numelements, interval1, interval2) =
    numelements == 1 ? x ∈ interval1 : (x ∈ interval1 || x ∈ interval2)

approx_indomain(x, d::PeriodicInterval, tolerance) =
    _approx_indomain(x, d, tolerance, d.numelements, d.interval1, d.interval2)

function _approx_indomain(x, d::PeriodicInterval, tolerance, numelements, interval1, interval2)
    if numelements == 1
        DomainSets.approx_indomain(x, interval1, tolerance)
    else
        DomainSets.approx_indomain(x, interval1, tolerance) || DomainSets.approx_indomain(x, interval2, tolerance)
    end
end

infimum(d::PeriodicInterval) = infimum(d.interval1)

extremum(d::PeriodicInterval) = numelements(d) == 1 ? extremum(d.interval1) : extremum(d.interval2)

# Type-unsafe: intersection with an interval
function Base.intersect(d::PeriodicInterval, a::AbstractInterval)
    if numelements(d) > 1
        UnionDomain(intersect(element(d,1), a),intersect(element(d,2), a))
    else
        intersect(element(d,1), a)
    end
end



end
