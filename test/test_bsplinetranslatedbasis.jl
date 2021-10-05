using BasisFunctions, DomainSets, CompactTranslatesDict, Test, LinearAlgebra

using BasisFunctions.Test: test_orthogonality_orthonormality
using BasisFunctions: evaluation, dense_evaluation

using CompactTranslatesDict: compatible_interpolationgrid

function test_generic_periodicbsplinebasis(B,T)
    tol = sqrt(eps(real(T)))
    n = 5
    b = B(n,3, T)
    @test support(b)≈UnitInterval{T}()

    @test length(b)==5
    @test CompactTranslatesDict.degree(b)==3

    @test infimum(CompactTranslatesDict.kernel_support(b)) <= 0 <= supremum(CompactTranslatesDict.kernel_support(b))
    @test 0 < supremum(CompactTranslatesDict.kernel_support(b)) - infimum(CompactTranslatesDict.kernel_support(b)) < BasisFunctions.period(b)

    @test resize(b, 20)==B(20,CompactTranslatesDict.degree(b),T)
    @test interpolation_grid(b)==PeriodicEquispacedGrid{T}(n,0,1)
    @test BasisFunctions.period(b)==T(1)
    @test step(b)==T(1//5)
end

function test_translatedbsplines(T)

    tol = sqrt(eps(real(T)))
    n = 5
    bb = BSplineTranslatesBasis(n, 1, T; scaled=true)
    b = BSplineTranslatesBasis(n, 1, T; scaled=false)
    e = rand(T,n)
    @test norm(gram(b)*e-gram(bb)*e/n) < tol

    b = BSplineTranslatesBasis(n,3, T)

    @test BasisFunctions.name(b) == "Periodic equispaced translates of B spline of degree 3"

    @test compatible_interpolationgrid(b, interpolation_grid(b))
    @test compatible_interpolationgrid(b, MidpointEquispacedGrid(n,0,1))
    @test !compatible_interpolationgrid(b, PeriodicEquispacedGrid(n+1,0,1))
    @test !compatible_interpolationgrid(b, PeriodicEquispacedGrid(n,0.1,1))
    @test !compatible_interpolationgrid(b, PeriodicEquispacedGrid(n,0,1.1))

    interpolation_grid(BSplineTranslatesBasis(n,2, T)) == MidpointEquispacedGrid(n,0,1)
    @test CompactTranslatesDict.degree(BSplineTranslatesBasis(5,2, T)) == 2
    b = BSplineTranslatesBasis(n,2,T)
    @test compatible_interpolationgrid(b, interpolation_grid(b))
    @test compatible_interpolationgrid(b, PeriodicEquispacedGrid(n,0,1))
    @test !compatible_interpolationgrid(b, MidpointEquispacedGrid(n+1,0,1))
    @test !compatible_interpolationgrid(b, MidpointEquispacedGrid(n,0.1,1))
    @test !compatible_interpolationgrid(b, MidpointEquispacedGrid(n,0,1.1))

    if T == Float64
        for K in 1:3
            for s2 in 5:6
                s1 = s2<<1
                b1 = BSplineTranslatesBasis(s1,K,T)
                b2 = BSplineTranslatesBasis(s2,K,T)

                @test dense_evaluation(T, b2, GridBasis(b2)) ≈ evaluation(T, b2, GridBasis(b2), interpolation_grid(b2))
                @test dense_evaluation(T, b2, GridBasis(b1)) ≈ evaluation(T, b2, GridBasis(b1), interpolation_grid(b1))
                @test dense_evaluation(T, b1, GridBasis(b1)) ≈ evaluation(T, b1, GridBasis(b1), interpolation_grid(b1))
                @test dense_evaluation(T, b1, GridBasis(b2)) ≈ evaluation(T, b1, GridBasis(b2), interpolation_grid(b2))
            end
        end

        for K in 1:3 # 0 not robust to test since it is discontinuous.
            for s2 in 5:6
                s1 = s2<<1
                b1 = BSplineTranslatesBasis(s1,K,T; scaled=true)
                b2 = BSplineTranslatesBasis(s2,K,T; scaled=true)

                @test dense_evaluation(T, b2, GridBasis(b2)) ≈ evaluation(T, b2, GridBasis(b2), interpolation_grid(b2))
                @test dense_evaluation(T, b2, GridBasis(b1)) ≈ evaluation(T, b2, GridBasis(b1), interpolation_grid(b1))
                @test dense_evaluation(T, b1, GridBasis(b1)) ≈ evaluation(T, b1, GridBasis(b1), interpolation_grid(b1))
                @test dense_evaluation(T, b1, GridBasis(b2)) ≈ evaluation(T, b1, GridBasis(b2), interpolation_grid(b2))
            end
        end
    end
end


function test_bspline_orthogonality_orthonormality()
    B = BSplineTranslatesBasis(4,3)
    for m in [FourierWeight(),
                discretemeasure(PeriodicEquispacedGrid(4,0,1)),
                discretemeasure(MidpointEquispacedGrid(4,0,1)),
                discretemeasure(PeriodicEquispacedGrid(8,0,1)),
                discretemeasure(MidpointEquispacedGrid(8,0,1))]
        @test BasisFunctions.unsafe_matrix(gram(B, m)) isa Circulant
        test_orthogonality_orthonormality(B, false, false, m; overquad=10)
    end
end



# using Plots, BasisFunctions, CompactTranslatesDict
# n = 7
# k = 0
# b = BSplineTranslatesBasis(n,k)
# t = LinRange(-0,2,200)
# f = map(x->BasisFunctions.unsafe_eval_element(b,1,x), t)
#
# plot(t,f)
# f = CompactTranslatesDict.eval_kernel.(Ref(b), t)
# plot!(t,f)
# f = map(x->BasisFunctions.eval_expansion(b,ones(n),x),t)
# # f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0,0,0,0,0,0],x),t)
# plot!(t,f,ylims=[-n,2n])
