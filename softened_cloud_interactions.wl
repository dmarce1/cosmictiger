(*
  Gravity between two spherical softened mass distributions.

  Open this file in Mathematica and evaluate it, or use:
    WolframKernel -noinit -noprompt -run 'Get["softened_cloud_interactions.wl"]; Quit[]'

  EDIT THE INPUT SECTION AT THE BOTTOM.
  A profile is a radial DENSITY shape, not a force or potential.
  makeCloud[Function[q, expression], scale, cutoff] interprets q = radius/scale.
  The default cutoff is 1: the density is zero outside radius = scale.
  Use cutoff = Infinity for an infinite-support profile, e.g. Exp[-q^2/2].
  Profiles are normalized automatically. Use nonnegative, bounded profiles
  with finite nonzero mass; infinite-support integrals must converge.
  The two clouds may have different profiles, scales, and masses.
  Supply exact constants (1/3, not 0.3333) for high-precision integration.

  pairForceMagnitude returns the positive attraction per G m1 m2.
  pairPotentialMagnitude returns -U/(G m1 m2), with U zero at infinity.
  Both include the full extent of BOTH clouds, at fixed profiles and scales.
  They do not include derivatives of adaptive, position-dependent scales.
  symbolicEqualCloudPair[profile] optionally derives exact expressions for
  two identical unit-radius compact clouds. It has a time limit and reports
  failure if Mathematica cannot finish the integrals; the numerical routines
  remain available for those profiles and for unequal or infinite supports.
  No parallel kernels are launched.
*)

ClearAll[makeCloud, cloudDensity, integrateLine, integrateRectangle,
  shellPotential, containedPotentialMagnitude, pairPotentialMagnitude, pairForceMagnitude,
  pointCloudForceMagnitude, pairEnergy, pairForceVector,
  symbolicEqualCloudPair,
  parabolicDensityConvolution, parabolicForceFactor, parabolicPotential,
  parabolicExactChecks, exampleTable];

workingPrecision = 35;
accuracyGoal = 10;
precisionGoal = 10;
maximumRecursion = 20;

integrateLine[f_, lower_, upper_] := Module[{x},
  If[TrueQ[lower == upper], Return[0]];
  NIntegrate[f[x], {x, lower, upper},
    WorkingPrecision -> workingPrecision,
    AccuracyGoal -> accuracyGoal, PrecisionGoal -> precisionGoal,
    MaxRecursion -> maximumRecursion,
    Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0}]
];

integrateRectangle[f_, xLower_, xUpper_, yLower_, yUpper_] := Module[{x, y},
  If[TrueQ[xLower == xUpper || yLower == yUpper], Return[0]];
  NIntegrate[f[x, y], {x, xLower, xUpper}, {y, yLower, yUpper},
    WorkingPrecision -> workingPrecision,
    AccuracyGoal -> accuracyGoal, PrecisionGoal -> precisionGoal,
    MaxRecursion -> maximumRecursion,
    Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0}]
];

makeCloud::scale = "The scale must be a finite positive real number and the cutoff must be positive or Infinity.";
makeCloud::mass = "The density profile did not produce a finite positive real normalization.";
makeCloud[profile_, scale_: 1, cutoff_: 1] := Module[{integrand, normalization},
  If[!TrueQ[0 < scale < Infinity] || !TrueQ[0 < cutoff <= Infinity],
    Message[makeCloud::scale]; Return[$Failed]];
  integrand[q_?NumericQ] := q^2 profile[q];
  normalization = 4 Pi integrateLine[integrand, 0, cutoff];
  If[!NumberQ[normalization] || !TrueQ[Im[normalization] == 0] ||
      !TrueQ[normalization > 0],
    Message[makeCloud::mass]; Return[$Failed]];
  <|"Profile" -> profile, "Scale" -> scale, "Cutoff" -> cutoff,
    "Support" -> scale cutoff, "Normalization" -> normalization|>
];

(* Unit-mass density in physical radial coordinates. *)
cloudDensity[cloud_Association, radius_?NumericQ] :=
  If[TrueQ[radius < 0 || radius > cloud["Support"]], 0,
    cloud["Profile"][radius/cloud["Scale"]]/
      (cloud["Normalization"] cloud["Scale"]^3)];

(* Unit-mass thin spherical shells, radii s and t, center separation r.
   This is their positive interaction-potential magnitude. *)
shellPotential[r_?NumericQ, s_?NumericQ, t_?NumericQ] := Which[
  TrueQ[Max[r, s, t] == 0], 0, (* zero-measure endpoint of the mass integral *)
  TrueQ[r <= Abs[s - t]], 1/Max[s, t],
  TrueQ[r >= s + t], 1/r,
  True, (2 (s + t) - r - (s - t)^2/r)/(4 s t)
];

(* Contribution from shells of a wholly inside shells of b: t >= s+r.
   The infinite tail, when present, is mapped to a unit interval. *)
containedPotentialMagnitude[a_Association, b_Association, r_?NumericQ] :=
  Module[{upper, integrand},
    upper = Min[a["Support"], Max[0, b["Support"] - r]];
    If[TrueQ[upper == 0], Return[0]];
    integrand[s_?NumericQ, z_?NumericQ] := Module[{lower, t, jacobian},
      lower = s + r;
      If[b["Support"] === Infinity,
        If[TrueQ[z == 1], Return[0]];
        t = lower + b["Scale"] z/(1 - z);
        jacobian = b["Scale"]/(1 - z)^2,
        jacobian = Max[0, b["Support"] - lower];
        t = lower + jacobian z
      ];
      16 Pi^2 s^2 t cloudDensity[a, s] cloudDensity[b, t] jacobian
    ];
    integrateRectangle[integrand, 0, upper, 0, 1]
  ];

pairPotentialMagnitude[a_Association, b_Association, separation_?NumericQ] :=
  Module[{r = N[Abs[separation], workingPrecision], supportA, supportB,
      contained, separatedIntegrand, overlapIntegrand, separated, overlap},
    supportA = a["Support"];
    supportB = b["Support"];
    If[TrueQ[r >= supportA + supportB], Return[1/r]];
    contained = containedPotentialMagnitude[a, b, r] +
      containedPotentialMagnitude[b, a, r];
    If[TrueQ[r == 0], Return[contained]];

    separatedIntegrand[s_?NumericQ, z_?NumericQ] := Module[{width, t},
      width = Min[supportB, r - s];
      t = width z;
      16 Pi^2 s^2 t^2 cloudDensity[a, s] cloudDensity[b, t] width/r
    ];
    separated = integrateRectangle[separatedIntegrand,
      0, Min[supportA, r], 0, 1];

    overlapIntegrand[s_?NumericQ, z_?NumericQ] :=
      Module[{etaLower, etaUpper, etaWidth, eta, t},
        etaLower = Max[-1, 1 - 2 s/r];
        etaUpper = Min[1, (supportB - s)/r];
        etaWidth = Max[0, etaUpper - etaLower];
        eta = etaLower + etaWidth z;
        t = s + r eta;
        4 Pi^2 s t cloudDensity[a, s] cloudDensity[b, t]
          (2 (s + t) - r (1 + eta^2)) r etaWidth
      ];
    overlap = integrateRectangle[overlapIntegrand,
      Max[0, r - supportB], Min[supportA, r + supportB], 0, 1];
    contained + separated + overlap
  ];

(* The shell-force kernel is 1/r^2 when r >= s+t, zero when r <= |s-t|,
   and (1-(s-t)^2/r^2)/(4 s t) otherwise. We integrate the nonzero regions
   separately. In the overlap region eta=(t-s)/r stays in [-1,1], even at
   tiny r. This avoids missing a thin diagonal strip or subtracting nearly
   equal shell radii to evaluate the force. Integrating force/r and restoring
   r at the end also prevents absolute error tolerances from swallowing the
   small force near coincident centers. *)
pairForceMagnitude[a_Association, b_Association, separation_?NumericQ] :=
  Module[{r = N[Abs[separation], workingPrecision], supportA, supportB,
      separatedIntegrand, overlapIntegrand, separated, overlap, lower, upper},
    If[TrueQ[r == 0], Return[0]];
    supportA = a["Support"];
    supportB = b["Support"];
    If[TrueQ[r >= supportA + supportB], Return[1/r^2]];

    separatedIntegrand[s_?NumericQ, z_?NumericQ] := Module[{width, t},
      width = Min[supportB, r - s];
      t = width z;
      16 Pi^2 s^2 t^2 cloudDensity[a, s] cloudDensity[b, t] width/r^3
    ];
    separated = integrateRectangle[separatedIntegrand,
      0, Min[supportA, r], 0, 1];

    overlapIntegrand[s_?NumericQ, z_?NumericQ] :=
      Module[{etaLower, etaUpper, etaWidth, eta, t},
        etaLower = Max[-1, 1 - 2 s/r];
        etaUpper = Min[1, (supportB - s)/r];
        etaWidth = Max[0, etaUpper - etaLower];
        eta = etaLower + etaWidth z;
        t = s + r eta;
        4 Pi^2 s t cloudDensity[a, s] cloudDensity[b, t]
          (1 - eta) (1 + eta) etaWidth
      ];
    lower = Max[0, r - supportB];
    upper = Min[supportA, r + supportB];
    overlap = integrateRectangle[overlapIntegrand, lower, upper, 0, 1];
    r (separated + overlap)
  ];

(* A point particle interacting with one cloud, for comparison. *)
pointCloudForceMagnitude[cloud_Association, separation_?NumericQ] :=
  Module[{r = N[Abs[separation], workingPrecision], integrand},
    If[TrueQ[r == 0], Return[0]];
    If[TrueQ[r >= cloud["Support"]], Return[1/r^2]];
    integrand[s_?NumericQ] := 4 Pi s^2 cloudDensity[cloud, s];
    integrateLine[integrand, 0, r]/r^2
  ];

pairEnergy[a_Association, b_Association, r_?NumericQ, m1_: 1, m2_: 1, g_: 1] :=
  -g m1 m2 pairPotentialMagnitude[a, b, r];

(* displacement = center of cloud a minus center of cloud b;
   the returned vector is the force on a. Divide by m1 for acceleration. *)
pairForceVector[a_Association, b_Association, displacement_List,
    m1_: 1, m2_: 1, g_: 1] := Module[{r = Norm[displacement]},
  If[TrueQ[r == 0], Return[ConstantArray[0, Length[displacement]]]];
  -g m1 m2 pairForceMagnitude[a, b, r] displacement/r
];

(* Optional exact calculation for identical compact clouds. Results inside
   0<q<2 are returned along with the local separation symbol. Outside this
   interval use density=0, forceFactor=1/q^3, potentialMagnitude=1/q.
   Evaluate origin limits explicitly if an expression contains removable
   singularities. Polynomial profiles in q^2 are particularly convenient. *)
symbolicEqualCloudPair[profile_, timeLimit_: 30] := TimeConstrained[
  Module[{q, z, t, s, normalization, integrand, convolution, mass,
      forceFactor, potential, result},
    normalization = 4 Pi Integrate[s^2 profile[s], {s, 0, 1}];
    integrand = profile[Sqrt[t + (z + q/2)^2]]
      profile[Sqrt[t + (z - q/2)^2]];
    convolution = FullSimplify[2 Pi/normalization^2 Integrate[
      Integrate[integrand, {t, 0, 1 - (z + q/2)^2},
        Assumptions -> 0 < q < 2 && 0 <= z <= 1 - q/2],
      {z, 0, 1 - q/2}, Assumptions -> 0 < q < 2], 0 < q < 2];
    mass = FullSimplify[4 Pi Integrate[s^2 (convolution /. q -> s),
      {s, 0, q}, Assumptions -> 0 < q < 2], 0 < q < 2];
    forceFactor = FullSimplify[mass/q^3, 0 < q < 2];
    potential = FullSimplify[mass/q +
      4 Pi Integrate[s (convolution /. q -> s), {s, q, 2},
        Assumptions -> 0 < q < 2], 0 < q < 2];
    result = <|"SeparationVariable" -> q, "Normalization" -> normalization,
      "Convolution" -> convolution, "ForceFactor" -> forceFactor,
      "PotentialMagnitude" -> potential|>;
    If[FreeQ[result, _Integrate | _ConditionalExpression], result,
      Failure["UnresolvedIntegrals", <|"MessageTemplate" ->
        "Exact integration was incomplete. Use the numerical pair routines."|>]]
  ], timeLimit,
  Failure["TimeLimit", <|"MessageTemplate" ->
    "Exact integration reached the time limit. Use the numerical pair routines or increase the limit."|>]
];

(* Exact equal-radius example: W(q)=15(1-q^2)/(8 Pi), 0<=q<=1.
   q is separation / radius of ONE cloud. The convolution supports q<=2.
   Force vector = -G m1 m2/R^3 K(q) displacement.
   Energy = -G m1 m2/R P(q). *)
parabolicDensityConvolution[q_] := Piecewise[{
  {15 (2 - q)^4 (3 q^3 + 24 q^2 + 64 q + 32)/(7168 Pi), 0 <= q <= 2}}, 0];
parabolicForceFactor[q_] := Piecewise[{
  {10/7 - 3 q^2/2 + 25 q^3/32 - 15 q^5/256 + 9 q^7/3584, 0 <= q <= 2}},
  1/q^3];
parabolicPotential[q_] := Piecewise[{
  {10/7 - 5 q^2/7 + 3 q^4/8 - 5 q^5/32 + 15 q^7/1792 - q^9/3584,
    0 <= q <= 2}}, 1/q];

(* Independent exact derivation from the intersection of two unit spheres.
   t is cylindrical radius squared and z is measured from the midpoint.
   The factors are the two unnormalized densities, integrated over the lens. *)
parabolicExactChecks[] := Module[{q, z, t, s, first, second, convolution,
    mass, forceFactor, potential},
  first = 1 - (z + q/2)^2;
  second = 1 - (z - q/2)^2;
  convolution = Expand[225/(32 Pi) Integrate[
    Integrate[(first - t) (second - t), {t, 0, first}], {z, 0, 1 - q/2}]];
  mass = Expand[4 Pi Integrate[s^2 (convolution /. q -> s), {s, 0, q}]];
  forceFactor = Cancel[mass/q^3];
  potential = Expand[mass/q +
    4 Pi Integrate[s (convolution /. q -> s), {s, q, 2}]];
  <|"Convolution" -> Factor[convolution], "ForceFactor" -> forceFactor,
    "PotentialMagnitude" -> potential,
    "Checks" -> {
      Simplify[mass /. q -> 2] == 1,
      Simplify[D[potential, q] + q forceFactor] == 0,
      FullSimplify[convolution == parabolicDensityConvolution[q], 0 < q < 2],
      FullSimplify[forceFactor == parabolicForceFactor[q], 0 < q < 2],
      FullSimplify[potential == parabolicPotential[q], 0 < q < 2],
      (forceFactor /. q -> 2) == 1/8,
      (potential /. q -> 2) == 1/2}|>
];

(* ============================ USER INPUT ============================ *)

profile1 = Function[q, 1 - q^2];
profile2 = Function[q, 1 - q^2];
radius1 = 1;
radius2 = 1;
cutoff1 = 1;
cutoff2 = 1;
mass1 = 1;
mass2 = 1;
gravitationalConstant = 1;

(* Other examples (uncomment and replace the inputs above):
   profile1 = Function[q, (1-q^2)^4];                 Ferrers
   profile1 = Function[q, (1-q)^4 (1+4 q)];          Wendland
   profile1 = Function[q, Piecewise[{
     {1-6 q^2+6 q^3, q<=1/2}}, 2 (1-q)^3]];        cubic spline
   profile1 = Function[q, Exp[-q^2/2]]; cutoff1 = Infinity; Gaussian
   profile1 = Function[q, (1+q^2)^(-5/2)]; cutoff1 = Infinity; Plummer density
   For Infinity, radius1 is a scale length, not a finite support radius.
*)

cloud1 = makeCloud[profile1, radius1, cutoff1];
cloud2 = makeCloud[profile2, radius2, cutoff2];

(* Small table by default. Increase the sampling only when needed: these
   are reference quadratures, not pretabulated production force kernels. *)
sampleSeparations = {0, 1/2, 1, 3/2, 2};
runExample = True;

exampleTable[separations_List] := Table[
  {r,
    gravitationalConstant mass1 mass2 pairForceMagnitude[cloud1, cloud2, r],
    gravitationalConstant mass1 mass2 pointCloudForceMagnitude[cloud2, r],
    pairEnergy[cloud1, cloud2, r, mass1, mass2, gravitationalConstant]},
  {r, separations}];

If[runExample && AssociationQ[cloud1] && AssociationQ[cloud2],
  Print["Exact unit-radius parabolic-profile checks: ", parabolicExactChecks[]["Checks"]];
  Print["For your selected profiles: columns are separation, cloud-cloud force magnitude, point-cloud force magnitude, pair energy."];
  Print[TableForm[N[exampleTable[sampleSeparations], 10],
    TableHeadings -> {None, {"r", "F clouds", "F point/cloud", "U clouds"}}]];
];

(* In a notebook, optional plotting and export:
   exact = symbolicEqualCloudPair[profile1];
   exact["ForceFactor"]

   samples = exampleTable[Subdivide[0, 3, 60]];
   ListLinePlot[{samples[[All,{1,2}]], samples[[All,{1,3}]]},
     PlotLegends->{"two clouds","point/cloud"},
     AxesLabel->{"separation","force magnitude"}, PlotRange->All]
   Export["softened_cloud_interactions.csv",
     Prepend[samples,{"separation","cloud_force","point_cloud_force","pair_energy"}]];
*)
