"""Regression checks for generated gravity constant precision and sign sharing."""
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys
import re
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fmm.constants import hoistConstants, scientific


class GravityConstants(unittest.TestCase):
    def test_exact_rational_rounding(self):
        self.assertEqual(scientific(Fraction(2, 3)), '6.666666666666666667E-01')
        self.assertEqual(scientific(Fraction(1, 3)), '3.333333333333333333E-01')
        self.assertEqual(scientific(Fraction(4, 3)), '1.333333333333333333E+00')
        self.assertEqual(scientific(Fraction(0)), '0.000000000000000000E+00')
        self.assertEqual(scientific(Fraction(1, 10**20)), '1.000000000000000000E-20')
        self.assertEqual(scientific(Fraction(10**30)), '1.000000000000000000E+30')

    def test_rounding_midpoints_and_exponent_carry(self):
        midpoint = Fraction('1.0000000000000000005')
        delta = Fraction(1, 10**200)
        self.assertEqual(scientific(midpoint - delta), '1.000000000000000000E+00')
        self.assertEqual(scientific(midpoint), '1.000000000000000000E+00')
        self.assertEqual(scientific(midpoint + delta), '1.000000000000000001E+00')
        self.assertEqual(scientific(Fraction('1.0000000000000000015')),
                         '1.000000000000000002E+00')
        carry = Fraction('9.9999999999999999995')
        self.assertEqual(scientific(carry - delta), '9.999999999999999999E+00')
        self.assertEqual(scientific(carry), '1.000000000000000000E+01')
        self.assertEqual(scientific(Fraction(1, 10**100)), '1.000000000000000000E-100')
        self.assertEqual(scientific(Fraction(10**100)), '1.000000000000000000E+100')

    def test_sorted_magnitudes_and_padding(self):
        for count in (1, 9, 10, 15, 100, 101):
            with self.subTest(count=count):
                body = ''.join(f'\n    x += math::broadcast<Real>({i}.0 / 7.0);'
                               f'\n    x -= math::broadcast<Real>(-{i}.0 / 7.0);'
                               for i in range(count, 0, -1))
                result = hoistConstants('CUDA_EXPORT void kernel(Real x) {' + body + '\n}')
                table = re.findall(r'constexpr auto (k\d+) = static_cast<Scalar>\(([^)]+)\);', result)
                width = len(str(count - 1))
                self.assertEqual([name for name, _ in table],
                                 [f'k{i:0{width}d}' for i in range(count)])
                self.assertEqual([literal for _, literal in table],
                                 [scientific(Fraction(i, 7)) for i in range(1, count + 1)])
                for i in range(1, count + 1):
                    name = f'k{i - 1:0{width}d}'
                    self.assertIn(f'x += math::broadcast<Real>({name});', result)
                    self.assertIn(f'x -= -math::broadcast<Real>({name});', result)

    def test_shared_signs_and_function_scope(self):
        source = '''CUDA_EXPORT void kernel(Real x) {
    constexpr auto k0 = -2.0 / 3.0;
    Real const a = math::broadcast<Real>(k0) * x;
    if (flag) { output = math::broadcast<Real>(2.0 / 3.0) * a; }
    output += math::broadcast<Real>(ix);
}
CUDA_EXPORT void second(Real x) {
    output = math::broadcast<Real>(1);
}
'''
        result = hoistConstants(source)
        self.assertEqual(result.count('using Scalar = SimdScalar<Real>;'), 2)
        self.assertEqual(result.count('constexpr auto k0'), 2)
        self.assertIn('static_cast<Scalar>(6.666666666666666667E-01)', result)
        self.assertIn('-math::broadcast<Real>(k0) * x', result)
        self.assertIn('math::broadcast<Real>(k0) * a', result)
        self.assertIn('math::broadcast<Real>(ix)', result)
        self.assertLess(result.index('constexpr auto'), result.index('Real const a'))
        self.assertNotIn('k1', result)

    def test_unqualified_geometry_constants(self):
        source = 'CUDA_EXPORT Real inverse(Real r) { return broadcast<Real>(1) / r; }'
        result = hoistConstants(source)
        self.assertIn('static_cast<Scalar>(1.000000000000000000E+00)', result)
        self.assertNotIn('broadcast<Real>(1)', result)

    def test_generated_coefficients_are_named(self):
        root = Path(__file__).resolve().parents[1] / 'cosmictiger/gravity/kernels'
        for path in root.rglob('*.hpp'):
            with self.subTest(path=path.name):
                code = path.read_text()
                self.assertIsNone(re.search(r'broadcast<Real>\([+-]?(?:\d|\.\d)', code))
                for literal in re.findall(r'constexpr auto k\d+ = static_cast<Scalar>\(([^)]+)\)', code):
                    self.assertRegex(literal, r'^\d\.\d{18}E[+-]\d{2,}$')

    def test_generated_tables_are_sorted_and_padded(self):
        root = Path(__file__).resolve().parents[1] / 'cosmictiger/gravity/kernels'
        for path in root.rglob('*.hpp'):
            for table in path.read_text().split('using Scalar = SimdScalar<Real>;')[1:]:
                declarations = re.findall(
                    r'constexpr auto (k\d+) = static_cast<Scalar>\(([^)]+)\);', table)
                with self.subTest(path=path.name, first=declarations[:1]):
                    width = len(str(len(declarations) - 1))
                    self.assertEqual([name for name, _ in declarations],
                                     [f'k{i:0{width}d}' for i in range(len(declarations))])
                    values = [Decimal(literal) for _, literal in declarations]
                    self.assertEqual(values, sorted(set(values)))
                    for name, _ in declarations:
                        self.assertRegex(table, rf'broadcast<Real>\({name}\)')

    def test_no_constants_no_alias(self):
        source = 'CUDA_EXPORT Real identity(Real x) { return x; }'
        self.assertEqual(hoistConstants(source), source)


if __name__ == '__main__':
    unittest.main()
