# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
"""Emit declaration-only operator headers and concrete CPU/CUDA definitions.

The optional arithmetic instrumentation sources contain generic definitions for
counting-scalar tests. Production targets compile only the concrete sources.
"""
import re
from math import isqrt
from .constants import hoistConstants

CPU_TYPES = ('float', 'double', 'SimdVector<float, 8>', 'SimdVector<double, 2>',
             'SimdVector<float, 4>', 'SimdVector<double, 4>', 'SimdVector<double, 8>')
FUNCTION = re.compile(r'template\s*<([^>]+)>\s*(?:requires\s*\([^)]*\)\s*)?'
                      r'(?:CUDA_EXPORT|__device__)\s+[^;{}]+\{')


def functions(cpp):
    position = 0
    while match := FUNCTION.search(cpp, position):
        end = match.end()
        depth = 1
        while depth:
            depth += (cpp[end] == '{') - (cpp[end] == '}')
            end += 1
        yield match, end, cpp[match.end():end - 1]
        position = end


# These operators accumulate into their final argument. Geometry and scheduling
# helpers retain their own signatures; they do not implement an X-to-Y operator.
OPERATORS = ('P2M', 'P2L', 'M2M', 'M2L', 'M2P', 'L2L', 'L2P',
             'L2LWarp', 'L2LWarpInto', 'evaluateGreen', 'evaluateEwaldGreen',
             'evaluateEwaldWorking', 'ewaldRegular')


def splitArguments(text):
    """Split C++ arguments while preserving nested template and call syntax."""
    result, start, depth = [], 0, 0
    for i, character in enumerate(text):
        if character in '(<[{':
            depth += 1
        elif character in ')>]}':
            depth -= 1
        elif character == ',' and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    result.append(text[start:].strip())
    return result


def reorderArguments(cpp):
    """Move the accumulator after the inputs in generated declarations and calls."""
    pattern = re.compile(r'\b(' + '|'.join(OPERATORS) + r')\(')
    matches = list(pattern.finditer(cpp))
    for match in reversed(matches):
        start, end, depth = match.end(), match.end(), 1
        while depth:
            depth += (cpp[end] == '(') - (cpp[end] == ')')
            end += 1
        args = splitArguments(cpp[start:end - 1])
        cpp = cpp[:start] + ', '.join(args[1:] + args[:1]) + cpp[end - 1:]
    return cpp


def singleOrderApi(outputs, degree):
    """Expose the build's one coefficient shape and deduced scalar overloads."""
    result = {}
    for path, cpp in outputs.items():
        if not path.endswith('.hpp'):
            result[path] = cpp
            continue
        cpp = re.sub(r'\b(Multipole|Expansion|Green|Derivatives)<Real, [^>]+>', r'\1<Real>', cpp)
        if path.endswith('types.hpp'):
            cpp = cpp.replace('class Real, int order', 'class Real')
            cpp = re.sub(r'\brequires\s*\(order == \d+\)', '', cpp)
            cpp = re.sub(r'\border\b', str(degree + 1), cpp)
            cpp = re.sub(r'^[ \t]*static_assert\(\d+ >= 3 && \d+ <= 9\);\n', '', cpp, flags=re.MULTILINE)
            for offset in (1, 2):
                cpp = cpp.replace(f'{degree + 1} - {offset}', str(degree + 1 - offset))
        else:
            for match, end, body in reversed(list(functions(cpp))):
                signature = match[0]
                if 'int degree' in match[1]:
                    signature = signature.replace('int degree, ', '')
                    body = re.sub(r'\bdegree\b', str(degree - 1 if path == 'spherical/p2m.hpp' else degree), body)
                if 'int order' in match[1]:
                    signature = signature.replace(', int order', '')
                    body = re.sub(r'\border\b', str(degree + 1), body)
                signature = re.sub(r'requires\s*\([^)]*\)\s*', '', signature)
                cpp = cpp[:match.start()] + signature + body + '}' + cpp[end:]
            # Scalar/degree arguments are inferred from the input and output types.
            names = '|'.join(OPERATORS + ('executeL2LWarp',))
            cpp = re.sub(r'\b(' + names + r')<[^;{}]*?>\(', r'\1(', cpp)
            cpp = reorderArguments(cpp)
        result[path] = cpp
    return result


def sphericalAccesses(path, name, body):
    """Spell harmonic storage accesses as (l, m), preserving raw-array APIs.

    The four L2P field components are not harmonic coefficients. Array-wide
    copies retain linear indexing; only harmonic arithmetic uses the views.
    """
    if not path.startswith('spherical/'):
        return body
    arrays = {
        'P2M': {'output': ('M', 'Real')},
        'L2P': {'source': ('L', 'Real const')},
        'evaluateGreen': {'output': ('G', 'Real')},
        'ewaldRegular': {'output': ('U', 'Real')},
        'evaluateEwaldWorking': {
            'output': ('G', 'Real'), 'harmonics': ('U', 'Real'),
            'compensation': ('C', 'Real'), 'imagePair': ('I', 'Real'),
        },
    }.get(name, {})
    prefix = []
    for array, (view, real) in arrays.items():
        def access(match):
            slot = int(match[1])
            l = isqrt(slot)
            m = slot - l * l - l
            return f'{view}({l}, {m})'
        body, count = re.subn(r'(?<!Real )\b' + array + r'\[(\d+)\]', access, body)
        if not count:
            continue
        declaration = f'    detail::CoefficientView<{real}> const {view}{{{array}}};\n'
        # Local scratch arrays must be declared before constructing their views.
        local = re.search(r'\bReal\s+' + array + r'\[\d+\];', body)
        if local:
            body = body[:local.end()] + '\n' + declaration + body[local.end():]
        else:
            prefix.append(declaration)
    return '\n' + ''.join(prefix) + body if prefix else body


def splitSources(outputs, degree):
    result = dict(outputs)
    order = degree + 1
    for path, original in outputs.items():
        if not path.endswith('.hpp') or path.endswith(('types.hpp', 'ewald_common.hpp')):
            continue
        if not list(functions(original)):
            continue
        warp = path.endswith('l2l_warp.hpp')
        cpp = original
        adapters = []
        # Public containers share contiguous coefficient storage. Keep these
        # tiny forwarding templates in the header; compile the arithmetic once.
        for match, end, body in list(functions(original)):
            if 'typename Output' not in match[1]:
                continue
            signature = match[0][:-1].strip()
            name = re.search(r'void (\w+)\(', signature)[1]
            raw = signature.replace(', typename Output', '').replace(', typename Source', '')
            raw = re.sub(r'Output\s*&\s*output', 'Real* output', raw)
            raw = re.sub(r'Source\s+const\s*&\s*source', 'Real const* source', raw)
            parameters = signature[signature.index(name + '(') + len(name) + 1:signature.rfind(')')]
            # requires precedes CUDA_EXPORT for spherical operators.
            args = [re.search(r'(\w+)\s*$', p)[1] for p in splitArguments(parameters)]
            args = ['&output[0]' if a == 'output' else '&source[0]' if a == 'source' else a for a in args]
            adapter = signature + ' {\n    ' + name + '(' + ', '.join(args) + ');\n}'
            adapters.append((name, adapter))
            cpp = cpp.replace(signature + ' {', raw + ' {')
        # Generator emitters may put the opening brace on the signature itself.
        if adapters and 'typename Output' in cpp:
            raise RuntimeError('Unconverted spherical storage parameter in ' + path)
        header = []
        sources = {key: [] for key in ('scalar', 'simd')}
        audit = []
        position = 0
        for match, end, body in functions(cpp):
            prefix = cpp[position:match.start()]
            signature = match[0][:-1].strip()
            namespace = list(re.finditer(r'namespace ([\w:]+)\s*\{', cpp[:match.start()]))[-1][1]
            conversion = '::operator ' in signature
            name = None if conversion else re.search(r'(\w+)\(', signature.split('CUDA_EXPORT')[-1].split('__device__')[-1])[1]
            body = sphericalAccesses(path, name, body)
            header.append(prefix + ('' if conversion else
                '#if defined(COSMICTIGER_FMM_COUNTING_ONLY) || defined(COSMICTIGER_FMM_INSTRUMENTATION)\n' + signature + ';\n#endif\n'))
            audit.append(f'namespace {namespace} {{\n{signature} {{{body}}}\n}}\n')
            types = CPU_TYPES[:2] if warp else CPU_TYPES
            for real in types:
                category = 'scalar' if real in ('float', 'double') else 'simd'
                scalar = 'float' if 'float' in real else 'double'
                variants = (True, False) if name == 'executeL2LWarp' else (None,)
                for replicate in variants:
                    concrete = re.sub(r'^template\s*<[^>]+>\s*', 'template <>\n' if conversion else '', signature)
                    concrete = re.sub(r'requires\s*\([^)]*\)\s*', '', concrete)
                    concrete = re.sub(r'\bReal\b', real, concrete)
                    concrete = re.sub(r'\border\b', str(order), concrete)
                    target = f'Expansion<{real}>' if replicate else f'{real}*'
                    concrete = re.sub(r'\bTarget\b', target, concrete)
                    text = hoistConstants(signature + ' {' + body + '}', scalar)
                    concreteBody = text[text.index('{') + 1:-1]
                    concreteBody = re.sub(r'\bReal\b', real, concreteBody)
                    if category == 'scalar':
                        # Constants already have the destination type. Runtime
                        # integer coordinates and masks still need conversion.
                        concreteBody = re.sub(
                            rf'(?:math::)?broadcast<{scalar}>\(([^()]*)\)',
                            lambda m: m[1] if re.fullmatch(r'k\d+', m[1])
                            else f'static_cast<{scalar}>({m[1]})', concreteBody)
                    aliases = ''
                    if conversion:
                        concreteBody = re.sub(r'\border\b', str(order), concreteBody)
                    if replicate is not None:
                        aliases += f'    constexpr bool kReplicate = {str(replicate).lower()};\n'
                    if 'int degree' in match[1]:
                        concreteBody = re.sub(r'\bdegree\b', str(degree - 1 if path.endswith("spherical/p2m.hpp") else degree), concreteBody)
                    definition = concrete + ' {' + aliases + concreteBody + '}\n'
                    sources[category].append(f'namespace {namespace} {{\n{definition}}}\n')
                    declaration = concrete + ';\n'
                    header.append('#ifndef COSMICTIGER_FMM_COUNTING_ONLY\n' + ('#ifndef __CUDACC__\n' if category == 'simd' else '') + declaration + ('#endif\n' if category == 'simd' else '') + '#endif\n')
            # Place container adapters after the pointer overloads.
            for adapterName, adapter in adapters:
                if adapterName == name:
                    header.append(adapter + '\n')
            position = end
        header.append(cpp[position:])
        header = ''.join(header)
        schedule = ''
        if warp:
            pattern = r'static __device__ unsigned (?:int|char) const l2lWarp(?:Schedule|Targets)\d+\[[^;]+;'
            schedule = '\n'.join(re.findall(pattern, header))
            header = re.sub(pattern, '', header)
            header = header.replace('// divergent constant-memory accesses. Internal linkage needs no device RDC.',
                                    '// divergent constant-memory accesses. Tables live in the CUDA library source.')
        result[path] = header
        stem = path.removesuffix('.hpp').replace('/', '_')
        include = '#include <cosmictiger/gravity/kernels/' + path + '>\n'
        for category, definitions in sources.items():
            if not definitions:
                continue
            source = '// Generated by tools/generate_fmm.py. Do not edit.\n' + include
            if warp:
                source += 'namespace cosmictiger::gravity::cartesian {\n' + schedule + '\n}\n'
            source += ''.join(definitions)
            name = 'sources/' + stem + '_' + category
            if not warp:
                result[name + '.cpp'] = source
            if category == 'scalar':
                # Emit the complete body in both translation units. The CUDA
                # host pass uses declarations only and links the C++ library.
                result[name + '.cu'] = (source if warp else
                    '// Generated by tools/generate_fmm.py. Do not edit.\n' + include +
                    '#ifdef __CUDA_ARCH__\n' + ''.join(definitions) + '#endif\n')
        if not warp:
            result['instrumentation/' + stem + '.cpp'] = (
                '// Generated counting-scalar definitions. Tests only; not a library source.\n' +
                include + hoistConstants(''.join(audit)))
    return result


def executionSpace(cpp, path):
    """Emit host definitions or device definitions with matching declarations.

    CUDA's host pass must see the same ordinary declarations as a C++ caller;
    its device pass sees device declarations. The scalar .cu definitions are
    guarded accordingly, so only the .cpp supplies host symbols.
    """
    if path.endswith('.cu'):
        return cpp.replace('CUDA_EXPORT', '__device__')
    if path.endswith('.cpp'):
        return cpp.replace('CUDA_EXPORT ', '')
    qualifier = '\n#ifdef __CUDA_ARCH__\n__device__\n#endif\n'
    if path.endswith(('cartesian_tracefree_types.hpp', 'spherical_types.hpp')):
        # Storage accessors and constructors remain shared inline utilities.
        return cpp.replace('CUDA_EXPORT explicit operator', qualifier + 'explicit operator')
    return cpp.replace('CUDA_EXPORT', qualifier)
