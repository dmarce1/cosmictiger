# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
"""Emit declaration-only operator headers and concrete CPU/CUDA definitions.

The optional arithmetic instrumentation sources contain generic definitions for
counting-scalar tests. Production targets compile only the concrete sources.
"""
import re
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
            args = [re.search(r'(\w+)\s*$', p)[1] for p in parameters.split(',')]
            args = ['&output[0]' if a == 'output' else '&source[0]' if a == 'source' else a for a in args]
            adapter = signature + ' {\n    ' + name + '<degree, Real>(' + ', '.join(args) + ');\n}'
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
            header.append(prefix + ('' if conversion else signature + ';\n'))
            audit.append(f'namespace {namespace} {{\n{signature} {{{body}}}\n}}\n')
            types = CPU_TYPES[:2] if warp else CPU_TYPES
            for real in types:
                category = 'scalar' if real in ('float', 'double') else 'simd'
                scalar = 'float' if 'float' in real else 'double'
                variants = (True, False) if name == 'executeL2LWarp' else (None,)
                for replicate in variants:
                    concrete = re.sub(r'^template\s*<[^>]+>\s*', 'template <>\n', signature)
                    concrete = re.sub(r'requires\s*\([^)]*\)\s*', '', concrete)
                    concrete = re.sub(r'\bReal\b', real, concrete)
                    concrete = re.sub(r'\border\b', str(order), concrete)
                    target = f'Expansion<{real}, {order}>' if replicate else f'{real}*'
                    concrete = re.sub(r'\bTarget\b', target, concrete)
                    if not conversion:
                        arguments = f'{degree - 1 if path.endswith("spherical/p2m.hpp") else degree}, {real}' if 'int degree' in match[1] else real
                        if replicate is not None:
                            arguments = f'{str(replicate).lower()}, {real}, {target}'
                        concrete = concrete.replace(name + '(', name + '<' + arguments + '>(')
                    text = hoistConstants(signature + ' {' + body + '}', scalar)
                    concreteBody = text[text.index('{') + 1:-1]
                    aliases = f'\n    using Real = {real};\n' if re.search(r'\bReal\b', concreteBody) else ''
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
            # Place container adapters after the pointer overload and its specializations.
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
                source += 'namespace cosmictiger::gravity::cartesian::detail {\n' + schedule + '\n}\n'
            source += ''.join(definitions)
            name = 'sources/' + stem + '_' + category
            if not warp:
                result[name + '.cpp'] = source
            if category == 'scalar':
                # Scalar host and device definitions are compiled together by
                # NVCC in CUDA builds. C++ builds compile the matching .cpp only.
                result[name + '.cu'] = source if warp else f'// Generated by tools/generate_fmm.py. Do not edit.\n#include "{stem}_scalar.cpp"\n'
        if not warp:
            result['instrumentation/' + stem + '.cpp'] = (
                '// Generated counting-scalar definitions. Tests only; not a library source.\n' +
                include + hoistConstants(''.join(audit)))
    return result
