#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Quote check for the numbers this paper reprints from the companion.

paper/collapse-without-rotation.tex cites the companion for the minima
0.7978967838…, 0.7448144569… and 0.7136801485…, and for eleven vortices at
alpha = 2 and sixty SQG vortices collapsing without rotation, in families
of dimension 9 and 58. Those enclosures are copies of the minimal-winding logs, kept in
data/minimal-winding so this archive can run the check. They are not a
new computation. The two-arm value at 603 vortices is labelled numerical.

This program only reads the manuscript and those logs. It does not import
a proof program and does not treat a binary64 figure as an enclosure.

Negative control, in memory only: the last digit of the first minimum is
increased by one.
"""
import os
import re
import sys
from decimal import Decimal, getcontext

getcontext().prec = 40

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, '..', 'paper', 'collapse-without-rotation.tex')
DATA = os.path.join(HERE, '..', 'data', 'minimal-winding')
COLLAPSE = os.path.join(DATA, 'certify-collapses-2026-09-25.txt')
SQG = os.path.join(DATA, 'certify-sqg60-2026-09-25.txt')
MANY = os.path.join(DATA, 'verify_many_vortices.txt')


def fail(printed, stored, reason=None):
    print('FAIL')
    if reason:
        print(reason)
    print('manuscript: %s' % printed)
    print('certificate: %s' % stored)
    sys.exit(1)


def chop_ok(lo, hi, prefix):
    places = len(prefix.split('.')[1])
    start = Decimal(prefix)
    step = Decimal(1).scaleb(-places)
    return start <= lo and hi < start + step


def bump(prefix):
    whole, dot, frac = prefix.partition('.')
    digits = list(frac)
    digits[-1] = str((int(digits[-1]) + 1) % 10)
    return whole + '.' + ''.join(digits)


def main():
    with open(TEX, encoding='utf-8') as handle:
        tex = handle.read()
    with open(COLLAPSE, encoding='utf-8') as handle:
        log = handle.read()
    with open(SQG, encoding='utf-8') as handle:
        sqg = handle.read()
    with open(MANY, encoding='utf-8') as handle:
        many = handle.read()

    found = re.search(
        r'strict local minima \$([0-9]+\.[0-9]+)\\ldots\$, '
        r'\$([0-9]+\.[0-9]+)\\ldots\$ and \$([0-9]+\.[0-9]+)\\ldots\$',
        tex)
    if not found:
        fail('three minima', '(phrase not found)')
    labels = ('P_4', 'P_5', 'P_6')
    for label, prefix in zip(labels, found.groups()):
        match = re.search(
            label + r' in \[([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+)\]', log)
        if not match:
            fail(prefix, '(interval not found)')
        lo, hi = Decimal(match.group(1)), Decimal(match.group(2))
        if not chop_ok(lo, hi, prefix):
            fail(prefix, match.group(0), 'the companion interval does not chop to these digits')
        if not (hi < Decimal(3).sqrt() / 2):
            fail(prefix, match.group(0), 'not below sqrt(3)/2')
        print('manuscript: %s' % prefix)
        print('certificate: %s' % match.group(0))
        print('file: data/minimal-winding/certify-collapses-2026-09-25.txt')

    if not re.search(r'eleven vortices at \$\\alpha = 2\$ and sixty SQG vortices collapse without rotation', tex):
        fail('without rotation', '(phrase not found)')
    if 'dimension $9$ and $58$' not in tex:
        fail('dimensions 9 and 58', '(phrase not found)')
    if not re.search(
            r'OK\s+alpha = 2, N = 11: all 11 kappa_j agree, 2 pi kappa = -1',
            log):
        fail('eleven vortices, kappa real', '(not in the companion log)')
    if '9-parameter family' not in log:
        fail('dimension 9', '(family not in the companion log)')
    if 'kappa real' not in sqg:
        fail('sixty SQG vortices, kappa real', '(not in the companion log)')
    if '58-parameter family' not in sqg:
        fail('dimension 58', '(family not in the companion log)')
    print('manuscript: eleven vortices at alpha = 2, kappa real, dimension 9')
    print('certificate: 2 pi kappa = -1; 9-parameter family')
    print('manuscript: sixty SQG vortices, kappa real, dimension 58')
    print('certificate: kappa real; 58-parameter family')

    numerical = re.search(
        r'Numerically, a two-arm family of Euler minimizers reaches \$P = ([0-9]+\.[0-9]+)\\ldots\$ at \$N = 603\$',
        tex)
    if not numerical:
        fail('N = 603', '(phrase not found)')
    prefix = numerical.group(1)
    row = re.search(r'N = 603.*P = ([0-9]+\.[0-9]+)', many)
    if row is None or not row.group(1).startswith(prefix):
        fail(prefix, row.group(1) if row else '(not found)')
    print('manuscript: %s at N = 603' % prefix)
    print('certificate: %s' % row.group(1))
    print('numerical, not an enclosure')

    moved = bump(found.group(1))
    match = re.search(r'P_4 in \[([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+)\]', log)
    lo, hi = Decimal(match.group(1)), Decimal(match.group(2))
    if chop_ok(lo, hi, moved):
        fail(moved, match.group(0), 'negative control still matched')
    print('negative control: last digit of %s mismatches' % found.group(1))
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
