#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""The abstract's numerical digits are prefixes of the stored logs.

P_infinity is not an enclosure. The digits before the ellipsis must begin
the forty-digit value in the continuum log, and the sentence must keep
the condition on the observed rate. The unstable-mode counts must be the
counts the stability log prints. Raising the last digit of P_infinity
must fail.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
TEX = os.path.join(ROOT, 'paper', 'collapse-without-rotation.tex')
CONT = os.path.join(ROOT, 'data', 'verify-continuum-limit.txt')
STAB = os.path.join(ROOT, 'data', 'verify-stability.txt')
PREFIX = '0.47736353369161202484'


def fail(msg):
    print(msg)
    print('FAIL')
    sys.exit(1)


def bump(text):
    return text[:-1] + str((int(text[-1]) + 1) % 10)


def main():
    tex = open(TEX, encoding='utf-8').read()
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\medskip', tex, re.S)
    if not abstract:
        fail('abstract not found')
    body = abstract.group(1)
    if ('$P_\\infty = %s\\ldots$' % PREFIX) not in body:
        fail('the abstract does not print %s with an ellipsis' % PREFIX)
    if 'if the observed exponential convergence persists' not in body:
        fail('the abstract dropped the condition on the convergence rate')
    if r'with $8$ and $57$ unstable modes' not in body:
        fail('the abstract no longer gives 8 and 57 unstable modes')
    if 'numerically, the two certified collapses' not in body:
        fail('the mode counts are no longer marked numerical')

    continuum = open(CONT, encoding='utf-8').read()
    stored = re.search(r'P_inf = ([0-9]+\.[0-9]+) to 40 digits', continuum)
    if not stored:
        fail('forty-digit P_inf line not found')
    value = stored.group(1)
    if not value.startswith(PREFIX):
        fail('stored %s does not begin with %s' % (value, PREFIX))
    if value.startswith(bump(PREFIX)):
        fail('the raised prefix still matched')

    stability = open(STAB, encoding='utf-8').read()
    if '16 shape modes, 8 unstable' not in stability or '114 shape modes, 57 unstable' not in stability:
        fail('the stability log does not record 8 and 57 unstable modes')

    print('P_inf %s begins the stored %s' % (PREFIX, value))
    print('8 and 57 unstable modes match data/verify-stability.txt')
    print('raising the last digit of P_inf is rejected')
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
