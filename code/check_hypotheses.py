#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Hypothesis ledger, the same gate as the Hodgkin-Huxley pulse paper.

code/hypotheses.json names each claim boxed, proved, numerical, cited or
unread. While any item is unread, the manuscript may not say "the first
proof", "for the first time", "has not been proved", or "no earlier"
unless that last one still sits next to a limit the paper already uses:
"within them", "we have not found", "as far as we could find", "claim no
priority", or "no claim to be first".

A copy of the manuscript with "for the first time" inserted must be
refused. Nothing here is written back to the files.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
HYPO = os.path.join(ROOT, 'code', 'hypotheses.json')
PAPER = os.path.join(ROOT, 'paper')
PHRASES = ('no earlier', 'the first proof that', 'the first proof of', 'has not been proved', 'for the first time')
LIMITS = (
    'within them',
    'we have not found',
    'as far as we could find',
    'claim no priority',
    'no claim to be first',
)


def manuscript():
    chunks = []
    for name in sorted(os.listdir(PAPER)):
        if name.endswith(('.tex', '.md')):
            with open(os.path.join(PAPER, name), encoding='utf-8') as handle:
                chunks.append(handle.read())
    if not chunks:
        raise SystemExit('no manuscript under paper/')
    return re.sub(r'\s+', ' ', '\n'.join(chunks))


def blocking(text, unread):
    found = [phrase for phrase in PHRASES if phrase in text]
    limited = any(limit in text for limit in LIMITS)
    blocked = []
    for phrase in found:
        if phrase == 'no earlier' and limited:
            continue
        if unread:
            blocked.append(phrase)
    return found, blocked, limited


def certificates(hypotheses):
    missing = []
    for item in hypotheses:
        raw = item.get('certificate') or ''
        for rel in [part.strip() for part in raw.split(',') if part.strip()]:
            if not os.path.isfile(os.path.join(ROOT, rel)):
                missing.append('%s -> %s' % (item.get('id'), rel))
    return missing


def main():
    hypotheses = json.load(open(HYPO, encoding='utf-8'))
    missing = certificates(hypotheses)
    if missing:
        print('certificate file missing')
        for line in missing:
            print(line)
        return 1
    text = manuscript()
    unread = [item['id'] for item in hypotheses if item.get('status') == 'unread']
    counts = {}
    for item in hypotheses:
        status = item.get('status', '')
        counts[status] = counts.get(status, 0) + 1
    found, blocked, limited = blocking(text, unread)

    planted = text + ' for the first time '
    _, planted_blocked, _ = blocking(planted, ['planted'] if not unread else unread)
    if 'for the first time' not in planted_blocked:
        print('the planted phrase "for the first time" was not refused')
        return 1

    if blocked:
        for phrase in blocked:
            print(phrase)
        print('unread: ' + ', '.join(unread))
        return 1
    if not found:
        print('no such phrase occurs')
    else:
        print('priority phrase found: ' + '; '.join(found))
        if limited:
            print('the earlier-work sentence keeps its limit')
    order = ('boxed', 'proved', 'cited', 'numerical', 'unread')
    print(' '.join('%s %d' % (name, counts.get(name, 0)) for name in order))
    if unread:
        print('unread: ' + ', '.join(unread))
    print('planted "for the first time" refused')
    return 0


if __name__ == '__main__':
    sys.exit(main())
