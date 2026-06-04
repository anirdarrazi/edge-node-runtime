#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { isDocPlaceholder, normalizeLine as normalizeTrailingWhitespace, normalizeText } from './text-utils.mjs';

const PAIRS = [['AGENTS.md', 'CLAUDE.md']];

const args = process.argv.slice(2);
const strictMode = args.includes('--strict');

function parseBoolArg(name, defaultValue = false) {
  return args.includes(`--${name}`) || defaultValue;
}

function normalizeLine(line, strict) {
  const trimmed = normalizeTrailingWhitespace(line).replace(/^\uFEFF/, '');
  if (!strict) {
    return trimmed;
  }
  return isDocPlaceholder(trimmed) ? '__DOC_PLACEHOLDER__' : trimmed;
}

function firstDiff(a, b, strict) {
  const left = a.split('\n');
  const right = b.split('\n');
  const maxLines = Math.max(left.length, right.length);

  for (let i = 0; i < maxLines; i += 1) {
    const leftLine = normalizeLine(left[i] || '', strict);
    const rightLine = normalizeLine(right[i] || '', strict);
    if (leftLine !== rightLine) {
      return i + 1;
    }
  }

  return 0;
}

export function runVerifyAgentClaudeSync({ root = process.cwd(), strict = strictMode } = {}) {
  let hasDrift = false;
  const flags = [];

  if (parseBoolArg('strict')) {
    strict = true;
  }

  for (const [leftName, rightName] of PAIRS) {
    const leftText = normalizeText(readFileSync(resolve(root, leftName), 'utf8'));
    const rightText = normalizeText(readFileSync(resolve(root, rightName), 'utf8'));

    if (leftText !== rightText || strict) {
      const diffLine = firstDiff(leftText, rightText, strict);
      if (diffLine > 0) {
        const leftLines = leftText.split('\n');
        const rightLines = rightText.split('\n');
        console.error(
          `[doc-sync] Drift detected (${strict ? 'strict' : 'normal'}) between ${leftName} and ${rightName}`,
        );
        console.error(`First diff at line ${diffLine}`);
        console.error(`${leftName}: ${leftLines[diffLine - 1] || '<EOF>'}`);
        console.error(`${rightName}: ${rightLines[diffLine - 1] || '<EOF>'}`);
        hasDrift = true;
        flags.push(leftName);
      }
    }
  }

  if (!flags.length) {
    console.log(
      `[doc-sync] AGENTS.md and CLAUDE.md are in sync for ${
        strict ? 'strict' : 'normal'
      } mode.`,
    );
    return;
  }

  if (hasDrift) {
    throw new Error('[doc-sync] AGENTS.md and CLAUDE.md drift check failed.');
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    runVerifyAgentClaudeSync();
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
