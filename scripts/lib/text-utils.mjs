#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { basename, extname } from 'node:path';

const CRLF_PATTERN = /\r\n?/g;

export const BINARY_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.pdf',
  '.zip',
  '.ico',
  '.ttf',
  '.otf',
  '.eot',
  '.woff',
  '.woff2',
  '.bin',
  '.exe',
  '.dll',
  '.so',
  '.dylib',
  '.class',
  '.jar',
  '.obj',
  '.pyc',
  '.lock',
  '.wasm',
]);

export const TEXT_EXTENSIONS = new Set([
  '.md',
  '.json',
  '.js',
  '.mjs',
  '.ts',
  '.py',
  '.yml',
  '.yaml',
  '.toml',
  '.sh',
  '.ps1',
  '.tsx',
  '.jsx',
  '.css',
  '.html',
  '.xml',
  '.txt',
  '.ini',
  '.cfg',
  '.conf',
  '.env',
  '.cmd',
  '.bat',
  '.psm1',
]);

export const DOC_PLACEHOLDER_PATTERNS = [
  /^\{\{\s*.+?\s*\}\}$/,
  /^\[\[[^\\]]+\]\]$/,
  /^<!--\s*placeholder:\s*.+?\s*-->$/i,
  /^<!--\s*architecture:\s*.+?\s*-->$/i,
];

export function normalizeText(value) {
  return value.replace(CRLF_PATTERN, '\n').trimEnd().replace(/^\uFEFF/, '');
}

export function normalizeLine(line) {
  return line.replace(/[ \t]+$/, '');
}

export function trimLine(line) {
  return line.trim();
}

export function isDocPlaceholder(line) {
  return DOC_PLACEHOLDER_PATTERNS.some((pattern) => pattern.test(line));
}

export function isTextLikePath(filePath) {
  const extension = extname(filePath).toLowerCase();
  if (TEXT_EXTENSIONS.has(extension)) {
    return true;
  }

  const base = basename(filePath);
  if (base === 'Dockerfile' || base === 'Makefile' || base === 'Dockerfile.service' || base === 'Dockerfile.single') {
    return true;
  }

  return false;
}

export function isBinaryPath(filePath) {
  return BINARY_EXTENSIONS.has(extname(filePath).toLowerCase());
}

export function hasCrLfInBuffer(buffer) {
  for (let index = 0; index < buffer.length - 1; index += 1) {
    if (buffer[index] === 0x0d && buffer[index + 1] === 0x0a) {
      return true;
    }
  }

  return false;
}

export function hasCrLfLineEndings(filePath) {
  const buffer = readFileSync(filePath);
  return hasCrLfInBuffer(buffer);
}
