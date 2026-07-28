#!/usr/bin/env node

import { access, copyFile, cp, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const client = path.join(dist, "client");
const inputs = [
  "index.html",
  "void.css",
  "void.js",
  "assets",
  "worker/index.js",
  ".openai/hosting.json",
];

for (const input of inputs) {
  await access(path.join(root, input));
}

await rm(dist, { recursive: true, force: true });
await mkdir(client, { recursive: true });
await mkdir(path.join(dist, "server"), { recursive: true });
await mkdir(path.join(dist, ".openai"), { recursive: true });

for (const file of ["index.html", "void.css", "void.js"]) {
  await copyFile(path.join(root, file), path.join(client, file));
}

await cp(path.join(root, "assets"), path.join(client, "assets"), {
  recursive: true,
});
await copyFile(
  path.join(root, "worker/index.js"),
  path.join(dist, "server/index.js"),
);
await copyFile(
  path.join(root, ".openai/hosting.json"),
  path.join(dist, ".openai/hosting.json"),
);

console.log("Prepared Sites artifact: dist/client + dist/server/index.js");
