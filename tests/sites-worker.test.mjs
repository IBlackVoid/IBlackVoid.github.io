import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import worker from "../worker/index.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function filesBelow(directory) {
  const files = [];

  async function walk(relative) {
    const entries = await readdir(path.join(directory, relative), {
      withFileTypes: true,
    });
    for (const entry of entries) {
      const child = path.join(relative, entry.name);
      if (entry.isDirectory()) {
        await walk(child);
      } else {
        files.push(child);
      }
    }
  }

  await walk("");
  return files.sort();
}

test("worker delegates every request to the static asset binding", async () => {
  const request = new Request("https://portfolio.test/void.js");
  const expected = new Response("asset", { status: 200 });
  let received;

  const actual = await worker.fetch(request, {
    ASSETS: {
      fetch(input) {
        received = input;
        return expected;
      },
    },
  });

  assert.equal(received, request);
  assert.equal(actual, expected);
});

test("Sites artifact preserves the public source byte-for-byte", async () => {
  for (const file of ["index.html", "void.css", "void.js"]) {
    const source = await readFile(path.join(root, file));
    const staged = await readFile(path.join(root, "dist/client", file));
    assert.deepEqual(staged, source, file);
  }

  const sourceAssets = path.join(root, "assets");
  const stagedAssets = path.join(root, "dist/client/assets");
  assert.deepEqual(await filesBelow(stagedAssets), await filesBelow(sourceAssets));

  for (const file of await filesBelow(sourceAssets)) {
    const source = await readFile(path.join(sourceAssets, file));
    const staged = await readFile(path.join(stagedAssets, file));
    assert.deepEqual(staged, source, file);
  }
});

test("Sites artifact contains its worker and hosting manifest", async () => {
  const workerSource = await readFile(path.join(root, "worker/index.js"));
  const stagedWorker = await readFile(path.join(root, "dist/server/index.js"));
  assert.deepEqual(stagedWorker, workerSource);

  const manifest = await readFile(path.join(root, ".openai/hosting.json"));
  const stagedManifest = await readFile(
    path.join(root, "dist/.openai/hosting.json"),
  );
  assert.deepEqual(stagedManifest, manifest);
});
