// Step 10.5: strip and re-compress 3DBAG tiles for the home page (see scripts/87_3dbag_subset.py).
// Usage: node strip.mjs PAIRS.json ['{"normals": false, "force": true}']   where PAIRS.json is [[in.glb, out.glb], ...]
// Each tile loses its 42-column attribute table (EXT_structural_metadata) and the feature-id extension; the
// per-building id stays as a plain vertex attribute, _BUILDING (about 1% of the size), so the page can animate
// buildings one by one. The geometry is re-encoded with Draco (positions 14 bit, normals 8 bit, ids 14 bit).
// With normals: false the normals go too (step 11.9): the walls and roofs are flat, so the page computes each
// face's normal from the surface itself, and Draco can then merge the corners the faces shared (about 45% smaller).
// force: true re-encodes tiles even when the output is newer than the input.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { draco, prune } from '@gltf-transform/functions';
import { MeshoptDecoder } from 'meshoptimizer';
import draco3d from 'draco3dgltf';
import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { dirname } from 'node:path';

await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'meshopt.decoder': MeshoptDecoder,
  'draco3d.encoder': await draco3d.createEncoderModule(),
  'draco3d.decoder': await draco3d.createDecoderModule(),
});

const pairs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const opts = { normals: true, force: false, ...JSON.parse(process.argv[3] || '{}') };
let done = 0, skipped = 0, inBytes = 0, outBytes = 0;
for (const [src, dst] of pairs) {
  if (!opts.force && existsSync(dst) && statSync(dst).mtimeMs > statSync(src).mtimeMs) { skipped++; continue; }
  const doc = await io.read(src);
  for (const mesh of doc.getRoot().listMeshes())
    for (const prim of mesh.listPrimitives())
      for (const name of prim.listSemantics()) {
        if (!name.startsWith('_FEATURE_ID')) continue;
        const ids = prim.getAttribute(name);
        prim.setAttribute(name, null);
        if (name === '_FEATURE_ID_0') prim.setAttribute('_BUILDING', ids);
      }
  if (!opts.normals)
    for (const mesh of doc.getRoot().listMeshes())
      for (const prim of mesh.listPrimitives()) prim.setAttribute('NORMAL', null);
  for (const ext of doc.getRoot().listExtensionsUsed())
    if (['EXT_mesh_features', 'EXT_structural_metadata', 'EXT_meshopt_compression'].includes(ext.extensionName)) ext.dispose();
  await doc.transform(prune({ keepAttributes: true }), draco({ method: 'edgebreaker', quantizePosition: 14, quantizeNormal: 8, quantizeGeneric: 14 }));
  const out = await io.writeBinary(doc);
  mkdirSync(dirname(dst), { recursive: true });
  writeFileSync(dst, out);
  done++; inBytes += statSync(src).size; outBytes += out.length;
  if (done % 25 === 0) console.log(`stripped ${done}/${pairs.length - skipped}`);
}
console.log(JSON.stringify({ stripped: done, skipped, in_mb: +(inBytes / 1e6).toFixed(1), out_mb: +(outBytes / 1e6).toFixed(1) }));
