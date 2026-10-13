// Read a glb back (with meshopt and Draco decoders) and print its world bounds, vertex and triangle counts.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { getBounds } from '@gltf-transform/functions';
import { MeshoptDecoder } from 'meshoptimizer';
import draco3d from 'draco3dgltf';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'draco3d.decoder': await draco3d.createDecoderModule() });
for (const f of process.argv.slice(2)) {
  const doc = await io.read(f);
  const bb = getBounds(doc.getRoot().listScenes()[0]);
  let v = 0, t = 0;
  for (const m of doc.getRoot().listMeshes()) for (const p of m.listPrimitives()) { v += p.getAttribute('POSITION').getCount(); t += (p.getIndices()?.getCount() ?? 0) / 3; }
  const pos = doc.getRoot().listMeshes()[0].listPrimitives()[0].getAttribute('POSITION');
  console.log(f.split('/').pop().padEnd(28), 'verts', v, 'tris', t, 'pos', pos.getComponentType(), pos.getNormalized(), 'bounds', bb.min.map(x => x.toFixed(1)).join(','), '|', bb.max.map(x => x.toFixed(1)).join(','));
}
