/**
 * The crowd — low-poly people on a ground plane.
 *
 * Figures belong to ARCHETYPES (see archetypes.ts), but nothing here ever
 * classifies the person. Affect drives a soft distribution and what you read
 * is the shifting MIX. Individuals are discrete; the crowd is a gradient.
 *
 * Posture is layered: a crowd-wide component from affect, plus each
 * archetype's own bias. So everyone hunches together when effort rises, and
 * the heavy ones are still the most folded.
 *
 * Mapping:
 *   intensity -> how many, speed, stride, bounce, and the bright/sharp mix
 *   effort    -> lean, hunch, cohesion, personal space, and the sharp mix
 *   movement  -> wander, head sway, and the skittish mix
 *   emphasis  -> an impulse that ripples outward from a point
 *
 * Seven instanced meshes for the whole crowd, whatever its size.
 */

import * as THREE from 'three';
import { store, trimEmphases, type Side } from '../state/emotion.js';
import { ARCHETYPES, distribution } from './archetypes.js';

const MAX_AGENTS = 200;
const BASE_AGENTS = 30;

/** Ground-plane bounds. Agents wrap inside this box. */
const BOUNDS = { x: 34, z: 26 };

/** How far the cursor's repulsion reaches, in world units. */
const CURSOR_RADIUS = 6;

interface Agent {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  /** Gait phase, so nobody walks in lockstep. */
  phase: number;
  /** Per-agent variation so the crowd doesn't look mechanical. */
  bias: number;
  height: number;
  archetype: number;
  /** Eases 0..1 on spawn and on retype, so the mix slides instead of popping. */
  blend: number;
}

function makeAgent(archetype: number): Agent {
  return {
    pos: new THREE.Vector3(
      (Math.random() - 0.5) * BOUNDS.x,
      0,
      (Math.random() - 0.5) * BOUNDS.z
    ),
    vel: new THREE.Vector3((Math.random() - 0.5) * 0.1, 0, (Math.random() - 0.5) * 0.1),
    phase: Math.random() * Math.PI * 2,
    bias: 0.75 + Math.random() * 0.5,
    height: 0.9 + Math.random() * 0.2,
    archetype,
    blend: 0,
  };
}

/** Box pivoted at its TOP so it can swing from a joint. */
function limbGeometry(w: number, len: number, d: number): THREE.BoxGeometry {
  const g = new THREE.BoxGeometry(w, len, d);
  g.translate(0, -len / 2, 0);
  return g;
}

export interface CrowdHandle {
  stop(): void;
  resize(): void;
}

export function startCrowd(canvas: HTMLCanvasElement, side: Side): CrowdHandle {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0a0a0c, 0.026);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 120);
  camera.position.set(0, 7, 20);
  camera.lookAt(0, 1.1, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  const key = new THREE.DirectionalLight(0xffffff, 1.05);
  key.position.set(5, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7fa8ff, 0.5);
  rim.position.set(-7, 3, -9);
  scene.add(rim);

  // Flat-shaded so the polygons read as polygons.
  const mat = new THREE.MeshStandardMaterial({
    roughness: 0.55,
    metalness: 0,
    flatShading: true,
  });

  const parts = {
    head: new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.26, 0), mat, MAX_AGENTS),
    crest: new THREE.InstancedMesh(new THREE.ConeGeometry(0.24, 0.42, 5), mat, MAX_AGENTS),
    torso: new THREE.InstancedMesh(limbGeometry(0.46, 0.7, 0.28), mat, MAX_AGENTS),
    armL: new THREE.InstancedMesh(limbGeometry(0.14, 0.58, 0.14), mat, MAX_AGENTS),
    armR: new THREE.InstancedMesh(limbGeometry(0.14, 0.58, 0.14), mat, MAX_AGENTS),
    legL: new THREE.InstancedMesh(limbGeometry(0.17, 0.62, 0.17), mat, MAX_AGENTS),
    legR: new THREE.InstancedMesh(limbGeometry(0.17, 0.62, 0.17), mat, MAX_AGENTS),
  };
  const meshes = Object.values(parts);
  for (const mesh of meshes) {
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    scene.add(mesh);
  }

  const agents: Agent[] = [];
  const consumed = new Set<number>();
  const dummy = new THREE.Object3D();
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  const tint = new THREE.Color();
  const shade = new THREE.Color();

  // Cursor as a repulsion field — agents scatter out of its way. Projected
  // onto the ground plane so it pushes in world space, not screen space.
  const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const cursor = new THREE.Vector3();
  let cursorActive = false;

  const onPointerMove = (e: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    cursorActive = raycaster.ray.intersectPlane(groundPlane, cursor) !== null;
  };
  const onPointerLeave = () => {
    cursorActive = false;
  };
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerleave', onPointerLeave);

  let raf = 0;
  let stopped = false;

  const resize = () => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();

  const frame = () => {
    if (stopped) return;
    raf = requestAnimationFrame(frame);

    const { affect, emphases } = store[side];
    trimEmphases();

    // Normalise the z-scored channels into 0..1 drive values.
    const intensity = Math.max(0, Math.min(1, (affect.intensity + 1) / 3));
    const effort = Math.max(0, Math.min(1, (affect.effort + 1) / 3));
    const movement = Math.max(0, Math.min(1, affect.movement / 2));

    // Quantity is itself an encoding: feeling more means more of them.
    const want = Math.round(BASE_AGENTS + intensity * (MAX_AGENTS - BASE_AGENTS));

    // Retype toward the target mix a few at a time. Gradual on purpose — the
    // crowd should slide between mixtures, never cut.
    const target = distribution(want, intensity, effort, movement);
    const counts = new Array(ARCHETYPES.length).fill(0);
    for (const a of agents) counts[a.archetype]++;

    while (agents.length > want) {
      agents.pop();
    }
    while (agents.length < want) {
      let pick = 0;
      let worst = -Infinity;
      for (let t = 0; t < ARCHETYPES.length; t++) {
        const deficit = target[t] - counts[t];
        if (deficit > worst) {
          worst = deficit;
          pick = t;
        }
      }
      counts[pick]++;
      agents.push(makeAgent(pick));
    }
    for (let n = 0; n < 2; n++) {
      let over = -1;
      let under = -1;
      let mostOver = 0.9;
      let mostUnder = 0.9;
      for (let t = 0; t < ARCHETYPES.length; t++) {
        if (counts[t] - target[t] > mostOver) {
          mostOver = counts[t] - target[t];
          over = t;
        }
        if (target[t] - counts[t] > mostUnder) {
          mostUnder = target[t] - counts[t];
          under = t;
        }
      }
      if (over < 0 || under < 0) break;
      const victim = agents.find((a) => a.archetype === over);
      if (!victim) break;
      victim.archetype = under;
      victim.blend = 0;
      counts[over]--;
      counts[under]++;
    }

    // Cohesion stays a mild ATTRACTION always — it's what keeps the crowd on
    // screen. Dispersal comes from local repulsion instead: inverted gravity
    // scales with distance, so it runs away and piles everyone on the edges.
    const cohesion = 0.00018 + effort * 0.0011;
    const personalSpace = 3.4 - effort * 1.9;
    const speed = 0.012 + intensity * 0.15;
    const jitter = 0.002 + intensity * 0.026;
    const turn = 0.02 + movement * 0.22;

    // Crowd-wide posture. Archetype bias is layered on top per agent.
    const lean = effort * 0.38;
    const hunch = effort * 0.45;
    const stride = 0.3 + intensity * 0.8;
    const armAmp = 0.22 + intensity * 0.7;

    // New emphasis events become an outward impulse — a syllable you leaned
    // on visibly disturbs the room.
    for (const e of emphases) {
      if (consumed.has(e.id)) continue;
      consumed.add(e.id);
      const ox = (Math.random() - 0.5) * BOUNDS.x;
      const oz = (Math.random() - 0.5) * BOUNDS.z;
      for (const a of agents) {
        const dx = a.pos.x - ox;
        const dz = a.pos.z - oz;
        const dist = Math.hypot(dx, dz) || 1;
        if (dist > 15) continue;
        const push = (e.weight * 0.45) / dist;
        a.vel.x += (dx / dist) * push;
        a.vel.z += (dz / dist) * push;
      }
    }
    if (consumed.size > 512) consumed.clear();

    for (let i = 0; i < agents.length; i++) {
      const a = agents[i];
      const arch = ARCHETYPES[a.archetype];
      a.blend = Math.min(1, a.blend + 0.05);

      // wander
      const heading = Math.atan2(a.vel.x, a.vel.z) + (Math.random() - 0.5) * turn;
      const planar = Math.hypot(a.vel.x, a.vel.z) || 0.0001;
      a.vel.x = Math.sin(heading) * planar;
      a.vel.z = Math.cos(heading) * planar;

      a.vel.x += (Math.random() - 0.5) * jitter;
      a.vel.z += (Math.random() - 0.5) * jitter;

      a.vel.x -= a.pos.x * cohesion;
      a.vel.z -= a.pos.z * cohesion;

      // Get out of the cursor's way.
      if (cursorActive) {
        const dx = a.pos.x - cursor.x;
        const dz = a.pos.z - cursor.z;
        const d2 = dx * dx + dz * dz;
        if (d2 > 0.01 && d2 < CURSOR_RADIUS * CURSOR_RADIUS) {
          const d = Math.sqrt(d2);
          const push = (1 - d / CURSOR_RADIUS) ** 2 * 0.09;
          a.vel.x += (dx / d) * push;
          a.vel.z += (dz / d) * push;
        }
      }

      // Personal space. Sampled rather than all-pairs — visually identical,
      // and keeps this O(n) instead of O(n^2).
      for (let s = 0; s < 6; s++) {
        const other = agents[(Math.random() * agents.length) | 0];
        if (other === a) continue;
        const dx = a.pos.x - other.pos.x;
        const dz = a.pos.z - other.pos.z;
        const d2 = dx * dx + dz * dz;
        if (d2 > 0.001 && d2 < personalSpace * personalSpace) {
          const d = Math.sqrt(d2);
          const push = ((personalSpace - d) / personalSpace) * 0.011;
          a.vel.x += (dx / d) * push;
          a.vel.z += (dz / d) * push;
        }
      }

      // Damp toward the speed this affect implies rather than clamping —
      // acceleration and settling both read as feeling.
      const targetSpeed = speed * a.bias * arch.strideBias;
      const current = a.vel.length() || 0.0001;
      a.vel.multiplyScalar(1 + (targetSpeed / current - 1) * 0.06);
      a.pos.add(a.vel);

      // wrap
      const hx = BOUNDS.x / 2;
      const hz = BOUNDS.z / 2;
      if (a.pos.x < -hx) a.pos.x = hx;
      if (a.pos.x > hx) a.pos.x = -hx;
      if (a.pos.z < -hz) a.pos.z = hz;
      if (a.pos.z > hz) a.pos.z = -hz;

      a.phase += 0.06 + current * 3.4;

      const yaw = Math.atan2(a.vel.x, a.vel.z);
      const swing = Math.sin(a.phase);
      const h = a.height * arch.heightBias * (0.5 + a.blend * 0.5);
      const limb = arch.limbLength;
      const myLean = lean + arch.leanBias;
      const myHunch = Math.max(0, Math.min(1, hunch + arch.hunchBias));
      const upright = 1 - myHunch * 0.45;
      const bounce =
        Math.abs(Math.cos(a.phase)) * 0.05 * (0.4 + intensity) * arch.bounceBias;
      const hipY = (0.62 * limb + bounce) * h;

      tint.setHSL(arch.hue, arch.sat, arch.light);

      // legs — swing from the hip
      for (const [mesh, dir, xo] of [
        [parts.legL, 1, -0.11],
        [parts.legR, -1, 0.11],
      ] as const) {
        dummy.position.set(
          a.pos.x + Math.cos(yaw) * xo,
          hipY,
          a.pos.z - Math.sin(yaw) * xo
        );
        dummy.rotation.set(swing * dir * stride * arch.strideBias, yaw, 0);
        dummy.scale.set(h * arch.girth, h * limb, h * arch.girth);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        shade.copy(tint).multiplyScalar(0.7);
        mesh.setColorAt(i, shade);
      }

      // torso — leans forward with effort, shortens when hunched
      const shoulderY = hipY + 0.68 * h * upright;
      dummy.position.set(a.pos.x, shoulderY, a.pos.z);
      dummy.rotation.set(myLean, yaw, 0);
      dummy.scale.set(h * arch.girth, h * upright, h * arch.girth);
      dummy.updateMatrix();
      parts.torso.setMatrixAt(i, dummy.matrix);
      parts.torso.setColorAt(i, tint);

      // arms — counter-swung, drawn inward as the shoulders hunch
      for (const [mesh, dir, xo] of [
        [parts.armL, -1, -0.3],
        [parts.armR, 1, 0.3],
      ] as const) {
        const inset = xo * arch.girth * (1 - myHunch * 0.3);
        dummy.position.set(
          a.pos.x + Math.cos(yaw) * inset,
          shoulderY - 0.04 * h,
          a.pos.z - Math.sin(yaw) * inset
        );
        dummy.rotation.set(
          swing * dir * armAmp * arch.strideBias + myLean * 0.5,
          yaw,
          -dir * myHunch * 0.3
        );
        dummy.scale.set(h, h * limb, h);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        shade.copy(tint).multiplyScalar(0.85);
        mesh.setColorAt(i, shade);
      }

      // head — drops as the crowd hunches, sways with movement
      const headY = shoulderY + 0.26 * h * upright * arch.headScale;
      dummy.position.set(
        a.pos.x + Math.sin(myLean) * 0.2 * h,
        headY,
        a.pos.z + Math.cos(yaw) * 0
      );
      dummy.rotation.set(myLean * 0.7, yaw, Math.sin(a.phase * 0.5) * movement * 0.2);
      dummy.scale.setScalar(h * arch.headScale);
      dummy.updateMatrix();
      parts.head.setMatrixAt(i, dummy.matrix);
      shade.copy(tint).multiplyScalar(1.12);
      parts.head.setColorAt(i, shade);

      // crest — pure silhouette. Scaled to nothing for archetypes without one.
      dummy.position.y = headY + 0.2 * h * arch.headScale;
      dummy.scale.setScalar(h * arch.crest);
      dummy.updateMatrix();
      parts.crest.setMatrixAt(i, dummy.matrix);
      parts.crest.setColorAt(i, shade);
    }

    for (let i = agents.length; i < MAX_AGENTS; i++) {
      for (const mesh of meshes) mesh.setMatrixAt(i, hidden);
    }
    for (const mesh of meshes) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    renderer.render(scene, camera);
  };

  raf = requestAnimationFrame(frame);

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      for (const mesh of meshes) mesh.geometry.dispose();
      mat.dispose();
      renderer.dispose();
    },
    resize,
  };
}
