/**
 * The crowd — low-poly people on a ground plane.
 *
 * The one rule that matters: figures are ANONYMOUS AND UNDIFFERENTIATED.
 * Posture and gait are driven by the SHARED affect, so the whole crowd hunches
 * or straightens together. The moment a figure carries its own emotion we've
 * smuggled discrete categories back in — that's a bar chart with better art.
 *
 * Why a crowd at all: it's the only representation that holds ambivalence
 * honestly. Two feelings at once don't average to neutral the way a point in
 * a coordinate space does — both populations are simply present, in tension.
 *
 * Mapping:
 *   intensity -> how many, speed, stride, bounce, how upright
 *   effort    -> forward lean, hunched shoulders, cohesion, personal space
 *   movement  -> wander, head sway
 *   emphasis  -> an impulse that ripples outward from a point
 *
 * Six instanced meshes (head, torso, 2 arms, 2 legs) = 6 draw calls for the
 * whole crowd, whatever its size.
 */

import * as THREE from 'three';
import { store, trimEmphases, type Side } from '../state/emotion.js';

const MAX_AGENTS = 200;
const BASE_AGENTS = 26;

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
}

function makeAgent(): Agent {
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
  scene.fog = new THREE.FogExp2(0x0a0a0c, 0.028);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 120);
  camera.position.set(0, 7.5, 21);
  camera.lookAt(0, 1.1, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.15);
  key.position.set(5, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7fa8ff, 0.55);
  rim.position.set(-7, 3, -9);
  scene.add(rim);

  // Flat-shaded so the polygons read as polygons.
  const mat = new THREE.MeshStandardMaterial({
    roughness: 0.55,
    metalness: 0,
    flatShading: true,
    emissive: new THREE.Color(0x000000),
  });

  const parts = {
    head: new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.23, 0), mat, MAX_AGENTS),
    torso: new THREE.InstancedMesh(limbGeometry(0.46, 0.7, 0.26), mat, MAX_AGENTS),
    armL: new THREE.InstancedMesh(limbGeometry(0.14, 0.58, 0.14), mat, MAX_AGENTS),
    armR: new THREE.InstancedMesh(limbGeometry(0.14, 0.58, 0.14), mat, MAX_AGENTS),
    legL: new THREE.InstancedMesh(limbGeometry(0.17, 0.62, 0.17), mat, MAX_AGENTS),
    legR: new THREE.InstancedMesh(limbGeometry(0.17, 0.62, 0.17), mat, MAX_AGENTS),
  };
  for (const mesh of Object.values(parts)) {
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    scene.add(mesh);
  }

  const agents: Agent[] = [];
  const consumed = new Set<number>();
  const dummy = new THREE.Object3D();
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  const fieldColor = new THREE.Color(0xdcdce4);

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
    while (agents.length < want) agents.push(makeAgent());
    if (agents.length > want) agents.length = want;

    // Cohesion stays a mild ATTRACTION always — it's what keeps the crowd on
    // screen. Dispersal comes from local repulsion instead: inverted gravity
    // scales with distance, so it runs away and piles everyone on the edges.
    const cohesion = 0.00018 + effort * 0.0011;
    const personalSpace = 3.4 - effort * 1.9;
    const speed = 0.012 + intensity * 0.15;
    const jitter = 0.002 + intensity * 0.026;
    const turn = 0.02 + movement * 0.22;

    // Posture, shared across the whole crowd. This is where "angry vs happy"
    // actually lives — continuously, in the body, never as a label.
    const lean = effort * 0.42;
    const hunch = effort * 0.5;
    const stride = 0.25 + intensity * 0.85;
    const armAmp = 0.2 + intensity * 0.75;
    const upright = 1 - hunch * 0.45;

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
      const target = speed * a.bias;
      const current = a.vel.length() || 0.0001;
      a.vel.multiplyScalar(1 + (target / current - 1) * 0.06);
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
      const h = a.height;
      const bounce = Math.abs(Math.cos(a.phase)) * 0.05 * (0.4 + intensity);
      const hipY = (0.62 + bounce) * h;

      // legs — swing from the hip
      dummy.rotation.set(0, yaw, 0);
      for (const [mesh, dir, xo] of [
        [parts.legL, 1, -0.11],
        [parts.legR, -1, 0.11],
      ] as const) {
        dummy.position.set(
          a.pos.x + Math.cos(yaw) * xo,
          hipY,
          a.pos.z - Math.sin(yaw) * xo
        );
        dummy.rotation.set(swing * dir * stride, yaw, 0);
        dummy.scale.setScalar(h);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }

      // torso — leans forward with effort, shortens when hunched
      const shoulderY = hipY + 0.68 * h * upright;
      dummy.position.set(a.pos.x, shoulderY, a.pos.z);
      dummy.rotation.set(lean, yaw, 0);
      dummy.scale.set(h, h * upright, h);
      dummy.updateMatrix();
      parts.torso.setMatrixAt(i, dummy.matrix);

      // arms — counter-swung, drawn inward as the shoulders hunch
      for (const [mesh, dir, xo] of [
        [parts.armL, -1, -0.3],
        [parts.armR, 1, 0.3],
      ] as const) {
        const inset = xo * (1 - hunch * 0.3);
        dummy.position.set(
          a.pos.x + Math.cos(yaw) * inset,
          shoulderY - 0.04 * h,
          a.pos.z - Math.sin(yaw) * inset
        );
        dummy.rotation.set(swing * dir * armAmp + lean * 0.5, yaw, -dir * hunch * 0.3);
        dummy.scale.setScalar(h);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }

      // head — drops as the crowd hunches, sways with movement
      dummy.position.set(
        a.pos.x + Math.sin(lean) * 0.18 * h,
        shoulderY + 0.24 * h * upright,
        a.pos.z
      );
      dummy.rotation.set(lean * 0.7, yaw, Math.sin(a.phase * 0.5) * movement * 0.2);
      dummy.scale.setScalar(h);
      dummy.updateMatrix();
      parts.head.setMatrixAt(i, dummy.matrix);
    }

    for (let i = agents.length; i < MAX_AGENTS; i++) {
      for (const mesh of Object.values(parts)) mesh.setMatrixAt(i, hidden);
    }
    for (const mesh of Object.values(parts)) mesh.instanceMatrix.needsUpdate = true;

    // Colour is a property of the FIELD, never the individual — otherwise
    // we're back to "that one's anger" and the categories return.
    fieldColor.setHSL(
      0.58 - intensity * 0.5 - effort * 0.08,
      0.12 + intensity * 0.5,
      0.6 + intensity * 0.08
    );
    mat.color.copy(fieldColor);
    mat.emissive.copy(fieldColor).multiplyScalar(0.06 + intensity * 0.22);

    renderer.render(scene, camera);
  };

  raf = requestAnimationFrame(frame);

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      for (const mesh of Object.values(parts)) mesh.geometry.dispose();
      mat.dispose();
      renderer.dispose();
    },
    resize,
  };
}
