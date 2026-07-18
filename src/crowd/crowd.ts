/**
 * The crowd.
 *
 * Nine emotions, each with its own body, gait and behaviour. Affect drives a
 * soft DISTRIBUTION over them — never a classification — so what you read is
 * the shifting mix. Individuals are discrete; the crowd is a gradient. Two
 * feelings at once show up as two populations, which no single label can do.
 *
 * Behaviours are what sell it: angry ones seek others out and swing at them,
 * desperate ones bolt around sweating, happy ones skip, afraid ones scatter.
 *
 * Fourteen instanced meshes for the whole crowd, whatever its size.
 */

import * as THREE from 'three';
import { store, trimEmphases, type Side } from '../state/emotion.js';
import { ARCHETYPES, distribution } from './archetypes.js';
import { pose, CADENCE } from './gait.js';
import { PARTICLES, particleGeometries, type ParticleShape } from './particles.js';

const MAX_AGENTS = 160;
const BASE_AGENTS = 26;

/**
 * Where agents may walk, derived from the camera frustum at resize rather than
 * hardcoded — a tilted camera sees a TRAPEZOID of ground, so a fixed rectangle
 * either leaks figures off-screen or leaves the corners empty.
 *
 * `halfXNear`/`halfXFar` bracket that trapezoid; wrapping interpolates between
 * them, so agents disappear exactly at the edge of frame and reappear at the
 * far side.
 */
interface Bounds {
  zNear: number;
  zFar: number;
  /** tan(vFov/2) * aspect — half-width per unit of view-axis depth. */
  spread: number;
  /** Camera position and forward axis, for the exact depth calculation. */
  cx: number;
  cy: number;
  cz: number;
  fx: number;
  fy: number;
  fz: number;
}

/** How far the cursor's repulsion reaches, in world units. */
const CURSOR_RADIUS = 6;

const EMITTER_SHAPES = ['drop', 'puff', 'heart', 'spark'] as const;

/** Velocity carry-over per frame. High = smooth paths, slow to turn. */
const INERTIA = 0.93;
const WANDER_FORCE = 0.0011;
/** How far from the edge they start turning back, in world units. */
const EDGE_MARGIN = 2;
const EDGE_FORCE = 0.0016;

interface Agent {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  phase: number;
  bias: number;
  height: number;
  seed: number;
  archetype: number;
  /** Eases 0..1 on spawn and retype, so the mix slides instead of popping. */
  blend: number;
  /** Index of whoever this agent is currently fixated on, or -1. */
  target: number;
  /** 0 = idle. Drives the punch/recoil animation. */
  action: number;
  cooldown: number;
  /** Frames left frozen (startle). */
  frozen: number;
  /** Emitted-particle life, 0..1. Wraps. */
  emit: number;
  /** Persistent heading target. Drifts slowly so paths curve, not twitch. */
  wander: number;
}

/**
 * Exact half-width of the frustum over the ground at depth `z`.
 *
 * Frustum width scales with distance along the VIEW AXIS, which is not linear
 * in z for a tilted camera — interpolating between the near and far corner
 * widths overestimates by a couple of units through the middle of the field,
 * which is precisely where most agents are, so they leak off the sides.
 */
function halfXAt(b: Bounds, z: number): number {
  const depth = (0 - b.cx) * b.fx + (0 - b.cy) * b.fy + (z - b.cz) * b.fz;
  return Math.max(1, b.spread * depth);
}

function makeAgent(archetype: number, b: Bounds): Agent {
  const z = b.zFar + Math.random() * (b.zNear - b.zFar);
  return {
    pos: new THREE.Vector3((Math.random() - 0.5) * 2 * halfXAt(b, z), 0, z),
    vel: new THREE.Vector3((Math.random() - 0.5) * 0.1, 0, (Math.random() - 0.5) * 0.1),
    phase: Math.random() * Math.PI * 2,
    bias: 0.8 + Math.random() * 0.4,
    height: 0.92 + Math.random() * 0.16,
    seed: Math.random() * Math.PI * 2,
    archetype,
    blend: 0,
    target: -1,
    action: 0,
    cooldown: Math.random() * 120,
    frozen: 0,
    emit: Math.random(),
    wander: Math.random() * Math.PI * 2,
  };
}

/** Capsule pivoted at its TOP so it can swing from a joint. */
function limb(radius: number, len: number): THREE.CapsuleGeometry {
  const g = new THREE.CapsuleGeometry(radius, Math.max(0.01, len - radius * 2), 4, 10);
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
  // Steep enough that the ground plane fills the frame rather than sitting in
  // a band across the middle.
  camera.position.set(0, 15, 13);
  camera.lookAt(0, 0.4, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.66));
  const key = new THREE.DirectionalLight(0xffffff, 1.0);
  key.position.set(5, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7fa8ff, 0.45);
  rim.position.set(-7, 3, -9);
  scene.add(rim);

  // Smooth-shaded and capsule-based — rounder reads as softer and more
  // characterful than the faceted version, and costs nothing at this scale.
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0 });
  const eyeMat = new THREE.MeshBasicMaterial({ color: 0x14141a });
  // Emitters fade by shrinking and darkening rather than by opacity — a
  // shared material can't vary alpha per instance, but it can vary colour.
  const puffMat = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
  });
  const solidMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9 });

  const parts = {
    head: new THREE.InstancedMesh(new THREE.SphereGeometry(0.27, 16, 14), mat, MAX_AGENTS),
    hair: new THREE.InstancedMesh(new THREE.SphereGeometry(0.29, 14, 10, 0, Math.PI * 2, 0, 1.5), mat, MAX_AGENTS),
    torso: new THREE.InstancedMesh(limb(0.25, 0.72), mat, MAX_AGENTS),
    armL: new THREE.InstancedMesh(limb(0.075, 0.54), mat, MAX_AGENTS),
    armR: new THREE.InstancedMesh(limb(0.075, 0.54), mat, MAX_AGENTS),
    handL: new THREE.InstancedMesh(new THREE.SphereGeometry(0.1, 8, 7), mat, MAX_AGENTS),
    handR: new THREE.InstancedMesh(new THREE.SphereGeometry(0.1, 8, 7), mat, MAX_AGENTS),
    legL: new THREE.InstancedMesh(limb(0.09, 0.58), mat, MAX_AGENTS),
    legR: new THREE.InstancedMesh(limb(0.09, 0.58), mat, MAX_AGENTS),
    footL: new THREE.InstancedMesh(new THREE.SphereGeometry(0.11, 8, 7), mat, MAX_AGENTS),
    footR: new THREE.InstancedMesh(new THREE.SphereGeometry(0.11, 8, 7), mat, MAX_AGENTS),
    eyeL: new THREE.InstancedMesh(new THREE.BoxGeometry(0.07, 0.09, 0.05), eyeMat, MAX_AGENTS),
    eyeR: new THREE.InstancedMesh(new THREE.BoxGeometry(0.07, 0.09, 0.05), eyeMat, MAX_AGENTS),
  };

  // One instanced mesh per particle SHAPE, shared across the emotions that use
  // it — four draw calls however many are emitting.
  const pgeo = particleGeometries();
  const emitters: Record<Exclude<ParticleShape, 'none'>, THREE.InstancedMesh> = {
    drop: new THREE.InstancedMesh(pgeo.drop, solidMat, MAX_AGENTS),
    puff: new THREE.InstancedMesh(pgeo.puff, puffMat, MAX_AGENTS),
    heart: new THREE.InstancedMesh(pgeo.heart, solidMat, MAX_AGENTS),
    spark: new THREE.InstancedMesh(pgeo.spark, solidMat, MAX_AGENTS),
  };

  const meshes = [...Object.values(parts), ...Object.values(emitters)];
  const skinParts = [
    parts.head, parts.torso, parts.armL, parts.armR, parts.handL, parts.handR,
    parts.legL, parts.legR, parts.footL, parts.footR,
  ];
  for (const mesh of meshes) {
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    scene.add(mesh);
  }

  const agents: Agent[] = [];
  const consumed = new Set<number>();
  const dummy = new THREE.Object3D();
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  const skin = new THREE.Color();
  const hairColor = new THREE.Color();
  const shade = new THREE.Color();

  // Cursor as a repulsion field, projected onto the ground plane so it pushes
  // in world space rather than screen space.
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
  let tick = 0;
  let stopped = false;

  const bounds: Bounds = {
    zNear: 12, zFar: -12, spread: 0.4,
    cx: 0, cy: 15, cz: 13, fx: 0, fy: -1, fz: -1,
  };
  const corner = new THREE.Vector3();
  const forward = new THREE.Vector3();

  /** Project the four screen corners onto the ground to get the walkable area. */
  const computeBounds = () => {
    camera.getWorldDirection(forward);
    bounds.cx = camera.position.x;
    bounds.cy = camera.position.y;
    bounds.cz = camera.position.z;
    bounds.fx = forward.x;
    bounds.fy = forward.y;
    bounds.fz = forward.z;
    bounds.spread = Math.tan((camera.fov * Math.PI) / 360) * camera.aspect;

    let zNear = -Infinity;
    let zFar = Infinity;
    for (const [nx, ny] of [
      [-1, -1], [1, -1], [-1, 1], [1, 1],
    ] as const) {
      ndc.set(nx, ny);
      raycaster.setFromCamera(ndc, camera);
      if (!raycaster.ray.intersectPlane(groundPlane, corner)) continue;
      if (ny < 0) zNear = Math.max(zNear, corner.z);
      else zFar = Math.min(zFar, corner.z);
    }
    if (!Number.isFinite(zNear) || !Number.isFinite(zFar)) return;
    // Pulled IN at the far edge — a figure standing there has its head above
    // the ground point, so it would poke out of the top of frame.
    bounds.zNear = zNear - 0.5;
    bounds.zFar = zFar + 2.5;
  };

  const resize = () => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    computeBounds();
  };
  resize();

  /** Nearest other agent within `range`, or -1. Sampled, not exhaustive. */
  const nearest = (self: Agent, selfIdx: number, range: number): number => {
    let best = -1;
    let bestD = range * range;
    for (let s = 0; s < 10; s++) {
      const j = (Math.random() * agents.length) | 0;
      if (j === selfIdx) continue;
      const o = agents[j];
      const dx = o.pos.x - self.pos.x;
      const dz = o.pos.z - self.pos.z;
      const d2 = dx * dx + dz * dz;
      if (d2 < bestD) {
        bestD = d2;
        best = j;
      }
    }
    return best;
  };

  const frame = () => {
    if (stopped) return;
    raf = requestAnimationFrame(frame);

    tick++;
    const { affect, emphases } = store[side];
    trimEmphases();

    const intensity = Math.max(0, Math.min(1, (affect.intensity + 1) / 3));
    const effort = Math.max(0, Math.min(1, (affect.effort + 1) / 3));
    const movement = Math.max(0, Math.min(1, affect.movement / 2));

    // Quantity is itself an encoding: feeling more means more of them.
    const want = Math.round(BASE_AGENTS + intensity * (MAX_AGENTS - BASE_AGENTS));

    const target = distribution(want, intensity, effort, movement);
    const counts = new Array(ARCHETYPES.length).fill(0);
    for (const a of agents) counts[a.archetype]++;

    while (agents.length > want) agents.pop();
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
      agents.push(makeAgent(pick, bounds));
    }
    // Retype a couple per frame so the mix slides rather than cuts.
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
      victim.target = -1;
      counts[over]--;
      counts[under]++;
    }

    // Cohesion stays a mild ATTRACTION always — it's what keeps the crowd on
    // screen. Dispersal comes from local repulsion: inverted gravity scales
    // with distance, so it runs away and piles everyone on the edges.
    // Near-zero at rest. Wrapping is what contains the crowd, so cohesion is
    // free to be purely expressive — any standing pull collapses everyone into
    // the middle and leaves the viewport empty.
    const cohesion = effort * 0.00035;
    const personalSpace = 3.2 - effort * 1.7;
    // Deliberately unhurried. A crowd that darts around reads as noise; the
    // emotion is in posture, gait and behaviour, and those need time to be
    // seen. Everything below is scaled off this.
    const speed = 0.0035 + intensity * 0.022;
    const turn = 0.006 + movement * 0.045;

    const crowdLean = effort * 0.34;
    const crowdHunch = effort * 0.42;

    // Emphasis ripples outward — a syllable you leaned on disturbs the room.
    for (const e of emphases) {
      if (consumed.has(e.id)) continue;
      consumed.add(e.id);
      const oz = bounds.zFar + Math.random() * (bounds.zNear - bounds.zFar);
      const ox = (Math.random() - 0.5) * 2 * halfXAt(bounds, oz);
      for (const a of agents) {
        const dx = a.pos.x - ox;
        const dz = a.pos.z - oz;
        const dist = Math.hypot(dx, dz) || 1;
        if (dist > 15) continue;
        const push = (e.weight * 0.16) / dist;
        a.vel.x += (dx / dist) * push;
        a.vel.z += (dz / dist) * push;
      }
    }
    if (consumed.size > 512) consumed.clear();

    for (let i = 0; i < agents.length; i++) {
      const a = agents[i];
      const arch = ARCHETYPES[a.archetype];
      a.blend = Math.min(1, a.blend + 0.04);
      if (a.cooldown > 0) a.cooldown--;
      if (a.action > 0) a.action = Math.max(0, a.action - 0.06);
      if (a.frozen > 0) a.frozen--;

      // --- behaviour ---------------------------------------------------
      switch (arch.behavior) {
        case 'strike': {
          // Seeks someone out, closes to arm's length, swings.
          if (a.target < 0 || a.target >= agents.length) a.target = nearest(a, i, 12);
          const t = agents[a.target];
          if (t) {
            const dx = t.pos.x - a.pos.x;
            const dz = t.pos.z - a.pos.z;
            const d = Math.hypot(dx, dz) || 1;
            // Stop OUTSIDE personal space. Closing any further means the
            // approach force and the separation force fight each other every
            // frame, and the whole body vibrates at the contact point.
            const reach = personalSpace + 0.4;
            if (d > reach) {
              a.vel.x += (dx / d) * 0.006;
              a.vel.z += (dz / d) * 0.006;
            } else if (a.cooldown <= 0) {
              a.action = 1;
              a.cooldown = 50 + Math.random() * 50;
              // knock them back
              t.vel.x += (dx / d) * 0.055;
              t.vel.z += (dz / d) * 0.055;
              t.frozen = 8;
              a.target = -1;
            }
          }
          break;
        }

        case 'flee': {
          const t = nearest(a, i, 7);
          if (t >= 0) {
            const o = agents[t];
            const dx = a.pos.x - o.pos.x;
            const dz = a.pos.z - o.pos.z;
            const d = Math.hypot(dx, dz) || 1;
            a.vel.x += (dx / d) * 0.007;
            a.vel.z += (dz / d) * 0.007;
          }
          break;
        }

        case 'frantic': {
          // Bolts, changes its mind constantly, and sweats doing it.
          if (a.cooldown <= 0) {
            const ang = Math.random() * Math.PI * 2;
            a.vel.x += Math.sin(ang) * 0.022;
            a.vel.z += Math.cos(ang) * 0.022;
            a.cooldown = 20 + Math.random() * 30;
          }
          break;
        }

        case 'approach': {
          if (a.target < 0 || a.target >= agents.length || a.cooldown <= 0) {
            a.target = nearest(a, i, 14);
            a.cooldown = 90 + Math.random() * 90;
          }
          const t = agents[a.target];
          if (t) {
            const dx = t.pos.x - a.pos.x;
            const dz = t.pos.z - a.pos.z;
            const d = Math.hypot(dx, dz) || 1;
            // Closes, then lingers rather than colliding.
            const pull = d > 2.4 ? 0.0045 : -0.0025;
            a.vel.x += (dx / d) * pull;
            a.vel.z += (dz / d) * pull;
          }
          break;
        }

        case 'withdraw': {
          // Drifts outward, keeps to the edges.
          const d = Math.hypot(a.pos.x, a.pos.z) || 1;
          a.vel.x += (a.pos.x / d) * 0.004;
          a.vel.z += (a.pos.z / d) * 0.004;
          break;
        }

        case 'startle': {
          // Freezes, then recoils.
          if (a.cooldown <= 0) {
            a.frozen = 14 + Math.random() * 20;
            a.action = 1;
            a.cooldown = 90 + Math.random() * 120;
            a.vel.x *= -0.7;
            a.vel.z *= -0.7;
          }
          break;
        }

        case 'wander':
        default:
          break;
      }

      // --- steering ----------------------------------------------------
      // Forces accumulate, then integrate with heavy inertia. White noise
      // applied straight to velocity reads as vibration; a slowly-drifting
      // wander angle plus damping reads as intent.
      let fx = 0;
      let fz = 0;

      a.wander += (Math.random() - 0.5) * turn;
      fx += Math.sin(a.wander) * WANDER_FORCE;
      fz += Math.cos(a.wander) * WANDER_FORCE;

      fx -= a.pos.x * cohesion;
      fz -= a.pos.z * cohesion;

      // Turn back before the edge rather than teleporting across it. Wrapping
      // reads as figures glitching in and out; a soft push plus the clamp
      // below reads as a room they're milling around inside.
      const edgeX = halfXAt(bounds, a.pos.z);
      if (a.pos.x > edgeX - EDGE_MARGIN) fx -= (a.pos.x - edgeX + EDGE_MARGIN) * EDGE_FORCE;
      if (a.pos.x < -edgeX + EDGE_MARGIN) fx += (-edgeX + EDGE_MARGIN - a.pos.x) * EDGE_FORCE;
      if (a.pos.z > bounds.zNear - EDGE_MARGIN)
        fz -= (a.pos.z - bounds.zNear + EDGE_MARGIN) * EDGE_FORCE;
      if (a.pos.z < bounds.zFar + EDGE_MARGIN)
        fz += (bounds.zFar + EDGE_MARGIN - a.pos.z) * EDGE_FORCE;

      if (cursorActive) {
        const dx = a.pos.x - cursor.x;
        const dz = a.pos.z - cursor.z;
        const d2 = dx * dx + dz * dz;
        const radius = arch.behavior === 'flee' ? CURSOR_RADIUS * 1.6 : CURSOR_RADIUS;
        if (d2 > 0.01 && d2 < radius * radius) {
          const d = Math.sqrt(d2);
          const push = (1 - d / radius) ** 2 * (arch.behavior === 'flee' ? 0.012 : 0.007);
          fx += (dx / d) * push;
          fz += (dz / d) * push;
        }
      }

      // Personal space. Sampled rather than all-pairs, but on a STABLE stride
      // that advances every 8 frames — re-rolling neighbours every frame makes
      // the force flicker, which is most of what read as jitter.
      for (let s = 0; s < 5; s++) {
        const j = (i + 1 + s * 29 + (tick >> 3)) % agents.length;
        const other = agents[j];
        if (!other || other === a) continue;
        const dx = a.pos.x - other.pos.x;
        const dz = a.pos.z - other.pos.z;
        const d2 = dx * dx + dz * dz;
        if (d2 > 0.001 && d2 < personalSpace * personalSpace) {
          const d = Math.sqrt(d2);
          const push = ((personalSpace - d) / personalSpace) * 0.0016;
          fx += (dx / d) * push;
          fz += (dz / d) * push;
        }
      }

      a.vel.x = a.vel.x * INERTIA + fx;
      a.vel.z = a.vel.z * INERTIA + fz;

      // Damp toward the speed this affect implies rather than clamping —
      // acceleration and settling both read as feeling.
      const wanted = a.frozen > 0 ? 0.002 : speed * a.bias * arch.speedBias;
      const current = a.vel.length() || 0.0001;
      a.vel.multiplyScalar(1 + (wanted / current - 1) * 0.08);
      a.pos.add(a.vel);

      // Hard stop. Anything that still reaches the edge — a punch impulse, a
      // panic bolt — is reflected rather than teleported, and its wander
      // target is turned inward so it doesn't grind along the wall.
      if (a.pos.z > bounds.zNear) {
        a.pos.z = bounds.zNear;
        a.vel.z = -Math.abs(a.vel.z);
        a.wander = Math.PI;
      } else if (a.pos.z < bounds.zFar) {
        a.pos.z = bounds.zFar;
        a.vel.z = Math.abs(a.vel.z);
        a.wander = 0;
      }
      const hx = halfXAt(bounds, a.pos.z);
      if (a.pos.x > hx) {
        a.pos.x = hx;
        a.vel.x = -Math.abs(a.vel.x);
        a.wander = -Math.PI / 2;
      } else if (a.pos.x < -hx) {
        a.pos.x = -hx;
        a.vel.x = Math.abs(a.vel.x);
        a.wander = Math.PI / 2;
      }

      const energy = Math.min(1, current * 52);
      // Frequency is capped and does NOT take bounceBias — that scales
      // amplitude. Letting it scale rate makes fast archetypes alias.
      a.phase += Math.min(0.13, (0.014 + current * 1.7) * CADENCE[arch.gait]);

      // --- pose ---------------------------------------------------------
      const p = pose(arch.gait, a.phase, energy, a.seed);
      const yaw = Math.atan2(a.vel.x, a.vel.z);
      const h = a.height * arch.heightBias * (0.55 + a.blend * 0.45);
      const len = arch.limbLength;
      const myLean = crowdLean + arch.leanBias + p.lean;
      const headPitch = myLean * 0.7 + p.headDrop;
      const myHunch = Math.max(0, Math.min(1, crowdHunch + arch.hunchBias));
      const upright = 1 - myHunch * 0.42;
      const hipY = (0.58 * len + p.bounce * arch.bounceBias) * h;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);

      skin.setHSL(arch.hue, arch.sat, arch.light);
      hairColor.setHSL(arch.hairHue, arch.hairSat, arch.hairLight);

      // legs + feet
      const legs = [
        { mesh: parts.legL, foot: parts.footL, dir: 1, xo: -0.1 },
        { mesh: parts.legR, foot: parts.footR, dir: -1, xo: 0.1 },
      ];
      for (const l of legs) {
        const swing = p.leg * l.dir;
        dummy.position.set(a.pos.x + cy * l.xo, hipY, a.pos.z - sy * l.xo);
        dummy.rotation.set(swing, yaw, 0);
        dummy.scale.set(h * arch.girth, h * len, h * arch.girth);
        dummy.updateMatrix();
        l.mesh.setMatrixAt(i, dummy.matrix);

        // foot at the end of the leg
        const legLen = 0.58 * h * len;
        dummy.position.set(
          a.pos.x + cy * l.xo + sy * Math.sin(swing) * legLen,
          hipY - Math.cos(swing) * legLen + 0.05 * h,
          a.pos.z - sy * l.xo + cy * Math.sin(swing) * legLen
        );
        dummy.rotation.set(0, yaw, 0);
        dummy.scale.setScalar(h);
        dummy.updateMatrix();
        l.foot.setMatrixAt(i, dummy.matrix);
      }

      // torso
      const shoulderY = hipY + 0.66 * h * upright;
      dummy.position.set(a.pos.x, shoulderY, a.pos.z);
      dummy.rotation.set(myLean, yaw, p.sway);
      dummy.scale.set(h * arch.girth, h * upright, h * arch.girth);
      dummy.updateMatrix();
      parts.torso.setMatrixAt(i, dummy.matrix);

      // arms + hands. The right arm carries the punch/recoil action.
      const arms = [
        { mesh: parts.armL, hand: parts.handL, dir: -1, xo: -0.29 },
        { mesh: parts.armR, hand: parts.handR, dir: 1, xo: 0.29 },
      ];
      for (const arm of arms) {
        let swing = p.arm * arm.dir + myLean * 0.5;
        if (a.action > 0) {
          // Thrust forward, hard, then ease back.
          const thrust = Math.sin(a.action * Math.PI) * 1.7;
          swing = arm.dir > 0 ? -thrust : swing * (1 - a.action * 0.6);
        }
        const inset = arm.xo * arch.girth * (1 - myHunch * 0.28);
        dummy.position.set(
          a.pos.x + cy * inset,
          shoulderY - 0.03 * h,
          a.pos.z - sy * inset
        );
        dummy.rotation.set(swing, yaw, -arm.dir * (myHunch * 0.28 + p.armSpread));
        dummy.scale.setScalar(h);
        dummy.updateMatrix();
        arm.mesh.setMatrixAt(i, dummy.matrix);

        const armLen = 0.54 * h;
        dummy.position.set(
          a.pos.x + cy * inset + sy * Math.sin(swing) * armLen,
          shoulderY - 0.03 * h - Math.cos(swing) * armLen,
          a.pos.z - sy * inset + cy * Math.sin(swing) * armLen
        );
        dummy.rotation.set(0, yaw, 0);
        dummy.scale.setScalar(h);
        dummy.updateMatrix();
        arm.hand.setMatrixAt(i, dummy.matrix);
      }

      // head
      const headY = shoulderY + 0.26 * h * upright * arch.headScale;
      const headX = a.pos.x + Math.sin(myLean) * sy * 0.2 * h;
      const headZ = a.pos.z + Math.sin(myLean) * cy * 0.2 * h;
      dummy.position.set(headX, headY, headZ);
      dummy.rotation.set(headPitch, yaw, p.sway * 1.4);
      dummy.scale.setScalar(h * arch.headScale);
      dummy.updateMatrix();
      parts.head.setMatrixAt(i, dummy.matrix);

      // hair
      dummy.position.y = headY + 0.2 * h * arch.headScale;
      dummy.scale.setScalar(h * arch.headScale * arch.hair);
      dummy.updateMatrix();
      parts.hair.setMatrixAt(i, dummy.matrix);
      parts.hair.setColorAt(i, hairColor);

      // eyes — small, but they do an enormous amount for character
      const eyeR = 0.24 * h * arch.headScale;
      for (const [mesh, dir] of [
        [parts.eyeL, -1],
        [parts.eyeR, 1],
      ] as const) {
        dummy.position.set(
          headX + sy * eyeR + cy * dir * 0.1 * h * arch.headScale,
          headY + 0.02 * h,
          headZ + cy * eyeR - sy * dir * 0.1 * h * arch.headScale
        );
        dummy.rotation.set(0, yaw, 0);
        dummy.scale.setScalar(h * arch.headScale);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }

      // emitted particle — steam, tears, hearts, sweat, shock marks
      const spec = PARTICLES[arch.id];
      for (const shape of EMITTER_SHAPES) {
        if (spec.shape !== shape) {
          emitters[shape].setMatrixAt(i, hidden);
          continue;
        }
        a.emit += spec.rate;
        if (a.emit > 1) a.emit -= 1;
        const life = a.emit;
        const wobble = Math.sin(life * Math.PI * 3 + a.seed) * spec.drift;
        dummy.position.set(
          headX + wobble * h + Math.sin(a.seed) * 0.16 * h,
          (spec.fromEyes ? headY - 0.02 * h : headY + 0.32 * h) + life * spec.rise * h,
          headZ + Math.cos(a.seed) * 0.16 * h + (spec.fromEyes ? 0.16 * h * cy : 0)
        );
        dummy.rotation.set(0, yaw, wobble * 0.6);
        // Fade by shrinking — alpha can't vary per instance on a shared material.
        const grow = 1 + (spec.grow - 1) * life;
        const fade = 1 - life * life;
        dummy.scale.setScalar(spec.size * grow * fade * h * 2);
        dummy.updateMatrix();
        emitters[shape].setMatrixAt(i, dummy.matrix);
        shade.setHSL(spec.color[0], spec.color[1], spec.color[2] * (1 - life * 0.55));
        emitters[shape].setColorAt(i, shade);
      }

      for (const mesh of skinParts) {
        shade.copy(skin).multiplyScalar(mesh === parts.head ? 1.1 : 1);
        mesh.setColorAt(i, shade);
      }
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
      eyeMat.dispose();
      puffMat.dispose();
      solidMat.dispose();
      renderer.dispose();
    },
    resize,
  };
}
