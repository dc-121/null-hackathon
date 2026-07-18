/**
 * Emitters.
 *
 * The cartoon vocabulary — steam off the angry ones, tears off the sad,
 * hearts off the loving. It's shameless, and it's the fastest-reading emotional
 * signal there is: nobody needs a legend for a heart.
 *
 * Kept in its own module so archetypes stay declarative. One instanced mesh
 * per SHAPE, shared across whichever emotions use it, so this costs four draw
 * calls total no matter how many are emitting.
 */

import * as THREE from 'three';

export type ParticleShape = 'drop' | 'puff' | 'heart' | 'spark' | 'none';

export interface ParticleSpec {
  shape: ParticleShape;
  /** Emitted from the eyes rather than the top of the head. */
  fromEyes?: boolean;
  /** Negative falls, positive rises. World units per unit of life. */
  rise: number;
  /** Sideways drift amplitude. */
  drift: number;
  /** Size at birth. */
  size: number;
  /** Multiplied into size as life runs out — puffs grow, drops shrink. */
  grow: number;
  color: [number, number, number];
  /** Life advanced per frame. Higher = faster cycle. */
  rate: number;
}

const NONE: ParticleSpec = {
  shape: 'none', rise: 0, drift: 0, size: 0, grow: 1, color: [0, 0, 0], rate: 0,
};

export const PARTICLES: Record<string, ParticleSpec> = {
  angry: {
    // Steam off the top of the head, billowing as it climbs.
    shape: 'puff',
    rise: 1.5,
    drift: 0.35,
    size: 0.16,
    grow: 2.6,
    color: [0.02, 0.5, 0.62],
    rate: 0.028,
  },
  sad: {
    // Tears, from the eyes, falling.
    shape: 'drop',
    fromEyes: true,
    rise: -1.4,
    drift: 0.05,
    size: 0.085,
    grow: 0.85,
    color: [0.55, 0.7, 0.72],
    rate: 0.022,
  },
  loving: {
    // Hearts, rising and swaying.
    shape: 'heart',
    rise: 1.8,
    drift: 0.5,
    size: 0.2,
    grow: 1.15,
    color: [0.95, 0.72, 0.66],
    rate: 0.016,
  },
  desperate: {
    // Sweat, flung off sideways.
    shape: 'drop',
    rise: -0.6,
    drift: 0.85,
    size: 0.08,
    grow: 0.9,
    color: [0.55, 0.6, 0.8],
    rate: 0.045,
  },
  surprised: {
    // A shock mark that pops out and vanishes.
    shape: 'spark',
    rise: 0.9,
    drift: 0.15,
    size: 0.22,
    grow: 0.5,
    color: [0.14, 0.9, 0.7],
    rate: 0.06,
  },
  afraid: {
    // Cold little beads, shaken loose.
    shape: 'drop',
    rise: 0.5,
    drift: 0.7,
    size: 0.07,
    grow: 0.7,
    color: [0.58, 0.55, 0.85],
    rate: 0.05,
  },
  guilty: {
    // A small dark cloud that hangs over them.
    shape: 'puff',
    rise: 0.55,
    drift: 0.18,
    size: 0.17,
    grow: 1.5,
    color: [0.62, 0.12, 0.26],
    rate: 0.014,
  },
  happy: NONE,
  calm: NONE,
};

/** Heart outline, extruded. Built once — bezier work is not frame-loop work. */
function heartGeometry(): THREE.BufferGeometry {
  const s = new THREE.Shape();
  s.moveTo(0, -0.5);
  s.bezierCurveTo(0.62, 0.08, 0.36, 0.72, 0, 0.34);
  s.bezierCurveTo(-0.36, 0.72, -0.62, 0.08, 0, -0.5);
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.18, bevelEnabled: false });
  g.center();
  return g;
}

export function particleGeometries(): Record<
  Exclude<ParticleShape, 'none'>,
  THREE.BufferGeometry
> {
  return {
    drop: new THREE.SphereGeometry(0.5, 6, 5),
    puff: new THREE.IcosahedronGeometry(0.5, 0),
    heart: heartGeometry(),
    spark: new THREE.ConeGeometry(0.32, 1, 4),
  };
}
