/**
 * REFERENCE ADAPTER — copy this shape for prosody / face / heart rate.
 *
 * Dev-only. Drives affect from the mouse so the crowd can be worked on before
 * the audio pipeline exists: X = intensity, Y = effort.
 *
 * Note what it does and doesn't do:
 *  - fills ONLY the channels it can speak to (`provides`)
 *  - emits frames; never touches `store.affect` directly
 *  - stamps `at` so fusion can decay it
 */

import type { SourceAdapter } from '../state/emotion.js';

export const pointerSource: SourceAdapter = {
  id: 'pointer',
  provides: ['intensity', 'effort'],

  async start(emit) {
    const onMove = (e: PointerEvent) => {
      const x = e.clientX / window.innerWidth;
      const y = 1 - e.clientY / window.innerHeight;
      emit({
        source: 'pointer',
        side: 'user',
        at: performance.now(),
        confidence: 1,
        channels: {
          // Map 0..1 into the z-score range the channels are defined in.
          intensity: x * 4 - 1,
          effort: y * 4 - 1,
        },
      });
    };

    window.addEventListener('pointermove', onMove);
    return () => window.removeEventListener('pointermove', onMove);
  },
};
