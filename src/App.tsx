/**
 * App shell. Two panes: what it heard in you (left), what it's feeling back
 * (right). The divider is permeable on purpose — figures should be able to
 * drift across, and the two crowds sync when it attunes to you. That
 * synchronisation is empathy made visible, with no labels anywhere.
 *
 * React owns layout and lifecycle only. The fusion tick and both render loops
 * run outside React entirely.
 */

import { useEffect } from 'react';
import { CrowdPane } from './crowd/CrowdPane.js';
import { fuse, startSources, updateAttunement, store } from './state/emotion.js';
import { pointerSource } from './sources/pointer.js';

export function App() {
  useEffect(() => {
    // Add prosody / face / heart-rate adapters here as they land. Each is one
    // file in src/sources — nothing else needs to change.
    const stopping = startSources([pointerSource]);

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      fuse();
      updateAttunement();

      // TEMPORARY: the model has no voice yet, so it lags the user with heavy
      // damping just to give the right pane something to do. Delete once
      // generation is wired up.
      const u = store.user.affect;
      const m = store.model.affect;
      m.intensity += (u.intensity * 0.7 - m.intensity) * 0.01;
      m.effort += (u.effort * 0.5 - m.effort) * 0.01;
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      void stopping.then((stop) => stop());
    };
  }, []);

  return (
    <main className="app">
      <CrowdPane side="user" label="you" />
      <div className="divider" />
      <CrowdPane side="model" label="it" />
    </main>
  );
}
