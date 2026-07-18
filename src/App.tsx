/**
 * App shell. Two panes: what it heard in you (left), what it's feeling back
 * (right). The divider is meant to be permeable — figures should be able to
 * drift across, and the two crowds sync when it attunes to you.
 *
 * Nothing real is wired up yet. Each pane is a mount point: drop your engine
 * into the <canvas> and never let it touch React state.
 */

import { CrowdPane } from './crowd/CrowdPane.js';

export function App() {
  return (
    <main className="app">
      <CrowdPane side="user" label="you" />
      <div className="divider" />
      <CrowdPane side="model" label="it" />
    </main>
  );
}
