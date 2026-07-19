import { useEffect, useMemo, useState } from 'react';
import './explain.css';

const EMOTIONS = ['happy', 'sad', 'angry', 'afraid', 'surprised'] as const;
type ExplainEmotion = (typeof EMOTIONS)[number];

interface EmotionStory {
  label: string;
  strategy: string;
  firstMove: string;
  face: number;
  voice: number;
  detail: string;
}

const STORIES: Record<ExplainEmotion, EmotionStory> = {
  happy: {
    label: 'Happy',
    strategy: 'Celebrate',
    firstMove: 'Share the positive momentum, then ask what feels most meaningful.',
    face: 78,
    voice: 57,
    detail: 'Positive face and voice evidence turns an ambiguous sentence into a moment worth sharing.',
  },
  sad: {
    label: 'Sad',
    strategy: 'Support',
    firstMove: 'Acknowledge that the moment may feel heavier than the words alone suggest.',
    face: 76,
    voice: 49,
    detail: 'The reply makes room for the person without pretending the system knows their inner state.',
  },
  angry: {
    label: 'Angry',
    strategy: 'De-escalate',
    firstMove: 'Recognize the frustration without mirroring its heat, then offer a calm way forward.',
    face: 81,
    voice: 61,
    detail: 'Strong nonverbal frustration changes the response plan from generic curiosity to steady de-escalation.',
  },
  afraid: {
    label: 'Afraid',
    strategy: 'Reassure',
    firstMove: 'Open with a grounding sentence, then offer one concrete next step.',
    face: 73,
    voice: 54,
    detail: 'Fear-like evidence asks the model to be steady and specific without minimizing the concern.',
  },
  surprised: {
    label: 'Surprised',
    strategy: 'Orient',
    firstMove: 'Acknowledge the surprise, then ask one clarifying question to make sense of it.',
    face: 84,
    voice: 52,
    detail: 'A sudden expression and vocal onset make the same words read like a new development.',
  },
};

const PIPELINE = [
  {
    number: '01',
    title: 'Observe',
    subtitle: 'Face, voice, words',
    copy: 'A local face model reads downscaled camera frames. A browser heuristic measures vocal energy and movement. Gemma scores the transcript against published emotion directions.',
  },
  {
    number: '02',
    title: 'Translate',
    subtitle: 'One shared contract',
    copy: 'Every source maps into the same five labels: happy, sad, angry, afraid and surprised. Each source keeps its own confidence, and unsupported probability mass stays visible as uncertainty.',
  },
  {
    number: '03',
    title: 'Fuse',
    subtitle: 'Confidence + agreement',
    copy: 'The backend takes a confidence-weighted mean. When face, voice and words disagree, their agreement falls and the combined confidence is reduced instead of forcing a neat answer.',
  },
  {
    number: '04',
    title: 'Condition',
    subtitle: 'Context becomes language',
    copy: 'The fused distribution, reliable modality summaries and a conservative conversation strategy are written into Gemma’s prompt. There is no hidden-state emotion injection in the current build.',
  },
  {
    number: '05',
    title: 'Reflect',
    subtitle: 'Trace the response',
    copy: 'After Gemma responds, layer-28 activations are compared with all nine published directions. Strong phrase-level evidence drives the model crowd and safe ElevenLabs delivery tags.',
  },
] as const;

function distributionFor(emotion: ExplainEmotion, withContext: boolean): Record<ExplainEmotion, number> {
  if (!withContext) {
    return { happy: 19, sad: 23, angry: 18, afraid: 21, surprised: 19 };
  }
  const remaining = 42;
  const base: Record<ExplainEmotion, number> = {
    happy: Math.floor(remaining / 4),
    sad: Math.floor(remaining / 4),
    angry: Math.floor(remaining / 4),
    afraid: Math.floor(remaining / 4),
    surprised: Math.floor(remaining / 4),
  };
  base[emotion] = 58;
  const remainder = 100 - Object.values(base).reduce((sum, value) => sum + value, 0);
  const fallback = EMOTIONS.find((name) => name !== emotion) ?? 'happy';
  base[fallback] += remainder;
  return base;
}

export function Explain() {
  const [emotion, setEmotion] = useState<ExplainEmotion>('angry');
  const [withContext, setWithContext] = useState(true);
  const [activeStep, setActiveStep] = useState(3);
  const story = STORIES[emotion];
  const distribution = useMemo(
    () => distributionFor(emotion, withContext),
    [emotion, withContext]
  );
  const strategy = withContext ? story.strategy : 'Stay curious';

  useEffect(() => {
    document.body.classList.add('explanation-page');
    return () => document.body.classList.remove('explanation-page');
  }, []);

  return (
    <main className="explainer">
      <header className="explain-nav">
        <a className="explain-brand" href="/" aria-label="Null Mirror demo">
          <span aria-hidden="true">◌</span>
          <strong>NULL MIRROR</strong>
        </a>
        <nav aria-label="Explanation sections">
          <a href="#pipeline">Pipeline</a>
          <a href="#conditioning">Conditioning</a>
          <a href="#steering">Steering</a>
          <a href="#honesty">Limits</a>
        </nav>
        <a className="explain-demo-link" href="/">Open live mirror <span aria-hidden="true">↗</span></a>
      </header>

      <section className="explain-hero">
        <div className="hero-copy">
          <p className="explain-kicker">THE PART SPEECH-TO-TEXT DELETES</p>
          <h1>The AI gets more than your words.</h1>
          <p className="hero-lede">
            NULL MIRROR turns facial expression, vocal movement and language into transparent,
            uncertain context—then shows exactly how that context changes Gemma’s reply.
          </p>
          <div className="hero-actions">
            <a href="#conditioning" className="explain-primary">See what Gemma receives</a>
            <a href="/?demo=1" className="explain-secondary">Run the same-words proof</a>
          </div>
        </div>

        <div className="hero-signal" aria-label="Three human signals merging into one context">
          <div className="signal-source signal-source--face">
            <span>FACE</span>
            <i aria-hidden="true"><b /><b /></i>
            <small>visible expression</small>
          </div>
          <div className="signal-source signal-source--voice">
            <span>VOICE</span>
            <i aria-hidden="true">
              {Array.from({ length: 11 }, (_, index) => <b key={index} />)}
            </i>
            <small>energy + movement</small>
          </div>
          <div className="signal-source signal-source--words">
            <span>WORDS</span>
            <strong>“I’m fine.”</strong>
            <small>literal transcript</small>
          </div>
          <div className="signal-join" aria-hidden="true"><span /><span /><span /></div>
          <div className="signal-result">
            <small>FUSED CONTEXT</small>
            <strong>How should this moment be read?</strong>
            <div className="mini-mixture">
              {EMOTIONS.map((name) => <i key={name} className={`is-${name}`} />)}
            </div>
          </div>
        </div>
      </section>

      <section className="explain-thesis" aria-label="Core idea">
        <p>Transcription tells the model <em>what</em> you said.</p>
        <p>NULL MIRROR also gives it careful evidence about <em>how the moment landed.</em></p>
      </section>

      <section className="pipeline-section" id="pipeline">
        <div className="section-heading">
          <p className="explain-kicker">END-TO-END</p>
          <h2>One conversation, five inspectable steps.</h2>
          <p>Choose a step to see what is measured, transformed, or generated.</p>
        </div>
        <div className="pipeline-explainer">
          <div className="pipeline-track" role="tablist" aria-label="Pipeline steps">
            {PIPELINE.map((step, index) => (
              <button
                key={step.number}
                type="button"
                role="tab"
                aria-selected={activeStep === index}
                className={activeStep === index ? 'is-active' : ''}
                onClick={() => setActiveStep(index)}
              >
                <span>{step.number}</span>
                <strong>{step.title}</strong>
                <small>{step.subtitle}</small>
              </button>
            ))}
          </div>
          <article className="pipeline-detail" aria-live="polite">
            <span>{PIPELINE[activeStep].number} / 05</span>
            <h3>{PIPELINE[activeStep].title}</h3>
            <p>{PIPELINE[activeStep].copy}</p>
            <div className={`step-visual step-visual--${activeStep}`} aria-hidden="true">
              <i /><i /><i /><i /><i />
            </div>
          </article>
        </div>
      </section>

      <section className="conditioning-section" id="conditioning">
        <div className="section-heading section-heading--light">
          <p className="explain-kicker">THE CRUCIAL HANDOFF</p>
          <h2>Emotion reaches Gemma as explicit context.</h2>
          <p>
            This walkthrough uses illustrative values, but mirrors the real request shape and
            prompt-conditioning path in the running demo.
          </p>
        </div>

        <div className="conditioning-controls">
          <fieldset>
            <legend>Pick the nonverbal context</legend>
            <div className="emotion-picker">
              {EMOTIONS.map((name) => (
                <button
                  key={name}
                  type="button"
                  className={`is-${name}${emotion === name ? ' is-selected' : ''}`}
                  aria-pressed={emotion === name}
                  onClick={() => setEmotion(name)}
                >
                  <i aria-hidden="true" />
                  {STORIES[name].label}
                </button>
              ))}
            </div>
          </fieldset>
          <div className="context-switch" role="group" aria-label="Context comparison">
            <button
              type="button"
              className={!withContext ? 'is-selected' : ''}
              aria-pressed={!withContext}
              onClick={() => setWithContext(false)}
            >Words only</button>
            <button
              type="button"
              className={withContext ? 'is-selected' : ''}
              aria-pressed={withContext}
              onClick={() => setWithContext(true)}
            >+ face &amp; voice</button>
          </div>
        </div>

        <div className="handoff-grid">
          <article className="handoff-panel handoff-panel--signals">
            <header><span>01</span><strong>Browser → backend</strong></header>
            <p className="sample-utterance">“I’m fine. Let’s just get this over with.”</p>
            <div className="source-readings">
              <div className={!withContext ? 'is-muted' : ''}>
                <span>FACE</span><strong>{withContext ? story.label : 'withheld'}</strong>
                <small>{withContext ? `${story.face}% confidence` : 'no reading sent'}</small>
              </div>
              <div className={!withContext ? 'is-muted' : ''}>
                <span>PROSODY</span><strong>{withContext ? story.label : 'withheld'}</strong>
                <small>{withContext ? `${story.voice}% confidence` : 'no reading sent'}</small>
              </div>
              <div>
                <span>WORDS</span><strong>Mixed</strong><small>confidence capped at 35%</small>
              </div>
            </div>
            <pre><code>{withContext
              ? `{
  "transcript": "I'm fine…",
  "face_scores": { "${emotion}": 0.${story.face} },
  "face_confidence": 0.${story.face},
  "prosody_scores": { "${emotion}": 0.${story.voice} },
  "prosody_confidence": 0.${story.voice}
}`
              : `{
  "transcript": "I'm fine…"
}`}</code></pre>
          </article>

          <div className="handoff-arrow" aria-hidden="true"><span>confidence-aware fusion</span>→</div>

          <article className="handoff-panel handoff-panel--fusion">
            <header><span>02</span><strong>Backend → Gemma</strong></header>
            <div className="fusion-heading">
              <div>
                <small>RESPONSE PLAN</small>
                <strong>{strategy}</strong>
              </div>
              <span>{withContext ? 'nonverbal context active' : 'control condition'}</span>
            </div>
            <div className="fusion-bars" aria-label="Illustrative fused emotion distribution">
              {EMOTIONS.map((name) => (
                <div key={name}>
                  <span>{STORIES[name].label}</span>
                  <i><b className={`is-${name}`} style={{ width: `${distribution[name]}%` }} /></i>
                  <strong>{distribution[name]}%</strong>
                </div>
              ))}
            </div>
            <div className="prompt-preview">
              <span>PROMPT CONTEXT</span>
              <p>
                You are the emotionally attuned half of a live conversation. Treat the affect
                estimate as uncertain context. <mark>Strategy: {strategy}.</mark>{' '}
                {withContext ? story.firstMove : 'Open with a tentative observation or question.'}
                {' '}Do not mention sensors, scores, or emotion labels.
              </p>
            </div>
          </article>
        </div>
        <p className={`conditioning-outcome is-${withContext ? emotion : 'control'}`}>
          <span>{withContext ? `Context changed the plan to ${strategy}.` : 'Without nonverbal evidence, the model stays curious.'}</span>
          {withContext ? story.detail : 'The literal sentence alone does not justify a confident emotional claim.'}
        </p>
      </section>

      <section className="steering-section" id="steering">
        <div className="section-heading">
          <div>
            <p className="explain-kicker">NEXT EXPERIMENT · NOT IN THE LIVE DEMO YET</p>
            <h2>Give Gemma an affect vector—not English about one.</h2>
          </div>
          <p>
            Activation steering would keep emotion as a hidden model channel. Instead of spelling
            scores and labels out in the prompt, we would add one fused direction inside Gemma at
            the exact layer where those directions were measured.
          </p>
        </div>

        <div className="steering-comparison">
          <article className="steering-now">
            <header>
              <span>CURRENT PATH</span>
              <strong>Prompt conditioning</strong>
            </header>
            <div className="steering-flow" aria-label="Current emotion conditioning path">
              <div><small>FUSED SCORES</small><strong>angry 58%</strong></div>
              <i aria-hidden="true">→</i>
              <div><small>ENGLISH PROMPT</small><strong>“Strategy: de-escalate”</strong></div>
              <i aria-hidden="true">→</i>
              <div><small>GEMMA</small><strong>generates reply</strong></div>
            </div>
            <p>
              Transparent and controllable, but it translates the emotional signal back into words
              before the model can use it.
            </p>
          </article>

          <article className="steering-next">
            <header>
              <span>PROPOSED PATH</span>
              <strong>Layer-28 activation steering</strong>
            </header>
            <div className="steering-flow" aria-label="Proposed activation steering path">
              <div><small>FUSED SCORES</small><strong>five-way context</strong></div>
              <i aria-hidden="true">→</i>
              <div><small>VECTOR MIX</small><strong>v<sub>affect</sub></strong></div>
              <i aria-hidden="true">→</i>
              <div><small>LAYER 28</small><strong>residual stream</strong></div>
            </div>
            <p>
              The person’s final sentence receives the fused affect direction. Layers 29–41 then
              interpret the altered representation normally.
            </p>
          </article>
        </div>

        <div className="steering-math">
          <div className="equation" aria-label="Affect vector equals confidence times the sum of each centered emotion probability times its emotion vector">
            <span>v<sub>affect</sub></span>
            <i>=</i>
            <strong>confidence</strong>
            <i>×</i>
            <span>Σ<sub>i</sub> (p<sub>i</sub> − 0.2) v<sub>i</sub></span>
          </div>
          <div className="equation-notes">
            <p><span>p<sub>i</sub></span> each fused score across the five shared emotions</p>
            <p><span>− 0.2</span> centers the mixture so complete uncertainty produces zero steering</p>
            <p><span>v<sub>i</sub></span> the normalized 2,560-dimensional layer-28 emotion direction</p>
          </div>
        </div>

        <div className="layer-diagram" aria-label="Activation steering injection point in Gemma">
          <div className="layer-input">
            <span>TRANSCRIPT TOKENS</span>
            <p>… “let’s just get this over with.”</p>
          </div>
          <div className="layer-stack">
            <span>LAYERS 00–27</span>
            <i aria-hidden="true">→</i>
            <strong>LAYER 28 <b>+ strength × v<sub>affect</sub></b></strong>
            <i aria-hidden="true">→</i>
            <span>LAYERS 29–41</span>
          </div>
          <div className="layer-output">
            <span>GENERATION</span>
            <p>Later layers respond to the altered user-sentence representation.</p>
          </div>
        </div>

        <div className="steering-guardrails">
          <article>
            <span>WHY LAYER 28?</span>
            <strong>Matching width does not mean matching space.</strong>
            <p>
              The emotion directions and input embeddings are both 2,560-dimensional, but they
              represent different things. These vectors were measured in the layer-28 residual
              stream, so that is where the first steering experiment belongs.
            </p>
          </article>
          <article>
            <span>WHY ONLY THE USER SENTENCE?</span>
            <strong>Perceive the emotion before expressing it.</strong>
            <p>
              Initially steering generated response tokens could make Gemma imitate anger, fear or
              sadness. The cleaner first test changes how the user’s last sentence is represented,
              then lets the response emerge without continuous emotional pressure.
            </p>
          </article>
          <article>
            <span>THE POLICY GAP</span>
            <strong>“Angry” does not mean “de-escalate.”</strong>
            <p>
              A diagnostic direction communicates affect, not the safe action to take. The proposed
              path keeps a short generic conversational and safety policy while removing emotion
              names, percentages and strategy labels from the text prompt.
            </p>
          </article>
        </div>

        <div className="experiment-sweep">
          <header>
            <div>
              <p className="explain-kicker">THE CONTROLLED COMPARISON</p>
              <h3>Same transcript. Same fused signal. Three paths.</h3>
            </div>
            <p>We sweep steering strength and compare response strategy, specificity, safety and phrase-level vector traces.</p>
          </header>
          <ol>
            <li>
              <span>01</span>
              <div><strong>Emotion as prompt text</strong><p>The current, explicit baseline.</p></div>
              <small>CONTROL</small>
            </li>
            <li className="is-preferred">
              <span>02</span>
              <div><strong>Vector on the final user sentence</strong><p>The clean first activation-steering experiment.</p></div>
              <small>START HERE</small>
            </li>
            <li>
              <span>03</span>
              <div><strong>Vector on generated tokens</strong><p>Tests stronger expressive control and the risk of emotional imitation.</p></div>
              <small>FOLLOW-UP</small>
            </li>
          </ol>
          <p className="steering-caveat">
            These published vectors were built as diagnostic emotion directions, not validated
            steering controls. The experiment must earn the claim through measured comparisons; the
            site does not present steering as a working live feature yet.
          </p>
        </div>
      </section>

      <section className="response-section">
        <div className="section-heading">
          <p className="explain-kicker">THE RETURN PATH</p>
          <h2>Then we measure what Gemma expresses back.</h2>
          <p>User context shapes the answer. A separate internal trace shapes how that answer becomes visible and audible.</p>
        </div>
        <div className="response-flow" aria-label="Gemma response analysis flow">
          <article>
            <span>GENERATE</span>
            <strong>Gemma writes a concise reply</strong>
            <p>The context affects content and strategy, not merely a voice filter added afterward.</p>
          </article>
          <i aria-hidden="true">→</i>
          <article>
            <span>REPLAY + TRACE</span>
            <strong>Layer 28 is compared with nine directions</strong>
            <p>Cosine alignments are neutral-calibrated and grouped into short phrases.</p>
          </article>
          <i aria-hidden="true">→</i>
          <article>
            <span>MAKE IT LEGIBLE</span>
            <strong>Crowd motion + expressive speech</strong>
            <p>Strong shared-five evidence controls the right crowd and safe ElevenLabs tags.</p>
          </article>
        </div>
        <div className="nine-directions">
          <span>ALL NINE GEMMA DIAGNOSTICS</span>
          {['afraid', 'angry', 'calm', 'desperate', 'guilty', 'happy', 'loving', 'sad', 'surprised'].map((name) => (
            <code key={name}>{name}</code>
          ))}
        </div>
      </section>

      <section className="honesty-section" id="honesty">
        <div className="section-heading">
          <p className="explain-kicker">WHAT IT MEANS—AND WHAT IT DOESN’T</p>
          <h2>Inspectable evidence, not emotional certainty.</h2>
        </div>
        <div className="truth-grid">
          <article>
            <span>CAMERA</span>
            <strong>Visible expression estimate</strong>
            <p>HSEmotion runs on the local backend. It does not know a person’s private inner state.</p>
          </article>
          <article>
            <span>VOICE</span>
            <strong>Conservative acoustic heuristic</strong>
            <p>Loudness, pitch movement, spectral centroid and onset are compared with the current speaker’s rolling baseline.</p>
          </article>
          <article>
            <span>FUSION</span>
            <strong>A distribution with confidence</strong>
            <p>Weak readings are omitted. Disagreement lowers confidence. Neutral, contempt and disgust are never relabeled.</p>
          </article>
          <article>
            <span>GEMMA INPUT</span>
            <strong>Prompt conditioning—not steering</strong>
            <p>The live build writes affect context into natural-language instructions. Activation steering is the clearly separated next experiment above.</p>
          </article>
          <article>
            <span>GEMMA OUTPUT</span>
            <strong>Directional geometry—not feelings</strong>
            <p>Layer-28 scores are calibrated alignments with published vectors, not probabilities or proof of subjective emotion.</p>
          </article>
          <article>
            <span>DEMO MODE</span>
            <strong>A labeled causal control</strong>
            <p>The judge proof injects known rehearsal signals so the same transcript can be compared honestly. It never masquerades as live sensing.</p>
          </article>
        </div>
      </section>

      <section className="privacy-section">
        <div>
          <p className="explain-kicker">DATA PATH</p>
          <h2>Local where possible. Explicit when it leaves.</h2>
        </div>
        <ul>
          <li><span>01</span><p><strong>Camera frames</strong> are downscaled, processed by the local face endpoint and not retained.</p></li>
          <li><span>02</span><p><strong>Live prosody features</strong> stay inside the browser.</p></li>
          <li><span>03</span><p><strong>Recorded audio</strong> goes to ElevenLabs Scribe only after the user stops recording.</p></li>
          <li><span>04</span><p><strong>Generated response text</strong> goes to ElevenLabs when expressive speech is enabled.</p></li>
        </ul>
      </section>

      <footer className="explain-footer">
        <div>
          <p className="explain-kicker">SEE THE DIFFERENCE</p>
          <h2>Same words. Different human context. Different answer.</h2>
        </div>
        <div>
          <a className="explain-primary" href="/?demo=1">Open judge proof</a>
          <a className="explain-secondary" href="/">Use the live mirror</a>
        </div>
        <small>NULL MIRROR · local Gemma 4 E4B · layer 28 · five shared emotions · nine internal diagnostics</small>
      </footer>
    </main>
  );
}
