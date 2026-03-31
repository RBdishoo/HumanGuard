# HumanGuard JavaScript SDK

Automatic bot detection for any website — one script tag, no dependencies, under 3 KB.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [JavaScript API](#javascript-api)
4. [Events](#events)
5. [Integration Examples](#integration-examples)
   - [Plain HTML](#plain-html)
   - [Form Protection](#form-protection)
   - [Content Gating](#content-gating)
   - [React](#react)
   - [Vue 3](#vue-3)
6. [How It Works](#how-it-works)
7. [Privacy](#privacy)
8. [Error Handling](#error-handling)
9. [API Key Setup](#api-key-setup)

---

## Installation

Add a single script tag to any page. No npm, no build step.

```html
<script
  src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
  data-api-key="hg_live_XXXX">
</script>
```

Place it anywhere in `<head>` or before `</body>`. The SDK initialises immediately
and begins collecting behavioral signals in the background.

---

## Configuration

All options are set as `data-*` attributes on the script tag.

| Attribute | Type | Default | Description |
|---|---|---|---|
| `data-api-key` | string | — | **Required.** Your `hg_live_*` API key. Get one at `POST /api/register`. |
| `data-threshold` | float | `0.5` | Bot probability cutoff passed through to your callbacks. The API always returns its own threshold; this is a client-side convenience value. |
| `data-auto-score` | boolean | `true` | Automatically score after `data-delay` ms. Set to `false` to score manually on demand. |
| `data-delay` | integer (ms) | `30000` | How long to collect signals before auto-scoring. Minimum effective value is ~5 000 ms. |
| `data-api-url` | string | HumanGuard API | Override the backend URL (useful for self-hosted deployments). |

### Example: manual scoring only

```html
<script
  src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
  data-api-key="hg_live_XXXX"
  data-auto-score="false">
</script>
```

### Example: faster scoring for short interactions

```html
<script
  src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
  data-api-key="hg_live_XXXX"
  data-delay="10000">
</script>
```

---

## JavaScript API

The SDK exposes `window.HumanGuard` after the script loads.

### `HumanGuard.sessionId` → `string`

A unique identifier for this browser session, generated at load time.
Format: `hg-<timestamp36>-<random>`.

```js
console.log(window.HumanGuard.sessionId);
// "hg-lxyz123-abc7d"
```

### `HumanGuard.result` → `object | null`

The most recent score result, or `null` if scoring hasn't completed yet.

```js
// {
//   prob_bot:   0.12,         // 0–1 bot probability
//   label:      "human",      // "human" | "bot"
//   threshold:  0.5,          // classification threshold used
//   session_id: "hg-..."      // same as HumanGuard.sessionId
// }
```

### `HumanGuard.getScore()` → `object | null`

Returns the current result object, or `null` before scoring completes. Equivalent to reading `HumanGuard.result`.

```js
const score = window.HumanGuard.getScore();
if (score) {
  console.log(`Bot probability: ${score.prob_bot}`);
}
```

### `HumanGuard.onScore(callback)` → `void`

Register a callback to receive the score result.

- If scoring has **already completed**, the callback fires immediately with the cached result.
- If scoring is **pending**, the callback is queued and fires when scoring completes.
- Safe to call multiple times with different callbacks.

```js
window.HumanGuard.onScore(function(result) {
  console.log(result.label); // "human" or "bot"
});
```

### `HumanGuard.score()` → `void`

Trigger scoring immediately, bypassing the auto-score timer. Idempotent — calling it multiple times has no effect after the first score.

```js
document.getElementById('myBtn').addEventListener('click', function() {
  window.HumanGuard.score();
});
```

---

## Events

The SDK dispatches events on `document`. Listen with `addEventListener`.

### `humanGuard:scored`

Fired when scoring completes successfully.

```js
document.addEventListener('humanGuard:scored', function(e) {
  const { prob_bot, label, threshold, session_id } = e.detail;
  console.log(`Label: ${label}, Score: ${prob_bot}`);
});
```

**`e.detail` shape:**

```js
{
  prob_bot:   number,  // 0.0–1.0
  label:      string,  // "human" | "bot"
  threshold:  number,  // classification threshold
  session_id: string,  // session identifier
}
```

### `humanGuard:error`

Fired when scoring fails (network error, invalid API key, etc.).

```js
document.addEventListener('humanGuard:error', function(e) {
  console.error('HumanGuard error:', e.detail.error);
});
```

**`e.detail` shape:**

```js
{
  error: string,  // human-readable error message
}
```

---

## Integration Examples

### Plain HTML

The simplest integration — just listen for the event:

```html
<!DOCTYPE html>
<html>
<head>
  <script
    src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
    data-api-key="hg_live_XXXX">
  </script>
</head>
<body>
  <div id="status">Verifying…</div>

  <script>
    document.addEventListener('humanGuard:scored', function(e) {
      document.getElementById('status').textContent =
        e.detail.label === 'human' ? '✓ Human verified' : '⚠ Bot detected';
    });
  </script>
</body>
</html>
```

---

### Form Protection

Block form submission when bot probability exceeds your threshold:

```html
<script
  src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
  data-api-key="hg_live_XXXX"
  data-auto-score="false">
</script>

<form id="signupForm">
  <input type="email" name="email" placeholder="Email" required />
  <button type="submit">Sign Up</button>
  <p id="msg"></p>
</form>

<script>
  document.getElementById('signupForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Score immediately and wait for result
    window.HumanGuard.score();
    const result = await new Promise(function(resolve) {
      window.HumanGuard.onScore(resolve);
    });

    if (result.prob_bot > 0.8) {
      document.getElementById('msg').textContent =
        'Submission blocked: suspicious activity detected.';
      return;
    }

    // Human — proceed
    this.submit();
  });
</script>
```

---

### Content Gating

Reveal premium content only to verified humans:

```html
<script
  src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
  data-api-key="hg_live_XXXX">
</script>

<div id="loading">Verifying your session…</div>
<div id="premium" style="display:none">
  <h2>Welcome, human!</h2>
  <p>Here is your exclusive content.</p>
</div>

<script>
  window.HumanGuard.onScore(function(result) {
    document.getElementById('loading').style.display = 'none';
    if (result.label === 'human') {
      document.getElementById('premium').style.display = 'block';
    } else {
      document.getElementById('loading').textContent =
        'Access restricted: bot activity detected.';
      document.getElementById('loading').style.display = 'block';
    }
  });
</script>
```

---

### React

Use a hook to integrate HumanGuard into React components:

```jsx
// hooks/useHumanGuard.js
import { useState, useEffect } from 'react';

export function useHumanGuard() {
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);

  useEffect(function() {
    function onScored(e) { setScore(e.detail); }
    function onError(e)  { setError(e.detail.error); }

    document.addEventListener('humanGuard:scored', onScored);
    document.addEventListener('humanGuard:error',  onError);

    // If already scored (script loaded before React mounted)
    if (window.HumanGuard?.result) {
      setScore(window.HumanGuard.result);
    }

    return function() {
      document.removeEventListener('humanGuard:scored', onScored);
      document.removeEventListener('humanGuard:error',  onError);
    };
  }, []);

  return { score, error, sessionId: window.HumanGuard?.sessionId };
}
```

```jsx
// In index.html (public/index.html):
// <script src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
//         data-api-key="hg_live_XXXX"></script>

// App.jsx
import { useHumanGuard } from './hooks/useHumanGuard';

function ProtectedFeature() {
  const { score, error } = useHumanGuard();

  if (!score)  return <p>Verifying…</p>;
  if (error)   return <p>Verification failed.</p>;

  return score.label === 'human'
    ? <div>Welcome, human! Here is the protected content.</div>
    : <div>Access restricted.</div>;
}
```

---

### Vue 3

Use a composable for Vue 3:

```js
// composables/useHumanGuard.js
import { ref, onMounted, onUnmounted } from 'vue';

export function useHumanGuard() {
  const score   = ref(null);
  const error   = ref(null);

  function onScored(e) { score.value = e.detail; }
  function onError(e)  { error.value = e.detail.error; }

  onMounted(function() {
    document.addEventListener('humanGuard:scored', onScored);
    document.addEventListener('humanGuard:error',  onError);
    if (window.HumanGuard?.result) {
      score.value = window.HumanGuard.result;
    }
  });

  onUnmounted(function() {
    document.removeEventListener('humanGuard:scored', onScored);
    document.removeEventListener('humanGuard:error',  onError);
  });

  return { score, error };
}
```

```vue
<!-- ProtectedContent.vue -->
<template>
  <div v-if="!score">Verifying…</div>
  <div v-else-if="score.label === 'human'">Welcome, human!</div>
  <div v-else>Bot detected. Access restricted.</div>
</template>

<script setup>
import { useHumanGuard } from '@/composables/useHumanGuard';
const { score } = useHumanGuard();
</script>
```

Add the script tag to `index.html`:

```html
<script
  src="https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js"
  data-api-key="hg_live_XXXX">
</script>
```

---

## How It Works

1. **Signal collection** begins silently the moment the script loads.
   - Mouse movements are sampled at most every 100 ms (up to 200 points).
   - Click events capture position, button, and timestamp.
   - Keystrokes record code and timing intervals only — **key content is never sent**.
   - Session duration is tracked from script load to scoring time.

2. **Scoring** sends the collected signals to `POST /api/score` via the HumanGuard API.
   The ML model (XGBoost / RandomForest) extracts 33 behavioral features and returns
   a bot probability score in ~100–300 ms.

3. **Result delivery** happens via:
   - The `humanGuard:scored` custom event on `document`
   - Any callbacks registered with `HumanGuard.onScore()`
   - The `HumanGuard.result` property

4. **Auto-scoring** triggers after `data-delay` ms (default 30 s), or immediately
   when a form is submitted — whichever comes first. Set `data-auto-score="false"`
   to disable the timer.

---

## Privacy

HumanGuard is designed with privacy as a first principle:

- **No PII collected.** Only pointer coordinates, click positions, keystroke timing
  codes, and session duration are captured. Key content (what you type) is never sent.
- **Do Not Track honoured.** If the browser sends `DNT: 1`, the SDK stops all signal
  collection. A score will still be requested (with empty signals), and the `DNT`
  header is forwarded to the API.
- **Transparency comment.** The SDK injects `<!-- HumanGuard active -->` into the
  document body so end-users who inspect the source can see it is active.
- **No cookies or localStorage.** Session IDs are ephemeral — generated fresh on
  each page load and not persisted across sessions.
- **HTTPS only.** All API communication uses HTTPS with TLS 1.2+.

---

## Error Handling

Listen for `humanGuard:error` to handle failures gracefully:

```js
document.addEventListener('humanGuard:error', function(e) {
  console.warn('HumanGuard:', e.detail.error);
  // e.g. "Invalid or inactive API key" (401)
  // e.g. "monthly limit reached" (429)
  // e.g. "Network error" (offline)
});
```

Common errors:

| Error | Cause | Fix |
|---|---|---|
| `Invalid or inactive API key` | Missing or wrong `data-api-key` | Get a key via `POST /api/register` |
| `monthly limit reached` | Free tier (1 000 req/month) exhausted | Upgrade plan or wait for reset |
| `data-api-key is required` | Attribute missing from script tag | Add `data-api-key="hg_live_..."` |
| `Network error` | Client offline or API unreachable | Check connectivity |

---

## API Key Setup

Get a free API key in one request:

```bash
curl -X POST https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com/api/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

Response:

```json
{
  "api_key": "hg_live_a1b2c3d4e5f6...",
  "plan": "free",
  "monthly_limit": 1000,
  "docs_url": "https://github.com/rubenbetabdishoo/HumanGuard#api"
}
```

Check your usage at any time:

```bash
curl https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com/api/usage \
  -H "X-Api-Key: hg_live_XXXX"
```

Response:

```json
{
  "count": 42,
  "limit": 1000,
  "percentage_used": 4.2,
  "plan": "free"
}
```
