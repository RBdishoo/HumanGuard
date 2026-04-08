// HumanGuard — API configuration
// Single source of truth for the backend URL.
// Edit HUMANGUARD_API to point at a different deployment.
var HUMANGUARD_API = 'https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com';

// Canonical site URL — used in share messages, links, etc.
var HUMANGUARD_SITE = 'https://humanguard.net';

// API key for first-party pages (demo, simulator).
// Injected server-side as <meta name="hg-api-key"> — never hardcode a secret here.
var _hgApiKeyMeta = document.querySelector('meta[name="hg-api-key"]');
var _hgApiKey = _hgApiKeyMeta ? _hgApiKeyMeta.getAttribute('content') : '';

/**
 * Drop-in replacement for fetch() that automatically attaches the
 * X-Api-Key header (from server-injected meta tag) on requests to
 * HUMANGUARD_API.
 *
 * Usage: humanguardFetch('/api/score', { method: 'POST', body: ... })
 */
function humanguardFetch(path, options) {
  options = options || {};
  options.headers = Object.assign({}, options.headers || {});
  if (_hgApiKey) {
    options.headers['X-Api-Key'] = _hgApiKey;
  }
  var url = path.startsWith('http') ? path : HUMANGUARD_API + path;
  return fetch(url, options);
}
