// HumanGuard — API configuration
// Single source of truth for the backend URL.
// Edit HUMANGUARD_API to point at a different deployment.
var HUMANGUARD_API = 'https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com';

// Canonical site URL — used in share messages, links, etc.
var HUMANGUARD_SITE = 'https://humanguard.net';

// Master API key used by first-party pages (demo, dashboard, simulator).
// Set HUMANGUARD_MASTER_KEY to your key before deploying.
var HUMANGUARD_MASTER_KEY = 'hg_master_cdf64cd876cfbb2adbf054b04f9b8f80';

/**
 * Drop-in replacement for fetch() that automatically attaches the master
 * X-Api-Key header on all requests to HUMANGUARD_API.
 *
 * Usage: humanguardFetch('/api/score', { method: 'POST', body: ... })
 */
function humanguardFetch(path, options) {
  options = options || {};
  options.headers = Object.assign({}, options.headers || {});
  if (HUMANGUARD_MASTER_KEY) {
    options.headers['X-Api-Key'] = HUMANGUARD_MASTER_KEY;
  }
  var url = path.startsWith('http') ? path : HUMANGUARD_API + path;
  return fetch(url, options);
}
