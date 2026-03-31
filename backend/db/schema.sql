-- HumanGuard PostgreSQL Schema
-- Run once against your database: psql $DATABASE_URL -f backend/db/schema.sql

BEGIN;

-- Sessions table — one row per unique browser session
CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(255) UNIQUE NOT NULL,
    user_agent      TEXT,
    viewport_width  INTEGER,
    viewport_height INTEGER,
    source          VARCHAR(100),
    label           VARCHAR(10),
    trained_at      TIMESTAMPTZ DEFAULT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions (session_id);

-- Signal batches — raw JSONB signal payloads linked to a session
CREATE TABLE IF NOT EXISTS signal_batches (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(255) NOT NULL REFERENCES sessions (session_id),
    raw_signals     JSONB NOT NULL,
    batch_timestamp TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signal_batches_session_id ON signal_batches (session_id);

-- Labels — human or bot classification per session
CREATE TABLE IF NOT EXISTS labels (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(255) UNIQUE NOT NULL REFERENCES sessions (session_id),
    label           VARCHAR(10) NOT NULL CHECK (label IN ('human', 'bot')),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_labels_session_id ON labels (session_id);

-- Predictions — model scoring results
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(255) NOT NULL REFERENCES sessions (session_id),
    prob_bot        DOUBLE PRECISION NOT NULL,
    label           VARCHAR(10) NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    scoring_type    VARCHAR(10) NOT NULL DEFAULT 'batch' CHECK (scoring_type IN ('batch', 'session')),
    source          VARCHAR(100),
    api_key         VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_session_id ON predictions (session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_api_key ON predictions (api_key);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);

-- Leaderboard — public challenge entries
CREATE TABLE IF NOT EXISTS leaderboard (
    id          SERIAL PRIMARY KEY,
    nickname    VARCHAR(30) NOT NULL,
    prob_bot    DOUBLE PRECISION NOT NULL,
    verdict     VARCHAR(10) NOT NULL,
    session_id  VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- API keys — multi-tenant scoring access
CREATE TABLE IF NOT EXISTS api_keys (
    id                  SERIAL PRIMARY KEY,
    key                 VARCHAR(50) UNIQUE NOT NULL,
    owner_email         VARCHAR(255) NOT NULL,
    plan                VARCHAR(20) NOT NULL DEFAULT 'free',
    monthly_limit       INTEGER NOT NULL DEFAULT 1000,
    current_month_count INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    active              BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys (key);

COMMIT;
