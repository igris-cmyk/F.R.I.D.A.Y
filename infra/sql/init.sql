-- F.R.I.D.A.Y Database Initialization

-- Note: We assume pgvector is installed in the image. 
-- Apache AGE installation via Docker image may require custom compilation.
-- For MVP, we will start with pgvector and enable AGE if available.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Attempt to create AGE if the image supports it. If it fails, it will safely error out.
DO $$ 
BEGIN
  CREATE EXTENSION IF NOT EXISTS age;
EXCEPTION
  WHEN OTHERS THEN
    RAISE NOTICE 'Apache AGE extension not found. Graph features will be disabled until custom image is built.';
END $$;


-- ============================================================
-- MEMORY TABLES
-- ============================================================

CREATE TABLE memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    layer           TEXT NOT NULL CHECK (layer IN (
                        'short_term', 'working', 'long_term', 
                        'episodic', 'procedural'
                    )),
    category        TEXT NOT NULL,          -- 'conversation', 'code', 'trade', 'research', etc.
    content         TEXT NOT NULL,          -- Raw content
    summary         TEXT,                   -- Condensed version
    embedding       vector(768),            -- nomic-embed-text
    importance      FLOAT DEFAULT 0.5,      -- 0.0 - 1.0
    access_count    INTEGER DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    source_agent    TEXT,                   -- Which agent created this
    source_event    UUID,                   -- Originating event
    metadata        JSONB DEFAULT '{}',     -- Flexible metadata
    is_encrypted    BOOLEAN DEFAULT FALSE,
    privacy_level   TEXT DEFAULT 'normal' CHECK (privacy_level IN (
                        'public', 'normal', 'sensitive', 'private'
                    ))
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_memories_embedding ON memories 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Composite index for filtered searches
CREATE INDEX idx_memories_layer_category ON memories (layer, category);
CREATE INDEX idx_memories_importance ON memories (importance DESC);
CREATE INDEX idx_memories_created ON memories (created_at DESC);
CREATE INDEX idx_memories_fts ON memories USING gin (to_tsvector('english', content));

-- ============================================================
-- EXECUTION LOG (Audit Trail)
-- ============================================================

CREATE TABLE execution_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent           TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    target          TEXT,
    risk_level      TEXT CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    permission_tier TEXT CHECK (permission_tier IN ('auto', 'logged', 'approval')),
    approval_status TEXT CHECK (approval_status IN ('auto', 'approved', 'rejected', 'pending')),
    input_summary   TEXT,
    output_summary  TEXT,
    duration_ms     INTEGER,
    success         BOOLEAN,
    error_message   TEXT,
    rollback_data   JSONB,
    trace_id        UUID,
    metadata        JSONB DEFAULT '{}'
);

CREATE INDEX idx_exec_log_time ON execution_log (timestamp DESC);
CREATE INDEX idx_exec_log_agent ON execution_log (agent, timestamp DESC);
