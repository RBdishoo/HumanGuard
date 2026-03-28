# Bot Detection System - Progress Log

## Overview
Building a bot detection system in phases. This log tracks progress, learnings, and blockers.

## Phase 1: Signal Collection
**Timeline**: Jan 20 - Feb 3, 2026 (2 weeks)
**Goal**: Build working data collection pipeline

### Week 1: Foundation
- [x] Environment setup (Python, venv, packages)
- [x] Git repository initialized
- [x] Project structure created
- [x] Flask server running
- [x] Frontend demo page working
- [x] Data being saved to JSON


## Phase 2: Feature Engineering
**Timeline**: Feb 3 - Feb 17, 2026 (2 weeks)
**Goal**: Extract meaningful features from signals

### Tasks
- [x] Implement mouse linearity calculation
- [x] Implement keystroke timing analysis
- [x] Implement session-level features
- [x] Document why each feature matters

## Phase 3: Data Collection & Labeling
**Timeline**: Feb 17 - Mar 3, 2026 (2 weeks)
**Goal**: Labeled dataset of humans vs bots

### Tasks
- [x] Collect human baseline data (100+ sessions)
- [x] Create bot simulations (5+ variations)
- [x] Label dataset (1 = bot, 0 = human)

## Phase 4: ML Model Training
**Timeline**: Mar 3 - Mar 24, 2026 (3 weeks)
**Goal**: Trained classifier (87%+ accuracy)

### Tasks
- [x] Baseline model (rule-based)
- [x] Logistic Regression
- [x] Random Forest
- [x] Model comparison & selection

## Phase 5: Deployment & Monitoring
**Timeline**: Mar 24 - Apr 7, 2026 (2 weeks)
**Goal**: Production-ready API + monitoring

### Tasks
- [ ] API endpoint for predictions
- [ ] AWS Lambda deployment
- [ ] CloudWatch monitoring
- [ ] Alerting system

## Phase 6: Advanced Features
**Timeline**: Apr 7+ (Ongoing)
**Goal**: Next-level portfolio work

### Tasks
- [ ] Adversarial testing
- [ ] SHAP explainability
- [ ] Ensemble models
- [ ] Continuous learning

---

## Key Decisions Made

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Database | JSON (Phase 1-3) → PostgreSQL (Phase 5+) | Simplicity first, scale later |
| Frontend | Vanilla JS (no React) | Learn fundamentals |
| Styling | Plain CSS (no Tailwind) | Understand core skills |
| ML Framework | Scikit-learn primary | Interpretability |


---

## Resources Used

- Python 3.11
- Flask 3.0.0
- Scikit-learn 1.3.0
- Git/GitHub

---

## Next Steps

1. Complete Flask server + frontend
2. Run demo and collect test data
3. Verify data is being saved correctly
4. Move to Phase 2: Feature Engineering
