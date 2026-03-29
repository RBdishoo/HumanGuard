"""
HumanGuard Model Registry

Manages versioned model artifacts in S3.

Storage layout:
  s3://{MODEL_BUCKET}/models/champion.json       ← current champion info
  s3://{MODEL_BUCKET}/models/v1.0.0/model.joblib
  s3://{MODEL_BUCKET}/models/v1.0.0/scaler.joblib
  s3://{MODEL_BUCKET}/models/v1.0.0/feature_names.json
  s3://{MODEL_BUCKET}/models/v1.0.0/threshold.json
  s3://{MODEL_BUCKET}/models/v1.0.0/metadata.json

Environment variables:
  MODEL_BUCKET  — S3 bucket name (required for registry mode)
"""

import io
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Raised when a registry operation cannot be completed."""


class ModelRegistry:
    """Versioned model storage backed by S3 with in-memory caching."""

    def __init__(self, bucket: str = None, prefix: str = "models", _s3_client=None):
        """
        Parameters
        ----------
        bucket      : S3 bucket name. Falls back to MODEL_BUCKET env var.
        prefix      : Key prefix inside the bucket (default: "models").
        _s3_client  : Injectable S3 client — used for unit testing.
        """
        self._bucket = bucket or os.environ.get("MODEL_BUCKET", "")
        if not self._bucket:
            raise RegistryError("MODEL_BUCKET is not set and no bucket was provided")
        self._prefix = prefix.rstrip("/")
        self._cache: dict = {}          # version string → bundle dict
        self._s3_client = _s3_client    # injected or lazily created

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def _s3(self):
        if self._s3_client is not None:
            return self._s3_client
        import boto3
        return boto3.client("s3")

    def _key(self, version: str, filename: str) -> str:
        return f"{self._prefix}/{version}/{filename}"

    def _champion_key(self) -> str:
        return f"{self._prefix}/champion.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _put_json(self, key: str, obj: dict):
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(obj, default=str).encode(),
            ContentType="application/json",
        )

    def _get_json(self, key: str) -> dict:
        resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        return json.loads(resp["Body"].read())

    # ── Version numbering ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_version(v: str):
        """Parse 'v1.2.3' → (1, 2, 3)."""
        return tuple(int(x) for x in v.lstrip("v").split("."))

    @staticmethod
    def _format_version(major: int, minor: int, patch: int) -> str:
        return f"v{major}.{minor}.{patch}"

    def _next_version(self, existing: list) -> str:
        if not existing:
            return "v1.0.0"
        latest = max(existing, key=self._parse_version)
        maj, min_, pat = self._parse_version(latest)
        return self._format_version(maj, min_ + 1, 0)

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, model, scaler, feature_names: list, threshold: float,
             metadata: dict) -> str:
        """
        Serialize and upload a new model version to S3.

        Parameters
        ----------
        model         : Trained sklearn-compatible model (must support predict_proba).
        scaler        : Fitted StandardScaler.
        feature_names : Ordered list of feature name strings.
        threshold     : Classification threshold (float).
        metadata      : Dict with keys: accuracy, precision, recall, f1, roc_auc,
                        training_date, training_samples, model_type.
                        Additional keys are forwarded as-is.

        Returns
        -------
        version : str  (e.g. "v1.1.0")
        """
        import joblib

        existing = self.list_versions()
        version = self._next_version(existing)

        # Serialize model and scaler to in-memory bytes
        for name, obj in [("model.joblib", model), ("scaler.joblib", scaler)]:
            buf = io.BytesIO()
            joblib.dump(obj, buf)
            self._s3.put_object(
                Bucket=self._bucket,
                Key=self._key(version, name),
                Body=buf.getvalue(),
                ContentType="application/octet-stream",
            )

        self._put_json(self._key(version, "feature_names.json"), feature_names)
        self._put_json(self._key(version, "threshold.json"), {"threshold": threshold})

        full_meta = {
            **metadata,
            "version": version,
            "champion": False,
            "feature_names": feature_names,
            "pushed_at": self._now_iso(),
        }
        self._put_json(self._key(version, "metadata.json"), full_meta)

        logger.info("Pushed model %s to registry bucket=%s", version, self._bucket)
        return version

    def load(self, version: str = "latest") -> dict:
        """
        Load a model bundle from S3, with in-memory caching.

        Parameters
        ----------
        version : "latest" resolves to the current champion; otherwise a literal
                  version string like "v1.2.0".

        Returns
        -------
        dict with keys: model, scaler, feature_names, threshold, metadata, version
        """
        import joblib

        if version == "latest":
            version = self.get_champion()
            if not version:
                raise RegistryError("No champion model found in registry")

        if version in self._cache:
            return self._cache[version]

        def _load_joblib(key: str):
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            return joblib.load(io.BytesIO(resp["Body"].read()))

        model        = _load_joblib(self._key(version, "model.joblib"))
        scaler       = _load_joblib(self._key(version, "scaler.joblib"))
        feature_names = self._get_json(self._key(version, "feature_names.json"))
        threshold    = self._get_json(self._key(version, "threshold.json")).get("threshold", 0.5)
        metadata     = self._get_json(self._key(version, "metadata.json"))

        bundle = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "threshold": float(threshold),
            "metadata": metadata,
            "version": version,
        }
        self._cache[version] = bundle
        logger.info("Loaded model %s from registry", version)
        return bundle

    def list_versions(self) -> list:
        """Return a sorted list of all version strings present in S3."""
        prefix = self._prefix + "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        versions = []
        try:
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix, Delimiter="/"):
                for cp in page.get("CommonPrefixes", []):
                    folder = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                    if folder.startswith("v") and "." in folder:
                        versions.append(folder)
        except Exception as exc:
            logger.warning("list_versions failed: %s", exc)
        return sorted(versions, key=self._parse_version)

    def get_champion(self) -> str | None:
        """Return the current champion version string, or None if not set."""
        try:
            data = self._get_json(self._champion_key())
            return data.get("version")
        except Exception:
            return None

    def get_metadata(self, version: str = "latest") -> dict:
        """Fetch only the metadata.json for a version without loading the model."""
        if version == "latest":
            version = self.get_champion()
            if not version:
                return {}
        try:
            return self._get_json(self._key(version, "metadata.json"))
        except Exception as exc:
            logger.warning("get_metadata(%s) failed: %s", version, exc)
            return {}

    def promote(self, version: str):
        """
        Designate *version* as the new champion.

        Updates:
          - metadata.json for the promoted version (champion=True)
          - metadata.json for the previous champion (champion=False)
          - champion.json with the new champion and previous_version reference
        """
        # Mark old champion as non-champion
        previous = self.get_champion()
        if previous and previous != version:
            try:
                prev_meta = self._get_json(self._key(previous, "metadata.json"))
                prev_meta["champion"] = False
                self._put_json(self._key(previous, "metadata.json"), prev_meta)
                # Invalidate cache
                self._cache.pop(previous, None)
            except Exception as exc:
                logger.warning("Could not update previous champion metadata: %s", exc)

        # Mark new version as champion
        try:
            new_meta = self._get_json(self._key(version, "metadata.json"))
            new_meta["champion"] = True
            self._put_json(self._key(version, "metadata.json"), new_meta)
            self._cache.pop(version, None)
        except Exception as exc:
            logger.warning("Could not update promoted version metadata: %s", exc)

        # Write champion.json
        self._put_json(self._champion_key(), {
            "version": version,
            "previous_version": previous,
            "promoted_at": self._now_iso(),
        })
        logger.info("Promoted %s to champion (was: %s)", version, previous)

    def rollback(self) -> str:
        """
        Roll back to the version that was champion before the current one.

        Returns the version that was promoted.
        Raises RegistryError if there is no previous champion to roll back to.
        """
        try:
            champion_data = self._get_json(self._champion_key())
        except Exception as exc:
            raise RegistryError("Cannot read champion.json") from exc

        previous = champion_data.get("previous_version")
        if not previous:
            raise RegistryError("No previous champion recorded — cannot roll back")

        self.promote(previous)
        logger.info("Rolled back to %s", previous)
        return previous

    def invalidate_cache(self):
        """Clear the in-memory bundle cache (forces next load to re-fetch from S3)."""
        self._cache.clear()
