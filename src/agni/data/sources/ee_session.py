from __future__ import annotations

import json
import logging
import os

LOGGER = logging.getLogger(__name__)

# The high-volume endpoint is designed for many small concurrent getInfo/reduceRegion
# requests, which is exactly the access pattern of a parallel/cluster build.
HIGH_VOLUME_ENDPOINT = "https://earthengine-highvolume.googleapis.com"

# Environment variables let unattended Slurm jobs configure auth without CLI flags.
ENV_KEY_FILE = "EARTHENGINE_KEY_FILE"
ENV_SERVICE_ACCOUNT = "EARTHENGINE_SERVICE_ACCOUNT"
ENV_PROJECT = "EARTHENGINE_PROJECT"


def _service_account_email(key_file: str, service_account: str | None) -> str | None:
    if service_account:
        return service_account
    with open(key_file, encoding="utf-8") as handle:
        return json.load(handle).get("client_email")


def initialize_earth_engine(
    project: str | None = None,
    high_volume: bool = True,
    key_file: str | None = None,
    service_account: str | None = None,
) -> bool:
    """Initialize an Earth Engine session for the current process.

    Supports two auth modes:

    * Interactive/default credentials written by ``earthengine authenticate``
      (good for a laptop or a login node).
    * Service-account key file (good for unattended cluster jobs / job arrays):
      pass ``key_file`` or set ``EARTHENGINE_KEY_FILE``. The service-account email
      is read from the key JSON if not given.

    Returns True on success. On failure (missing package, no credentials, bad key)
    it logs a warning and returns False rather than raising, so pipelines that mock
    the adapters or run offline are unaffected. Interactive authentication is never
    triggered here; run ``earthengine authenticate`` once per machine first.
    """
    try:
        import ee
    except ImportError:
        LOGGER.warning("earthengine-api is not installed; skipping Earth Engine initialization")
        return False

    project = project or os.environ.get(ENV_PROJECT) or None
    key_file = key_file or os.environ.get(ENV_KEY_FILE) or None
    service_account = service_account or os.environ.get(ENV_SERVICE_ACCOUNT) or None
    opt_url = HIGH_VOLUME_ENDPOINT if high_volume else None

    try:
        if key_file:
            email = _service_account_email(key_file, service_account)
            credentials = ee.ServiceAccountCredentials(email, key_file)
            ee.Initialize(credentials, project=project, opt_url=opt_url)
            LOGGER.info(
                "Earth Engine initialized via service account %s (project=%s, high_volume=%s)",
                email,
                project,
                high_volume,
            )
        else:
            ee.Initialize(project=project, opt_url=opt_url)
            LOGGER.info(
                "Earth Engine initialized (project=%s, high_volume=%s)",
                project,
                high_volume,
            )
        return True
    except Exception as exc:
        LOGGER.warning(
            "Earth Engine initialization failed: %s. Run `earthengine authenticate` (interactive) "
            "or provide a service-account key via --ee-key / %s.",
            exc,
            ENV_KEY_FILE,
        )
        return False
