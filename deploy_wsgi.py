"""Deployment entrypoint for stock-market.

This file is additive: it does not modify app.py. It imports the existing
Flask app, bolts on a /healthz probe for Docker/Cloudflare, and lets the
session secret come from the environment instead of the hardcoded literal.

Run with:  gunicorn deploy_wsgi:app
"""

import os

from app import app  # noqa: E402  (app.py defines the Flask instance)

# --- session secret from the environment -----------------------------------
# app.py hardcodes app.secret_key. Setting SECRET_KEY overrides it without
# touching the source file. Nothing here fails if the variable is unset.
_secret = os.environ.get("SECRET_KEY")
if _secret:
    app.secret_key = _secret

# --- health probe -----------------------------------------------------------
if "healthz" not in app.view_functions:

    @app.route("/healthz")
    def healthz():
        """Liveness probe. Deliberately does no I/O and touches no model."""
        return {"status": "ok"}, 200


if __name__ == "__main__":  # pragma: no cover - local smoke test only
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5757)))
