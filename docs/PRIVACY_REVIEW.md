# Privacy review before sharing DJN artifacts

Automated redaction is a safeguard, not proof that an export is safe to publish.

- [ ] Export with `python manage.py export_sanitized_runs`; do not distribute `db.sqlite3`.
- [ ] Confirm `.env`, `credentials.json`, access tokens, OAuth data, and authorization headers are absent.
- [ ] Search the export for email addresses, phone numbers, API-key labels, bearer tokens, and personal names.
- [ ] Review raw and clarified prompts for confidential, personal, copyrighted, or licensed material.
- [ ] Review juror and judge outputs because a model may repeat sensitive prompt content.
- [ ] Confirm benchmark licenses permit redistribution of prompts and reference answers.
- [ ] Record the reviewer, review date, source commit, and export checksum.
- [ ] Share only the redacted export and configurations required to reproduce published metrics.

The sanitization rules live in `djn_engine/privacy.py` and should be extended for the deployment region and dataset.
