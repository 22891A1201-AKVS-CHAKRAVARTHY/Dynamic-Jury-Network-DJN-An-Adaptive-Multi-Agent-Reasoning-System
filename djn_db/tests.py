from django.test import TestCase

from djn_engine.run import (
    JUROR_PROMPT_VERSION,
    ROLE_INSTRUCTIONS,
    ROLE_INSTRUCTIONS_VERSION,
    build_role_aware_juror_prompt,
)

from .db_writer import upsert_run, write_round
from .models import JurorResponse


class RolePromptAndAuditTests(TestCase):

    def test_all_roles_render_distinct_instructions(self):
        rendered_prompts = {}

        for role, instruction in ROLE_INSTRUCTIONS.items():
            prompt = build_role_aware_juror_prompt(role)

            messages = prompt.format_messages(
                query="Should this system be deployed?",
                round_context="No previous round.",
            )

            system_message = messages[0].content

            self.assertIn(
                f"Assigned role: {role}",
                system_message,
            )
            self.assertIn(
                instruction,
                system_message,
            )

            rendered_prompts[role] = system_message

        self.assertEqual(
            len(set(rendered_prompts.values())),
            len(ROLE_INSTRUCTIONS),
        )

    def test_writer_persists_role_audit_metadata(self):
        run = upsert_run({
            "session_id": "test-role-audit",
            "q_raw": "Role audit verification",
            "q_final": "Role audit verification",
            "category": "general",
            "jury_roster": [],
            "role_map": {"J1": "PROPOSER"},
            "final": {},
        })

        round_row = write_round(run, {
            "round": 1,
            "agreement": 1.0,
            "majority_label": "YES",
            "outputs": [{
                "juror_id": "J1",
                "model_id": "",
                "role": "PROPOSER",
                "role_instruction": (
                    ROLE_INSTRUCTIONS["PROPOSER"]
                ),
                "role_instruction_version": (
                    ROLE_INSTRUCTIONS_VERSION
                ),
                "juror_prompt_version": (
                    JUROR_PROMPT_VERSION
                ),
                "verdict_label": "YES",
                "tldr": "Verification output.",
                "reasoning": [
                    "Test one",
                    "Test two",
                    "Test three",
                ],
                "status": "OK",
                "schema_valid": True,
            }],
        })

        response = JurorResponse.objects.get(
            round=round_row,
            juror_id="J1",
        )

        self.assertEqual(
            response.role,
            "PROPOSER",
        )
        self.assertEqual(
            response.role_instruction,
            ROLE_INSTRUCTIONS["PROPOSER"],
        )
        self.assertEqual(
            response.role_instruction_version,
            ROLE_INSTRUCTIONS_VERSION,
        )
        self.assertEqual(
            response.juror_prompt_version,
            JUROR_PROMPT_VERSION,
        )