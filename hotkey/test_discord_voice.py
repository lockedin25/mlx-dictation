"""Offline tests for discord_voice.py. No network, no credentials.

Run: python3 -B -m unittest hotkey/test_discord_voice.py  (from the repo root)
"""
import email.message
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import discord_voice as dv  # noqa: E402

GUILD = "100000000000000003"
USER = "100000000000000001"
PEER = "100000000000000002"
CHANNEL = "200000000000000001"
OTHER_CHANNEL = "200000000000000002"
TOKEN = "test-token-DO-NOT-LEAK"

GOOD_ENV = {
    "DISCORD_TOKEN": TOKEN,
    "DISCORD_GUILD_ID": GUILD,
    "DISCORD_USER_ID": USER,
    "DISCORD_PEER_ID": PEER,
}


def make_config(**overrides):
    cfg = dv.load_config(GOOD_ENV)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class FakeClient:
    """Replays a script of per-poll results: (user_channel, peer_channel) or an exception."""

    def __init__(self, script):
        self._script = list(script)
        self._pending = None

    def voice_channel(self, guild_id, user_id):
        if self._pending is None:
            step = self._script.pop(0)
            if isinstance(step, Exception):
                raise step
            self._pending = step
        user_channel, peer_channel = self._pending
        if user_id == USER:
            self._pending = (user_channel, peer_channel)
            return user_channel
        self._pending = None
        return peer_channel


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def http_error(code, headers=None):
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError("https://discord.com/api/v10/x", code, "err", message,
                                  io.BytesIO(b"{}"))


class ConfigTests(unittest.TestCase):
    def test_loads_required_values_and_defaults(self):
        cfg = dv.load_config(GOOD_ENV)
        self.assertEqual(cfg.guild_id, GUILD)
        self.assertEqual(cfg.user_id, USER)
        self.assertEqual(cfg.peer_id, PEER)
        self.assertIsNone(cfg.channel_id)
        self.assertEqual(cfg.poll_seconds, dv.DEFAULT_POLL_SECONDS)
        self.assertEqual(cfg.leave_grace_seconds, dv.DEFAULT_LEAVE_GRACE_SECONDS)

    def test_token_is_redacted_in_repr(self):
        self.assertNotIn(TOKEN, repr(dv.load_config(GOOD_ENV)))

    def test_channel_can_be_identified_without_peer(self):
        env = {
            "DISCORD_TOKEN": "test-token", "DISCORD_GUILD_ID": GUILD,
            "DISCORD_USER_ID": USER, "DISCORD_CHANNEL_ID": CHANNEL,
        }
        self.assertIsNone(dv.load_config(env).peer_id)
        env.pop("DISCORD_CHANNEL_ID")
        with self.assertRaises(dv.ConfigError):
            dv.load_config(env)

    def test_missing_token_rejected(self):
        env = dict(GOOD_ENV, DISCORD_TOKEN="  ")
        with self.assertRaisesRegex(dv.ConfigError, "DISCORD_TOKEN"):
            dv.load_config(env)

    def test_bad_snowflake_rejected(self):
        for bad in ["abc", "123", "1" * 21, "12345678901234567x"]:
            with self.subTest(bad=bad):
                with self.assertRaises(dv.ConfigError):
                    dv.load_config(dict(GOOD_ENV, DISCORD_GUILD_ID=bad))

    def test_optional_channel_validated(self):
        cfg = dv.load_config(dict(GOOD_ENV, DISCORD_CHANNEL_ID=CHANNEL))
        self.assertEqual(cfg.channel_id, CHANNEL)
        with self.assertRaises(dv.ConfigError):
            dv.load_config(dict(GOOD_ENV, DISCORD_CHANNEL_ID="nope"))

    def test_user_and_peer_must_differ(self):
        with self.assertRaises(dv.ConfigError):
            dv.load_config(dict(GOOD_ENV, DISCORD_PEER_ID=USER))

    def test_poll_and_grace_bounds(self):
        cfg = dv.load_config(dict(GOOD_ENV, DISCORD_VOICE_POLL_SECONDS="5",
                                  DISCORD_VOICE_LEAVE_GRACE_SECONDS="0"))
        self.assertEqual(cfg.poll_seconds, 5.0)
        self.assertEqual(cfg.leave_grace_seconds, 0.0)
        with self.assertRaises(dv.ConfigError):
            dv.load_config(dict(GOOD_ENV, DISCORD_VOICE_POLL_SECONDS="0.1"))
        with self.assertRaises(dv.ConfigError):
            dv.load_config(dict(GOOD_ENV, DISCORD_VOICE_LEAVE_GRACE_SECONDS="-1"))
        with self.assertRaises(dv.ConfigError):
            dv.load_config(dict(GOOD_ENV, DISCORD_VOICE_POLL_SECONDS="fast"))


class SharedChannelTests(unittest.TestCase):
    def test_shared_channel_rules(self):
        self.assertEqual(dv.shared_channel(CHANNEL, CHANNEL, None), CHANNEL)
        self.assertIsNone(dv.shared_channel(CHANNEL, OTHER_CHANNEL, None))
        self.assertIsNone(dv.shared_channel(None, CHANNEL, None))
        self.assertIsNone(dv.shared_channel(CHANNEL, None, None))
        self.assertIsNone(dv.shared_channel(None, None, None))
        self.assertEqual(dv.shared_channel(CHANNEL, CHANNEL, CHANNEL), CHANNEL)
        self.assertIsNone(dv.shared_channel(OTHER_CHANNEL, OTHER_CHANNEL, CHANNEL))


class SessionTrackerTests(unittest.TestCase):
    def test_join_once_then_leave_after_grace(self):
        t = dv.SessionTracker(60)
        self.assertEqual(t.observe(0, CHANNEL), [{"event": "join", "channel_id": CHANNEL}])
        self.assertEqual(t.observe(2, CHANNEL), [])
        self.assertEqual(t.observe(10, None), [])
        self.assertEqual(t.observe(69, None), [])
        self.assertEqual(t.observe(70, None), [{"event": "leave", "channel_id": CHANNEL}])
        self.assertIsNone(t.active_channel)

    def test_reconnect_within_grace_keeps_session(self):
        t = dv.SessionTracker(60)
        t.observe(0, CHANNEL)
        self.assertEqual(t.observe(10, None), [])
        self.assertEqual(t.observe(30, CHANNEL), [])  # back: no leave, no second join
        self.assertEqual(t.observe(100, None), [])    # grace restarts from here
        self.assertEqual(t.observe(160, None), [{"event": "leave", "channel_id": CHANNEL}])

    def test_move_emits_leave_then_join(self):
        t = dv.SessionTracker(60)
        t.observe(0, CHANNEL)
        self.assertEqual(t.observe(5, OTHER_CHANNEL), [
            {"event": "leave", "channel_id": CHANNEL},
            {"event": "join", "channel_id": OTHER_CHANNEL},
        ])

    def test_zero_grace_leaves_on_first_absence(self):
        t = dv.SessionTracker(0)
        t.observe(0, CHANNEL)
        self.assertEqual(t.observe(1, None), [{"event": "leave", "channel_id": CHANNEL}])

    def test_no_events_without_session(self):
        t = dv.SessionTracker(60)
        self.assertEqual(t.observe(0, None), [])


class SignalerTests(unittest.TestCase):
    def run_signaler(self, script, grace=60.0, max_polls=None):
        if max_polls is None:
            max_polls = len(script)
        cfg = make_config(leave_grace_seconds=grace, poll_seconds=2.0)
        out, err = io.StringIO(), io.StringIO()
        clock = FakeClock()
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock.now += seconds

        signaler = dv.VoiceSignaler(cfg, FakeClient(script), out, err,
                                    clock=clock, sleep=fake_sleep)
        code = signaler.run(max_polls=max_polls)
        return code, out.getvalue(), err.getvalue(), sleeps

    def test_confirmed_leave_only_after_grace_period(self):
        script = [
            (CHANNEL, CHANNEL),       # t=1000 join
            (None, CHANNEL),          # t=1002 absent; grace starts
            (None, None),             # t=1004 still inside 60s grace: no leave
            (None, None),             # t=1006 ...
        ]
        # Clock advances 2s per poll, so the 4s grace is crossed on the fourth poll (t=1006).
        code, out, err, _ = self.run_signaler(script, grace=4.0)
        self.assertEqual(code, dv.EXIT_OK)
        self.assertEqual([json.loads(line)["event"] for line in out.splitlines()],
                         ["join", "leave"])
        self.assertNotIn(TOKEN, err)

    def test_stdout_contract_exact(self):
        # Grace of 0 keeps the script short and deterministic.
        script = [(CHANNEL, CHANNEL), (CHANNEL, CHANNEL), (None, None)]
        code, out, err, _ = self.run_signaler(script, grace=0.0)
        self.assertEqual(code, dv.EXIT_OK)
        self.assertEqual(out.splitlines(), [
            json.dumps({"event": "join", "channel_id": CHANNEL}),
            json.dumps({"event": "leave", "channel_id": CHANNEL}),
        ])
        self.assertIn("join channel", err)
        self.assertIn("leave channel", err)

    def test_transient_failures_never_emit_leave(self):
        script = [
            (CHANNEL, CHANNEL),                      # join
            dv.TransientError("timeout"),            # unknown, must not count as absence
            (CHANNEL, CHANNEL),                      # still there
            dv.TransientError("rate limited", 5.0),
            (CHANNEL, CHANNEL),
        ]
        code, out, err, sleeps = self.run_signaler(script, grace=0.0)
        self.assertEqual(code, dv.EXIT_OK)
        self.assertEqual(out.splitlines(), [json.dumps({"event": "join", "channel_id": CHANNEL})])
        self.assertIn("retrying", err)
        self.assertIn("polling recovered", err)
        self.assertIn(5.0, sleeps)  # Retry-After honoured

    def test_sustained_outage_stops_automatic_capture_owner(self):
        code, out, err, _ = self.run_signaler([
            (CHANNEL, CHANNEL), dv.TransientError("timeout"), dv.TransientError("timeout"),
        ])
        self.assertEqual(code, dv.EXIT_FATAL)
        self.assertEqual(out.splitlines(), [json.dumps({"event": "join", "channel_id": CHANNEL})])
        self.assertIn("voice state unavailable", err)

    def test_backoff_grows_and_is_capped(self):
        script = [dv.TransientError("down")] * 7
        _, _, _, sleeps = self.run_signaler(script, max_polls=7)
        self.assertEqual(sleeps[0], 4.0)  # poll 2s * 2**1
        self.assertLessEqual(max(sleeps), dv.MAX_BACKOFF_SECONDS)
        self.assertEqual(sleeps[-1], dv.MAX_BACKOFF_SECONDS)

    def test_fatal_error_exits_without_events(self):
        code, out, err, _ = self.run_signaler([dv.FatalError("bad token")])
        self.assertEqual(code, dv.EXIT_FATAL)
        self.assertEqual(out, "")
        self.assertIn("fatal: bad token", err)

    def test_shared_channel_filter_ignores_other_channels(self):
        cfg = make_config(channel_id=CHANNEL, leave_grace_seconds=0.0)
        out, err = io.StringIO(), io.StringIO()
        signaler = dv.VoiceSignaler(cfg, FakeClient([(OTHER_CHANNEL, OTHER_CHANNEL)]), out, err,
                                    sleep=lambda _s: None)
        self.assertEqual(signaler.run(max_polls=1), dv.EXIT_OK)
        self.assertEqual(out.getvalue(), "")

    def test_stop_request_emits_no_leave(self):
        cfg = make_config(leave_grace_seconds=0.0)
        out, err = io.StringIO(), io.StringIO()
        stops = iter([False, False, True])
        signaler = dv.VoiceSignaler(cfg, FakeClient([(CHANNEL, CHANNEL), (CHANNEL, CHANNEL)]),
                                    out, err, sleep=lambda _s: None,
                                    should_stop=lambda: next(stops))
        self.assertEqual(signaler.run(), dv.EXIT_OK)
        self.assertEqual(out.getvalue().count('"leave"'), 0)
        self.assertIn("no leave emitted", err.getvalue())


class RestClientTests(unittest.TestCase):
    def make_client(self, responder):
        calls = []

        def opener(request, timeout):
            calls.append(request)
            return responder(request)

        return dv.DiscordREST(TOKEN, opener=opener), calls

    @staticmethod
    def ok(payload):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response(json.dumps(payload).encode())

    def test_voice_channel_parsed_and_auth_header_sent(self):
        client, calls = self.make_client(lambda req: self.ok({"channel_id": CHANNEL}))
        self.assertEqual(client.voice_channel(GUILD, USER), CHANNEL)
        self.assertEqual(calls[0].get_header("Authorization"), f"Bot {TOKEN}")
        self.assertTrue(calls[0].full_url.endswith(f"/guilds/{GUILD}/voice-states/{USER}"))
        self.assertEqual(calls[0].get_method(), "GET")

    def test_not_connected_is_none(self):
        def responder(req):
            raise http_error(404)
        client, _ = self.make_client(responder)
        self.assertIsNone(client.voice_channel(GUILD, USER))

    def test_null_channel_is_none(self):
        client, _ = self.make_client(lambda req: self.ok({"channel_id": None}))
        self.assertIsNone(client.voice_channel(GUILD, USER))

    def test_429_is_transient_with_retry_after(self):
        def responder(req):
            raise http_error(429, {"Retry-After": "7"})
        client, _ = self.make_client(responder)
        with self.assertRaises(dv.TransientError) as ctx:
            client.voice_channel(GUILD, USER)
        self.assertEqual(ctx.exception.retry_after, 7.0)

    def test_5xx_and_network_errors_are_transient(self):
        def server_error(req):
            raise http_error(503)

        def network_error(req):
            raise urllib.error.URLError("no route")

        for responder in (server_error, network_error):
            client, _ = self.make_client(responder)
            with self.subTest(responder=responder.__name__):
                with self.assertRaises(dv.TransientError):
                    client.voice_channel(GUILD, USER)

    def test_auth_error_is_fatal_and_token_not_in_message(self):
        def responder(req):
            raise http_error(401)
        client, _ = self.make_client(responder)
        with self.assertRaises(dv.FatalError) as ctx:
            client.voice_channel(GUILD, USER)
        self.assertNotIn(TOKEN, str(ctx.exception))

    def test_check_guild_maps_404_to_fatal(self):
        def responder(req):
            raise http_error(404)
        client, _ = self.make_client(responder)
        with self.assertRaises(dv.FatalError):
            client.check_guild(GUILD)


class MainTests(unittest.TestCase):
    def test_config_error_exit_code(self):
        err = io.StringIO()
        code = dv.main([], env={}, out=io.StringIO(), err=err)
        self.assertEqual(code, dv.EXIT_CONFIG)
        self.assertIn("DISCORD_TOKEN", err.getvalue())

    def test_check_mode_with_fake_guild_access(self):
        original = dv.DiscordREST.check_guild
        dv.DiscordREST.check_guild = lambda self, guild_id: None
        try:
            err = io.StringIO()
            code = dv.main(["--check"], env=GOOD_ENV, out=io.StringIO(), err=err)
        finally:
            dv.DiscordREST.check_guild = original
        self.assertEqual(code, dv.EXIT_OK)
        self.assertNotIn(TOKEN, err.getvalue())


if __name__ == "__main__":
    unittest.main()
