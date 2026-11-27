#!/usr/bin/env python3
"""Discord guild voice-state start/stop signal for Fluid Dictation.

Polls the Discord REST API with a bot token from the environment and reports
when the configured user and peer share a voice channel in the configured
guild. It never reads the microphone and never infers identity from audio.

Interface
  stdout  one JSON object per line, exactly {"event": "join"|"leave", "channel_id": "<id>"}
  stderr  human-readable status and error lines prefixed with "[discord-voice]"
  exit    0 on clean shutdown (SIGINT/SIGTERM or --check), 2 on configuration
          error, 3 on a fatal Discord error (bad token, no guild access)

Environment
  DISCORD_TOKEN                   bot token (required; never printed)
  DISCORD_GUILD_ID                guild snowflake (required)
  DISCORD_USER_ID                 the user being watched, e.g. Shubham (required)
  DISCORD_PEER_ID                 optional participant filter
  DISCORD_CHANNEL_ID              restrict signals to this channel (optional)
  DISCORD_VOICE_POLL_SECONDS      poll interval, default 2, minimum 1
  DISCORD_VOICE_LEAVE_GRACE_SECONDS  absence required before a leave, default 60

Semantics
  join   emitted on the first successful poll where user and peer share a
         channel (and it matches DISCORD_CHANNEL_ID when set).
  leave  emitted only after a successful poll shows the user is no longer
         sharing a channel with the peer and that absence has lasted for the
         grace period. Failed polls never count as absence.
  move   a shared channel that differs from the active one emits leave for the
         old channel, then join for the new one.
  Shutdown never emits a leave: a leave is only emitted once Discord confirms
  it, so the consumer decides how to stop on process exit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Iterable, Optional, TextIO

API_BASE = "https://discord.com/api/v10"
USER_AGENT = "fluid-dictation-voice-signal/1.0"
SNOWFLAKE = re.compile(r"^\d{17,20}$")

DEFAULT_POLL_SECONDS = 2.0
MIN_POLL_SECONDS = 1.0
DEFAULT_LEAVE_GRACE_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_BACKOFF_SECONDS = 60.0

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_FATAL = 3

STATUS_PREFIX = "[discord-voice]"


class ConfigError(Exception):
    """The environment does not describe a usable configuration."""


class TransientError(Exception):
    """A poll failed in a way that may succeed later (network, 429, 5xx)."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(Exception):
    """Retrying cannot help (bad token, missing access, bad request)."""


class NotFound(Exception):
    """Discord answered 404 for the requested resource."""


class Config:
    """Validated settings. The token is excluded from repr and logs."""

    def __init__(
        self,
        token: str,
        guild_id: str,
        user_id: str,
        peer_id: Optional[str],
        channel_id: Optional[str],
        poll_seconds: float,
        leave_grace_seconds: float,
    ):
        self.token = token
        self.guild_id = guild_id
        self.user_id = user_id
        self.peer_id = peer_id
        self.channel_id = channel_id
        self.poll_seconds = poll_seconds
        self.leave_grace_seconds = leave_grace_seconds

    def __repr__(self) -> str:
        return (
            f"Config(guild_id={self.guild_id!r}, user_id=<set>, peer_id=<set>, "
            f"channel_id={self.channel_id!r}, poll_seconds={self.poll_seconds}, "
            f"leave_grace_seconds={self.leave_grace_seconds}, token=<redacted>)"
        )


def _read_float(env, name: str, default: float, minimum: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number") from None
    if value != value or value < minimum:  # NaN or below minimum
        raise ConfigError(f"{name} must be at least {minimum:g}")
    return value


def load_config(env: Optional[dict] = None) -> Config:
    env = os.environ if env is None else env
    token = (env.get("DISCORD_TOKEN") or "").strip()
    if not token:
        raise ConfigError("DISCORD_TOKEN is not set")

    ids = {}
    for name in ("DISCORD_GUILD_ID", "DISCORD_USER_ID"):
        value = (env.get(name) or "").strip()
        if not SNOWFLAKE.match(value):
            raise ConfigError(f"{name} must be a Discord ID (17-20 digits)")
        ids[name] = value

    peer = (env.get("DISCORD_PEER_ID") or "").strip() or None
    if peer is not None and not SNOWFLAKE.match(peer):
        raise ConfigError("DISCORD_PEER_ID must be a Discord ID (17-20 digits)")
    if ids["DISCORD_USER_ID"] == peer:
        raise ConfigError("DISCORD_USER_ID and DISCORD_PEER_ID must differ")

    channel = (env.get("DISCORD_CHANNEL_ID") or "").strip() or None
    if channel is not None and not SNOWFLAKE.match(channel):
        raise ConfigError("DISCORD_CHANNEL_ID must be a Discord ID (17-20 digits)")

    if peer is None and channel is None:
        raise ConfigError("Set DISCORD_CHANNEL_ID when no peer is configured")

    return Config(
        token=token,
        guild_id=ids["DISCORD_GUILD_ID"],
        user_id=ids["DISCORD_USER_ID"],
        peer_id=peer,
        channel_id=channel,
        poll_seconds=_read_float(
            env, "DISCORD_VOICE_POLL_SECONDS", DEFAULT_POLL_SECONDS, MIN_POLL_SECONDS
        ),
        leave_grace_seconds=_read_float(
            env, "DISCORD_VOICE_LEAVE_GRACE_SECONDS", DEFAULT_LEAVE_GRACE_SECONDS, 0.0
        ),
    )


class DiscordREST:
    """Minimal bot-authenticated GET client built on urllib."""

    def __init__(self, token: str, opener: Optional[Callable] = None,
                 timeout: float = REQUEST_TIMEOUT_SECONDS):
        self._token = token
        self._opener = opener or urllib.request.urlopen
        self._timeout = timeout

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(
            API_BASE + path,
            method="GET",
            headers={
                "Authorization": f"Bot {self._token}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status == 404:
                raise NotFound(path) from None
            if status == 429:
                raise TransientError("rate limited", _retry_after(exc)) from None
            if status >= 500:
                raise TransientError(f"Discord server error HTTP {status}") from None
            if status in (401, 403):
                raise FatalError(
                    f"Discord rejected the request (HTTP {status}); check DISCORD_TOKEN "
                    "and that the bot is in the guild with access to the voice channels"
                ) from None
            raise FatalError(f"Discord request failed (HTTP {status})") from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise TransientError(f"network error ({type(exc).__name__})") from None
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TransientError("Discord returned an unreadable response") from None

    def check_guild(self, guild_id: str) -> None:
        try:
            self._get(f"/guilds/{guild_id}")
        except NotFound:
            raise FatalError("guild not found or the bot is not a member") from None

    def voice_channel(self, guild_id: str, user_id: str) -> Optional[str]:
        """Return the user's voice channel id, or None when not connected."""
        try:
            data = self._get(f"/guilds/{guild_id}/voice-states/{user_id}")
        except NotFound:
            return None
        channel = data.get("channel_id")
        return str(channel) if channel else None


def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
    headers = getattr(exc, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def shared_channel(user_channel: Optional[str], peer_channel: Optional[str],
                   required_channel: Optional[str]) -> Optional[str]:
    """Return the channel the user and peer share, if it is an allowed one."""
    if user_channel is None or user_channel != peer_channel:
        return None
    if required_channel is not None and user_channel != required_channel:
        return None
    return user_channel


class SessionTracker:
    """Turns successful observations into join and leave events.

    Only successful observations are passed in, so a failed poll can never
    start or end a session.
    """

    def __init__(self, leave_grace_seconds: float):
        self._grace = leave_grace_seconds
        self.active_channel: Optional[str] = None
        self._absent_since: Optional[float] = None

    def observe(self, now: float, shared: Optional[str]) -> list:
        if shared is not None:
            self._absent_since = None
            if shared == self.active_channel:
                return []
            events = []
            if self.active_channel is not None:
                events.append(_event("leave", self.active_channel))
            events.append(_event("join", shared))
            self.active_channel = shared
            return events

        if self.active_channel is None:
            return []
        if self._absent_since is None:
            self._absent_since = now
        if now - self._absent_since >= self._grace:
            ended = self.active_channel
            self.active_channel = None
            self._absent_since = None
            return [_event("leave", ended)]
        return []


def _event(name: str, channel_id: str) -> dict:
    return {"event": name, "channel_id": channel_id}


class VoiceSignaler:
    def __init__(
        self,
        config: Config,
        client,
        out: TextIO,
        err: TextIO,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        should_stop: Callable[[], bool] = lambda: False,
    ):
        self._config = config
        self._client = client
        self._out = out
        self._err = err
        self._clock = clock
        self._sleep = sleep
        self._should_stop = should_stop
        self._tracker = SessionTracker(config.leave_grace_seconds)

    def status(self, message: str) -> None:
        print(f"{STATUS_PREFIX} {message}", file=self._err, flush=True)

    def emit(self, events: Iterable[dict]) -> None:
        for event in events:
            self._out.write(json.dumps(event) + "\n")
            self._out.flush()
            self.status(f"{event['event']} channel {event['channel_id']}")

    def poll_once(self) -> list:
        """Run one poll. Raises TransientError or FatalError."""
        user_channel = self._client.voice_channel(self._config.guild_id, self._config.user_id)
        peer_channel = user_channel
        if self._config.peer_id is not None:
            peer_channel = self._client.voice_channel(self._config.guild_id, self._config.peer_id)
        shared = shared_channel(user_channel, peer_channel, self._config.channel_id)
        return self._tracker.observe(self._clock(), shared)

    def run(self, max_polls: Optional[int] = None) -> int:
        failures = 0
        polls = 0
        self.status("watching voice state (polling Discord REST)")
        while not self._should_stop():
            if max_polls is not None and polls >= max_polls:
                break
            polls += 1
            try:
                events = self.poll_once()
            except FatalError as exc:
                self.status(f"fatal: {exc}")
                return EXIT_FATAL
            except TransientError as exc:
                failures += 1
                delay = exc.retry_after
                if delay is None:
                    delay = min(MAX_BACKOFF_SECONDS, self._config.poll_seconds * (2 ** failures))
                delay = max(delay, self._config.poll_seconds)
                if self._tracker.active_channel is not None and (failures >= 2 or delay > 10):
                    self.status("voice state unavailable during a recording")
                    return EXIT_FATAL
                self.status(f"poll failed ({exc}); state unchanged, retrying in {delay:.0f}s")
                self._sleep(delay)
                continue
            if failures:
                self.status("polling recovered")
                failures = 0
            self.emit(events)
            self._sleep(self._config.poll_seconds)

        active = self._tracker.active_channel
        if active is not None:
            self.status("stopping; no leave emitted because Discord has not confirmed one")
        else:
            self.status("stopped")
        return EXIT_OK


def main(argv: Optional[list] = None, env: Optional[dict] = None,
         out: Optional[TextIO] = None, err: Optional[TextIO] = None) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="validate config and guild access, then exit")
    args = parser.parse_args(argv)

    try:
        config = load_config(env)
    except ConfigError as exc:
        print(f"{STATUS_PREFIX} configuration error: {exc}", file=err, flush=True)
        return EXIT_CONFIG

    client = DiscordREST(config.token)
    try:
        client.check_guild(config.guild_id)
    except FatalError as exc:
        print(f"{STATUS_PREFIX} fatal: {exc}", file=err, flush=True)
        return EXIT_FATAL
    except TransientError as exc:
        print(f"{STATUS_PREFIX} guild check deferred ({exc}); continuing", file=err, flush=True)

    if args.check:
        print(f"{STATUS_PREFIX} configuration ok; guild reachable", file=err, flush=True)
        return EXIT_OK

    stop = {"requested": False}

    def request_stop(signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    signaler = VoiceSignaler(config, client, out, err,
                             should_stop=lambda: stop["requested"])
    return signaler.run()


if __name__ == "__main__":
    sys.exit(main())
