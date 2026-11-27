"""Plug-and-play LLM provider abstraction for meeting translation, classification, and notes.

Uses Python standard library only. Supports OpenRouter (default, optional ZDR),
any OpenAI-compatible HTTP endpoint (including local Ollama / mlx_lm.server),
Anthropic Messages API, Google Gemini API, and ephemeral local MLX subprocesses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LLMError(RuntimeError):
    """Problem talking to an LLM provider. Messages never contain API keys."""


def read_env_key(credentials_file: str | None, env_var: str) -> str:
    """Read `env_var` from process env first, then from an env-style file."""
    direct = os.environ.get(env_var, "").strip()
    if direct:
        return direct
    if not credentials_file:
        raise LLMError(f"{env_var} is not set and no credentials_file was configured")
    path = Path(credentials_file).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMError(f"Credentials file not readable: {path}") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].strip()
        name, sep, value = stripped.partition("=")
        if sep and name.strip() == env_var:
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    raise LLMError(f"{env_var} is not set in the credentials file")


def _urllib_post(url: str, body: bytes, headers: dict, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _is_localhost_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1")


def _normalize_openai_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1") or cleaned.endswith("/api/v1"):
        return f"{cleaned}/chat/completions"
    return base_url.strip()


class LLMProvider:
    """Base interface implemented by all chat providers."""

    provider_name: str = "base"
    model: str = ""

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """Chat completions via any OpenAI-compatible HTTP endpoint (OpenRouter, Ollama, vLLM, etc.)."""

    def __init__(
        self,
        model: str,
        base_url: str = OPENROUTER_URL,
        api_key: str | None = None,
        zdr: bool = False,
        extra_headers: dict | None = None,
        transport=None,
        retries: int = 2,
        backoff: float = 2.0,
        timeout: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
        provider_name: str = "openai_compatible",
    ):
        self.model = model
        self.base_url = _normalize_openai_url(base_url)
        self.api_key = api_key or None
        self.zdr = bool(zdr)
        self.extra_headers = dict(extra_headers or {})
        self.transport = transport or _urllib_post
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.sleep = sleep
        self.provider_name = provider_name

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if self.zdr:
            body["provider"] = {"zdr": True}
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Title": "fluid-speech-pro meeting pipeline",
        }
        for key, value in self.extra_headers.items():
            headers[str(key)] = str(value)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.retries + 1):
            try:
                raw = self.transport(self.base_url, payload, headers, self.timeout)
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    return "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
                if content is None:
                    raise ValueError("empty content")
                return str(content)
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 429) or exc.code >= 500
                if not retryable or attempt == self.retries:
                    raise LLMError(f"{self.provider_name} request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise LLMError(f"{self.provider_name} request failed (network or timeout)") from None
            except (KeyError, IndexError, TypeError, ValueError):
                raise LLMError(f"{self.provider_name} returned an unexpected response") from None
            self.sleep(self.backoff * (attempt + 1))
        raise LLMError(f"{self.provider_name} request failed")


class OpenRouter(OpenAICompatibleProvider):
    """OpenRouter chat completions client (backward-compatible constructor)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        zdr: bool,
        transport=None,
        retries: int = 2,
        backoff: float = 2.0,
        timeout: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(
            model=model,
            base_url=OPENROUTER_URL,
            api_key=api_key,
            zdr=zdr,
            transport=transport,
            retries=retries,
            backoff=backoff,
            timeout=timeout,
            sleep=sleep,
            provider_name="OpenRouter",
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = ANTHROPIC_URL,
        max_tokens: int = 4096,
        extra_headers: dict | None = None,
        transport=None,
        retries: int = 2,
        backoff: float = 2.0,
        timeout: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider_name = "Anthropic"
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.strip()
        self.max_tokens = int(max_tokens)
        self.extra_headers = dict(extra_headers or {})
        self.transport = transport or _urllib_post
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.sleep = sleep

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        body = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        for key, value in self.extra_headers.items():
            headers[str(key)] = str(value)

        for attempt in range(self.retries + 1):
            try:
                raw = self.transport(self.base_url, payload, headers, self.timeout)
                data = json.loads(raw)
                blocks = data["content"]
                text = "".join(
                    str(b["text"]) for b in blocks if isinstance(b, dict) and b.get("type") == "text"
                )
                if not text:
                    raise ValueError("empty content")
                return text
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 429) or exc.code >= 500
                if not retryable or attempt == self.retries:
                    raise LLMError(f"Anthropic request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise LLMError("Anthropic request failed (network or timeout)") from None
            except (KeyError, IndexError, TypeError, ValueError):
                raise LLMError("Anthropic returned an unexpected response") from None
            self.sleep(self.backoff * (attempt + 1))
        raise LLMError("Anthropic request failed")


class GeminiProvider(LLMProvider):
    """Google Gemini Generative Language API provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = GEMINI_BASE_URL,
        max_tokens: int = 4096,
        extra_headers: dict | None = None,
        transport=None,
        retries: int = 2,
        backoff: float = 2.0,
        timeout: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider_name = "Gemini"
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.strip().rstrip("/")
        self.max_tokens = int(max_tokens)
        self.extra_headers = dict(extra_headers or {})
        self.transport = transport or _urllib_post
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.sleep = sleep

    def endpoint_url(self) -> str:
        if ":generateContent" in self.base_url:
            return self.base_url
        return f"{self.base_url}/models/{self.model}:generateContent"

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        body = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.max_tokens,
            },
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        for key, value in self.extra_headers.items():
            headers[str(key)] = str(value)

        url = self.endpoint_url()
        for attempt in range(self.retries + 1):
            try:
                raw = self.transport(url, payload, headers, self.timeout)
                data = json.loads(raw)
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
                if not text:
                    raise ValueError("empty content")
                return text
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 429) or exc.code >= 500
                if not retryable or attempt == self.retries:
                    raise LLMError(f"Gemini request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise LLMError("Gemini request failed (network or timeout)") from None
            except (KeyError, IndexError, TypeError, ValueError):
                raise LLMError("Gemini returned an unexpected response") from None
            self.sleep(self.backoff * (attempt + 1))
        raise LLMError("Gemini request failed")


_MLX_SUBPROCESS_SCRIPT = """
import json, sys
from mlx_lm import generate, load
req = json.loads(sys.stdin.read())
model, tokenizer = load(req["model"])
messages = [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["user"]}]
if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    prompt = req["system"] + "\\n\\n" + req["user"]
out = generate(model, tokenizer, prompt=prompt, max_tokens=int(req.get("max_tokens", 2048)), verbose=False)
print(json.dumps({"text": out}), flush=True)
"""


class LocalMLXProvider(LLMProvider):
    """Ephemeral local mlx-lm subprocess provider (unloads all weights on exit)."""

    def __init__(
        self,
        model: str,
        python_path: str | Path | None = None,
        max_tokens: int = 2048,
        timeout: float = 300.0,
        runner=None,
    ):
        self.provider_name = "mlx_local"
        self.model = model
        resolved = Path(python_path).expanduser() if python_path else (PROJECT_ROOT / ".venv-cleanup/bin/python")
        self.python_path = str(resolved) if resolved.is_file() else sys.executable
        self.max_tokens = int(max_tokens)
        self.timeout = timeout
        self.runner = runner

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        payload = json.dumps({
            "model": self.model,
            "system": system,
            "user": user,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        })
        cmd = [self.python_path, "-c", _MLX_SUBPROCESS_SCRIPT]
        if self.runner is not None:
            return self.runner(cmd, payload, self.timeout)
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise LLMError("Local MLX model timed out") from None
        except OSError as exc:
            raise LLMError(f"Failed to start local MLX process: {exc}") from None
        if proc.returncode != 0:
            raise LLMError(f"Local MLX model failed (exit {proc.returncode})")
        try:
            last_line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
            data = json.loads(last_line)
            return str(data["text"])
        except (IndexError, KeyError, TypeError, ValueError):
            raise LLMError("Local MLX returned an unexpected response") from None


def create_llm_provider(config: dict, *, transport=None, runner=None) -> LLMProvider:
    """Build an LLMProvider from a config dictionary."""
    provider = str(config.get("provider") or "openrouter").strip().lower()
    model = str(config.get("model") or "openai/gpt-4.1-mini").strip()
    timeout = float(config.get("timeout_seconds", 180.0))
    extra_headers = config.get("extra_headers") if isinstance(config.get("extra_headers"), dict) else None
    max_tokens = int(config.get("max_tokens", 4096))

    if provider == "openrouter":
        base_url = str(config.get("base_url") or OPENROUTER_URL)
        env_var = str(config.get("api_key_env") or "OPENROUTER_API_KEY")
        creds = config.get("credentials_file", "~/.config/openrouter/credentials.env")
        api_key = config.get("api_key") or (None if _is_localhost_url(base_url) else read_env_key(creds, env_var))
        zdr = bool(config.get("zdr", True))
        if base_url == OPENROUTER_URL and not extra_headers and max_tokens == 4096:
            return OpenRouter(api_key or "", model, zdr, transport=transport, timeout=timeout)
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            zdr=zdr,
            extra_headers=extra_headers,
            transport=transport,
            timeout=timeout,
            provider_name="OpenRouter",
        )

    if provider in ("openai_compatible", "openai", "ollama"):
        default_url = "http://localhost:11434/v1/chat/completions" if provider == "ollama" else "https://api.openai.com/v1/chat/completions"
        base_url = str(config.get("base_url") or default_url)
        env_var = str(config.get("api_key_env") or "OPENAI_API_KEY")
        creds = config.get("credentials_file")
        api_key = config.get("api_key")
        if not api_key:
            if _is_localhost_url(base_url):
                try:
                    api_key = read_env_key(creds, env_var)
                except LLMError:
                    api_key = None
            else:
                api_key = read_env_key(creds, env_var)
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            zdr=bool(config.get("zdr", False)),
            extra_headers=extra_headers,
            transport=transport,
            timeout=timeout,
            provider_name=provider,
        )

    if provider == "anthropic":
        env_var = str(config.get("api_key_env") or "ANTHROPIC_API_KEY")
        creds = config.get("credentials_file")
        api_key = config.get("api_key") or read_env_key(creds, env_var)
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            base_url=str(config.get("base_url") or ANTHROPIC_URL),
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            transport=transport,
            timeout=timeout,
        )

    if provider == "gemini":
        env_var = str(config.get("api_key_env") or "GEMINI_API_KEY")
        creds = config.get("credentials_file", "~/.config/fluid-dictation/gemini-api-key")
        # Support either raw key file or env-style file for Gemini convenience
        api_key = config.get("api_key")
        if not api_key:
            try:
                api_key = read_env_key(creds, env_var)
            except LLMError:
                raw_path = Path(str(creds)).expanduser()
                if raw_path.is_file():
                    raw_text = raw_path.read_text(encoding="utf-8").strip()
                    if raw_text and "=" not in raw_text.splitlines()[0]:
                        api_key = raw_text.splitlines()[0].strip()
                if not api_key:
                    raise
        return GeminiProvider(
            api_key=api_key,
            model=model,
            base_url=str(config.get("base_url") or GEMINI_BASE_URL),
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            transport=transport,
            timeout=timeout,
        )

    if provider in ("mlx_local", "mlx"):
        return LocalMLXProvider(
            model=model,
            python_path=config.get("python_path"),
            max_tokens=min(max_tokens, 2048),
            timeout=timeout,
            runner=runner,
        )

    raise LLMError(f"Unsupported LLM provider: {provider!r}")
