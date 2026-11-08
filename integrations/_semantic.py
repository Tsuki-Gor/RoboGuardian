"""Small OpenAI-compatible visual assessor; no SDK, retries, or implicit model."""

import base64
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from roboguardian import Signal
from roboguardian.types import json_data


def write_json(path, value, secret=""):
    text = json.dumps(json_data(value), ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(text.replace(secret, "[REDACTED]") if secret else text, encoding="utf-8")


class SemanticClient:
    def __init__(self, base_url, api_key, model, output, *, timeout=120,
                 temperature=0.2, max_tokens=800, image_detail="low"):
        if not all(isinstance(value, str) and value.strip() for value in (base_url, api_key, model)):
            raise ValueError("explicit endpoint, API key, and model are required")
        if timeout <= 0 or max_tokens < 1 or image_detail not in ("low", "high", "auto"):
            raise ValueError("invalid semantic request parameters")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.secret, self.model, self.timeout = api_key, model, timeout
        self.temperature, self.max_tokens, self.image_detail = temperature, max_tokens, image_detail
        self.output = Path(output) / "model_requests"
        self.output.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def assess(self, question, observation):
        images = observation.images[:1]
        if not images:
            raise ValueError("visual assessment requires a current image")
        image_roles = observation.state.get("image_roles")
        if image_roles is not None:
            current_roles = ("CURRENT agent view", "CURRENT wrist view")
            roles = tuple(image_roles) if isinstance(image_roles, (list, tuple)) else ()
            if (roles not in (current_roles, (*current_roles, "ORIGINAL BEFORE reference"))
                    or len(roles) != len(observation.images)):
                raise ValueError("image_roles must match the current agent/wrist views and optional original-before reference")
            if len(roles) == 3 and not observation.state.get("visual_reference_before"):
                raise ValueError("the original-before image role requires its visual reference metadata")
            images = observation.images
            evidence = " ".join(f"Image {index} is the {role}." for index, role in enumerate(roles, 1))
            if len(roles) == 3:
                evidence += " Judge task completion from the two CURRENT views; use the ORIGINAL BEFORE reference only as visual context. "
            else:
                evidence += " Judge task completion from both CURRENT views. "
        else:
            evidence = "Image 1 is the current observation. "
            if observation.state.get("visual_reference_before") and len(observation.images) > 1:
                images = observation.images[:2]
                evidence += ("Image 2 is the original observation BEFORE the same action, provided only as a visual reference. "
                             "Judge the current state, using the reference only to compare visible changes. ")
        prompt = (question + "\n" + evidence + "If occluded or ambiguous, abstain. "
                  'Return exactly {"judgment":"pass|fail|indeterminate","reason":"brief visual evidence"}.')
        content = [{"type": "text", "text": prompt}]
        for path in images:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded, "detail": self.image_detail}})
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": "Assess the supplied visual evidence. Return only the requested JSON object."},
                             {"role": "user", "content": content}]}
        self.calls += 1
        prefix = self.output / f"{self.calls:04d}-semantic"
        write_json(prefix.with_suffix(".request.json"), body, self.secret)
        started = time.monotonic()
        try:
            request = Request(self.endpoint, data=json.dumps(body, allow_nan=False).encode(),
                              headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.secret}, method="POST")
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            write_json(prefix.with_suffix(".response.json"), {"elapsed_seconds": time.monotonic() - started,
                       "response": value}, self.secret)
            text = value["choices"][0]["message"]["content"].strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            judgment = json.loads(text)
            if (not isinstance(judgment, dict) or set(judgment) != {"judgment", "reason"}
                    or not isinstance(judgment["reason"], str) or not judgment["reason"].strip()):
                raise ValueError("semantic response must contain exactly judgment and a nonempty reason")
            return Signal(judgment["judgment"])
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}".replace(self.secret, "[REDACTED]")
            write_json(prefix.with_suffix(".error.json"), {"error": reason, "elapsed_seconds": time.monotonic() - started})
            raise RuntimeError(reason) from None
