# core/agent_base.py
import re
from typing import Callable, Dict, Tuple, List
from core.llm import chat

# core/agent_base.py
ACTION_RE = re.compile(r"^Action:\s*([A-Za-z_]\w*)(?::\s*(.*))?$")


class ConversationAgent:
    def __init__(self, system_prompt: str, actions: Dict[str, Callable]):
        self.system = system_prompt
        self.actions = actions
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def _step(self, user_text: str) -> Tuple[str, List[Tuple[str, str]]]:
        self.messages.append({"role": "user", "content": user_text})
        result = chat(self.messages, temperature=0.2)
        self.messages.append({"role": "assistant", "content": result})

        actions = []
        for line in result.split("\n"):
            m = ACTION_RE.match(line.strip())
            if m:
                actions.append((m.group(1), m.group(2) or ""))  # allow empty params
        return result, actions

    def run(self, user_text: str, max_turns: int = 15):
        next_prompt = user_text
        for _ in range(max_turns):
            text, actions = self._step(next_prompt)
            if not actions:
                return text  # final answer

            # Execute declared actions in order; feed observations back in
            for name, raw in actions:
                func = self.actions.get(name)
                if not func:
                    next_prompt = f"Observation: ❌ Unknown action '{name}'"
                    break
                try:
                    observation = func(raw)
                except Exception as e:
                    observation = f"❌ {type(e).__name__}: {e}"
                next_prompt = f"Observation: {observation}"
        return "(ended)"