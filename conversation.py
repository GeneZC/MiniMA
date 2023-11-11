"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    PHOENIX = auto()
    MINICHAT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # System prompts
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separator
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    # Used for the state in the gradio servers.
    # TODO(lmzheng): refactor this
    conv_id: Any = None
    skip_next: bool = False
    model_name: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.BAIZE:
            ret = self.system + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + message + "\n"
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.MINICHAT:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + " " + message + "</s>"
                else:
                    ret += role # No space is needed.
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            conv_id=self.conv_id,
            model_name=self.model_name,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
        }


conv_vicuna = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

conv_baize = Conversation(
    system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
    roles=("[|Human|]", "[|AI|]"),
    messages=(
        ("[|Human|]", "Hello!"),
        ("[|AI|]", "Hi!"),
    ),
    offset=2,
    sep_style=SeparatorStyle.BAIZE,
    sep="\n",
    stop_str="[|Human|]",
)

conv_phoenix = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PHOENIX,
    sep="</s>",
)

conv_chatgpt = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=None,
    sep=None,
)

conv_minichat = Conversation(
    system="‘MiniChat’是一个由‘Beccurio’开发的AI语言模型。下面是人类和MiniChat之间的一段对话。MiniChat的回复应当尽可能详细，并且以Markdown的形式输出。MiniChat应当拒绝参与违背伦理的讨论。</s>",
    roles=("[|User|]", "[|Assistant|]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MINICHAT,
    sep="</s>",
)


conv_templates = {
    "vicuna": conv_vicuna,
    "baize": conv_baize,
    "phoenix": conv_phoenix,
    "chatgpt": conv_chatgpt,
    "minichat": conv_minichat,
}

def get_default_conv_template(model_name):
    model_name = model_name.lower()
    try:
        ret = conv_templates[model_name]
        return ret.copy()
    except:
        raise NotImplementedError(f"No support for model {model_name}.")
    

if __name__ == "__main__":
    conv = conv_templates["minichat"].copy()
    conv.append_message(conv.roles[0], "Write a Python function that checks if a given number is even or odd.")
    conv.append_message(conv.roles[1], None)
    print([conv.get_prompt()])
