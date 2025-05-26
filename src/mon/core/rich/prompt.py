#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extend ``rich.prompt.Prompt``."""

import rich
from rich.columns import Columns
from rich.prompt import *

from mon.core import type_extensions


# ----- Prompt Class -----
class SelectionOrInputPrompt(Prompt):
    """Extend ``rich.prompt.Prompt`` to allow for either selecting an index or
    directly entering value.

    Args:
        prompt: Prompt text. Defaults to "".
        console: A Console instance or None to use global console. Defaults to None.
        password: Enable password input. Defaults to False.
        choices: A list of valid choices. Defaults to None.
        case_sensitive: Matching of choices should be case-sensitive. Defaults to True.
        show_default: Show default in prompt. Defaults to True.
        show_choices: Show choices in prompt. Defaults to True.
        allow_empty: Allow empty input. Defaults to False.
        column_first: Align Align items from top to bottom (rather than left to right).
            Defaults to False.
        stream: Optional text file open for reading to get input. Defaults to None.
    """
    
    response_type: type = str

    def __init__(
        self,
        prompt        : TextType            = "",
        *,
        console       : Optional[Console]   = None,
        password      : bool                = False,
        choices       : Optional[List[str]] = None,
        case_sensitive: bool                = True,
        show_default  : bool                = True,
        show_choices  : bool                = True,
        column_first  : bool                = False,
        allow_empty   : bool                = False,
    ):
        self.allow_empty  = allow_empty
        self.column_first = column_first
        super().__init__(
            prompt         = prompt,
            console        = console,
            password       = password,
            choices        = choices,
            case_sensitive = case_sensitive,
            show_default   = show_default,
            show_choices   = show_choices,
        )

    def print_choices(self):
        """Print columns of choices to the console."""
        choices_ = []
        for i, choice in enumerate(self.choices):
            choices_.append(f"{f'{i}.':>6} {choice}")
        columns = Columns(choices_, equal=True, column_first=self.column_first)
        rich.print(columns)

    @classmethod
    def ask(
        cls,
        prompt        : TextType            = "",
        *,
        console       : Optional[Console]   = None,
        password      : bool                = False,
        choices       : Optional[List[str]] = None,
        case_sensitive: bool                = True,
        show_default  : bool                = True,
        show_choices  : bool                = True,
        allow_empty   : bool                = False,
        column_first  : bool                = False,
        default       : Any                 = ...,
        stream        : Optional[TextIO]    = None,
    ) -> Any:
        """Shortcut to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt: Prompt text. Defaults to "".
            console: A Console instance or None to use global console. Defaults to None.
            password: Enable password input. Defaults to False.
            choices: A list of valid choices. Defaults to None.
            case_sensitive: Matching of choices should be case-sensitive. Defaults to True.
            show_default: Show default in prompt. Defaults to True.
            show_choices: Show choices in prompt. Defaults to True.
            allow_empty: Allow empty input. Defaults to False.
            column_first: Align Align items from top to bottom (rather than left to right).
                Defaults to False.
            default: Default value to return if no input is given. Defaults to ``...``.
            stream: Optional text file open for reading to get input. Defaults to None.
        """
        _prompt = cls(
            prompt,
            console        = console,
            password       = password,
            choices        = choices,
            case_sensitive = case_sensitive,
            show_default   = show_default,
            show_choices   = show_choices,
            allow_empty    = allow_empty,
            column_first   = column_first,
        )
        return _prompt(default=default, stream=stream)

    def render_default(self, default: DefaultType) -> Text:
        """Turn the supplied default in to a Text instance.

        Args:
            default: Default value.

        Returns:
            Text containing rendering of default value.
        """
        return Text(f"[{default}]", "prompt.default")
    
    def make_prompt(self, default: DefaultType) -> Text:
        """Make prompt text.

        Args:
            default: Default value.

        Returns:
            Text to display in prompt.
        """
        if self.show_choices and self.choices and len(self.choices) > 0:
            rich.print(self.prompt)
            self.print_choices()
            prompt = Text.from_markup("", style="prompt")
        else:
            prompt = self.prompt.copy()
        prompt.end = ""
        
        if (
            default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt
    
    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value: Value entered by user.

        Returns:
            ``True`` if choice was valid, otherwise ``False``.
        """
        assert self.choices is not None
        if self.case_sensitive:
            return value in self.choices
        return value.lower() in [choice.lower() for choice in self.choices]
    
    def process_response(self, value: str) -> PromptType:
        """Process response from user, convert to prompt type.

        Args:
            value: String typed by user.

        Raises:
            If ``value`` is invalid.

        Returns:
            The value to be returned from ask method.
        """
        value = value.strip() if isinstance(value, str) else value

        if self.choices is not None:
            if len(self.choices) == 0:
                return value
            if len(self.choices) > 0 and value == "" and not self.allow_empty:
                raise InvalidResponse(self.illegal_choice_message)
            # If the whole value is a choice, return it
            if value in self.choices:
                return value

            # Convert index (if any) to choice
            value = type_extensions.to_list(value, sep=[",", ";"])
            if any(v for v in value if type_extensions.is_int(v) and not 0 <= int(v) <= len(self.choices) - 1):
                raise InvalidResponse(self.illegal_choice_message)
            value = [self.choices[int(v)] if type_extensions.is_int(v) else v for v in value]
            
            '''
            for i, v in enumerate(value):
                if not self.check_choice(v):
                    raise InvalidResponse(self.illegal_choice_message)
                if not self.case_sensitive:
                    # return the original choice, not the lower case version
                    value[i] = self.choices[[choice.lower() for choice in self.choices].index(v.lower())]
            '''
            # value = value[0] if len(value) == 1 else value
            
        return value
    
    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Run the prompt loop.

        Args:
            default (Any, optional): Optional default value.

        Returns:
            PromptType: Processed value.
        """
        while True:
            self.pre_prompt()
            prompt = self.make_prompt(default)
            value  = self.get_input(self.console, prompt, self.password, stream=stream)
            if value == "" and default != ...:
                # return default
                value = default
            try:
                return_value = self.process_response(value)
            except InvalidResponse as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value
