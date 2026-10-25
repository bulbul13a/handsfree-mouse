"""Executes controller actions on the real mouse (via pyautogui)."""
from __future__ import annotations

import logging
from typing import Iterable

from .controller import Action, ActionType

log = logging.getLogger(__name__)


class MouseDriver:
    """Thin adapter around pyautogui. ``dry_run`` only logs, never touches the mouse."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._pg = None
        if not dry_run:
            import pyautogui  # imported lazily: needs a display server

            # FAILSAFE would raise whenever the cursor touches a screen corner,
            # which is exactly where close/minimise buttons live.
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
            self._pg = pyautogui

    def execute(self, actions: Iterable[Action]) -> None:
        for action in actions:
            if action.type is ActionType.MOVE:
                if self._pg:
                    self._pg.moveRel(action.dx, action.dy)
            elif action.type is ActionType.CLICK_LEFT:
                log.debug("left click")
                if self._pg:
                    self._pg.click(button="left")
            elif action.type is ActionType.CLICK_RIGHT:
                log.debug("right click")
                if self._pg:
                    self._pg.click(button="right")
            elif action.type is ActionType.SCROLL:
                if self._pg:
                    self._pg.scroll(action.dy)
            # CONTROL_* / SCROLL_* are state notifications for the UI only.
